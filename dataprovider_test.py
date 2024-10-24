"""
Title: Automatic Speech Recognition with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2021/01/13
Last modified: 2021/01/13
Description: Training a sequence-to-sequence Transformer for automatic speech recognition.
Accelerator: GPU
Prog: transformer_asr_mltu.py
"""
"""
## Introduction

Automatic speech recognition (ASR) consists of transcribing audio speech segments into text.
ASR can be treated as a sequence-to-sequence problem, where the
audio can be represented as a sequence of feature vectors
and the text as a sequence of characters, words, or subword tokens.

For this demonstration, we will use the LJSpeech dataset from the
[LibriVox](https://librivox.org/) project. It consists of short
audio clips of a single speaker reading passages from 7 non-fiction books.
Our model will be similar to the original Transformer (both encoder and decoder)
as proposed in the paper, "Attention is All You Need".


**References:**

- [Attention is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Very Deep Self-Attention Networks for End-to-End Speech Recognition](https://arxiv.org/pdf/1904.13377.pdf)
- [Speech Transformers](https://ieeexplore.ieee.org/document/8462506)
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
"""

import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

import os
import random
from glob import glob

#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from keras import layers

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,LearningRateScheduler
#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,LearningRateScheduler


import pandas as pd
from tqdm import tqdm
import sys
import numpy as np


from mltu.preprocessors import WavReader
from mltu.tensorflow.dataProvider import DataProvider
#from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from mltu.transformers import LabelPadding, SpectrogramPadding

from configs import ModelConfigs

import matplotlib.pyplot as plt

#keras.initializers.Initializer()
#initializer = tf.keras.initializers.GlorotNormal()

"""
## Define the Transformer Input Layer

When processing past target tokens for the decoder, we compute the sum of
position embeddings and token embeddings.

When processing audio features, we apply convolutional layers to downsample
them (via convolution strides) and process local relationships.
"""
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
            #num_hid, 3, strides=2, padding="same", activation="relu"        # changed by nishi 2023.8.10
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
            #num_hid, 3, strides=2, padding="same", activation="relu"        # changed by nishi 2023.8.10
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
            #num_hid, 3, strides=2, padding="same", activation="relu"        # changed by nishi 2023.8.10
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


"""
## Transformer Encoder Layer
"""
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Transformer Decoder Layer
"""
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


"""
## Complete the Transformer model

Our model takes audio spectrograms as inputs and predicts a sequence of characters.
During training, we give the decoder the target character sequence shifted to the left
as input. During inference, the decoder uses its own past predictions to predict the
next token.
"""
class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
        d_provider=0    # add by nishi 0:original 1:mltu provider
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        self.d_provider = d_provider    # add by nishi 0:original 1:mltu provider

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        print('call():#1')
        #print('type(inputs):',type(inputs))
        #print('len(inputs):',len(inputs))
        source = inputs[0]
        target = inputs[1]
        #print('source.shape:',source.shape)
        #print('target.shape:',target.shape)
        # source.shape: (1392, 193)
        # source.shape: (None, None, 193)
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        print('train_step():#1')
        #print('type(batch):',type(batch))
        #print('len(batch):',len(batch))
        if self.d_provider==0:
            source = batch["source"]
            target = batch["target"]
        else:
            bt=batch[0]
            source = bt[0]
            target = bt[1]
            
        #print('source.shape:',source.shape)
        # source.shape: (None, None, 193)
        #print('target.shape:',target.shape)
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        print('test_step():#1')
        if self.d_provider==0:
            source = batch["source"]
            target = batch["target"]
        else:
            bt=batch[0]
            source = bt[0]
            target = bt[1]

        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        #print('generate():#1')
        #print('tf.shape(source)',tf.shape(source))
        # tf.shape(source) tf.Tensor([   8 1392  193], shape=(3,), dtype=int32)
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)

        #print('tf.shape(dec_input)',tf.shape(dec_input))
        # tf.shape(dec_input) tf.Tensor([  8 186], shape=(2,), dtype=int32)
        #print('tf.shape(dec_logits)',tf.shape(dec_logits))
        # tf.shape(dec_logits) tf.Tensor([185   8   1], shape=(3,), dtype=int32)
        return dec_input

def get_data(wavs, id_to_text, maxlen=50):
    """returns mapping of audio paths and transcription texts"""
    data = []
    for w in wavs:
        w=w.replace('\\', '/')
        #print("w:",w)
        id = w.split("/")[-1].split(".")[0]
        #print("id",id)
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})
    return data


"""
## Preprocess the dataset
"""
class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


def create_text_ds(data):
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds


def path_to_audio(path):
    # spectrogram using stft
    # https://work-in-progress.hatenablog.com/entry/2020/02/26/214211

    if False:
        frame_length=200
        frame_step=80
        fft_length=256
    else:
        # https://keras.io/examples/audio/ctc_asr/ values
        # An integer scalar Tensor. The window length in samples.
        frame_length = 256
        # An integer scalar Tensor. The number of samples to step.
        frame_step = 160
        # An integer scalar Tensor. The size of the FFT to apply.
        # If not provided, uses the smallest power of 2 enclosing frame_length.
        fft_length = 384    

    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    #stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    stfts = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def create_audio_ds(data):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(path_to_audio, num_parallel_calls=tf.data.AUTOTUNE)
    return audio_ds

'''
def create_tf_dataset(data,bs)
    bs: batch size
'''
def create_tf_dataset(data, bs=4):
    audio_ds = create_audio_ds(data)
    text_ds = create_text_ds(data)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

"""
## Callbacks to display predictions
"""
class DisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")


"""
## Learning rate schedule
オリジナルは、バグがあります。
1) tensorflow.keras.callbacks.TensorBoard を併用すると、epoch が、 tf.int64 になって、tf.float64 にキャストできない旨のエラーがでる。
2) lr が、通常とは違って、一度上がって、また下がって行くみたい。--> これで、oK か
"""
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """linear warm up - linear decay"""
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / self.decay_epochs,
        )
        return tf.math.minimum(warmup_lr, decay_lr)


    # https://www.tensorflow.org/api_docs/python/tf/print
    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        #print('epoch:',epoch)
        #tf.print(" epoch:",epoch, output_stream=sys.stdout)
        epoch_f = tf.cast(epoch, tf.float32)
        #tf.print(" epoch_f:",epoch_f, output_stream=sys.stdout)
        lrx = self.calculate_lr(epoch_f)
        return lrx
        #return self.calculate_lr(epoch_f)


# https://analytics-note.xyz/machine-learning/keras-learningratescheduler/
# 一度上がって、再度下げていくみたい。
def lr_schedul(epoch):
    x = 1e-4
    if epoch >= 30:
        x = 1e-4
    elif epoch >= 10:
        x = 1e-3
    return x

lr_decay = LearningRateScheduler(
    lr_schedul,
    # verbose=1で、更新メッセージ表示。0の場合は表示しない
    verbose=1,
)

import typing
#from mltu.transformers import Transformer as TransformerX
import mltu.transformers as trp
class LabelIndexer_my(trp.Transformer):
    """Convert label to index by vocab
    
    Attributes:
        vocab (typing.List[str]): List of characters in vocab
        vocab : vectorizer.char_index
    """
    def __init__(
        self, 
        vocab: typing.List[str]
        ) -> None:
        self.vocab = vocab
        self.char_index = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        #print('LabelIndexer_my():#7')
        return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

    def __call_test__(self, data: np.ndarray, label: np.ndarray):
        ss=[]
        #print('LabelIndexer_my():#7')
        for l in label:
            if l in self.char_index:
                n = self.char_index[l]
                ss.append(n)
        return data, np.array(ss)
        #return {"source":data, "target":np.array(ss)}
        #return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])


class SpectrogramPadding_my(Transformer):
    """Pad spectrogram to max_spectrogram_length
    
    Attributes:
        max_spectrogram_length (int): Maximum length of spectrogram
        padding_value (int): Value to pad
    """
    def __init__(
        self, 
        max_spectrogram_length: int, 
        padding_value: int,
        append: bool = True
        ) -> None:
        self.max_spectrogram_length = max_spectrogram_length
        self.padding_value = padding_value
        self.append=append

    def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
        #print('spectrogram.shape:',spectrogram.shape)
        # spectrogram.shape: (1032, 193)
        if self.append==False:
            padded_spectrogram = np.pad(spectrogram, 
                ((self.max_spectrogram_length - spectrogram.shape[0], 0),(0,0)),mode="constant",constant_values=self.padding_value)
        else:
            l,h =spectrogram.shape
            lng = self.max_spectrogram_length - l
            if lng > 0:
                a = np.full((lng,h),self.padding_value)
                padded_spectrogram = np.append(spectrogram, a, axis=0)
            else:
                padded_spectrogram = spectrogram
        return padded_spectrogram, label


#  DataProvider for asr
class DataProviderAsr(DataProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index: int):
        """ Returns a batch of data by batch index"""
        dataset_batch = self.get_batch_annotations(index)
        
        # First read and preprocess the batch data
        batch_data, batch_annotations = [], []
        for index, batch in enumerate(dataset_batch):

            data, annotation = self.process_data(batch)

            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, skipping.")
                continue

            batch_data.append(data)
            batch_annotations.append(annotation)

        so=np.array(batch_data)
        #print('so.shape:',so.shape)
        sa=np.array(batch_annotations)
        #print('sa.shape:',sa.shape)
        
        r=[so, sa]
        #print('type',type(r))
        #print('np.shape(r[0])',np.shape(r[0]))
        #print('np.shape(r[1])',np.shape(r[1]))
        return [r]

def plot_spectrogram(spectrogram: np.ndarray, title:str = "", transpose: bool = True, invert: bool = True) -> None:
    """Plot the spectrogram of a WAV file

    Args:
        spectrogram (np.ndarray): Spectrogram of the WAV file.
        title (str, optional): Title of the plot. Defaults to None.
        transpose (bool, optional): Transpose the spectrogram. Defaults to True.
        invert (bool, optional): Invert the spectrogram. Defaults to True.
    """
    if transpose:
        spectrogram = spectrogram.T

    if invert:
        spectrogram = spectrogram[::-1]

    plt.figure(figsize=(15, 5))
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.title(f"Spectrogram: {title}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    #plt.colorbar()
    plt.tight_layout()
    plt.show()


vectorizer = VectorizeChar()

#-----------------
# main start
#-----------------
if __name__ == "__main__":
    """
    ## Download the dataset

    Note: This requires ~3.6 GB of disk space and
    takes ~5 minutes for the extraction of files.
    """
    from mltu.configs import BaseModelConfigs

    CONT_F=True
    test_date="202308191155"
    
    USE_lr_MY=False
    USE_TEST_DATA_ORG = False
    USE_TEST_DATA_MLTU = True

    initial_epoch=0             # start 0

    if CONT_F==False:
        # Create a ModelConfigs object to store model configurations
        configs = ModelConfigs()
        configs.vocab =  vectorizer.vocab
        configs.char_to_idx = vectorizer.char_to_idx        
    else:
        configs = BaseModelConfigs.load("Models/"+test_date+"/configs.yaml")

    checkpoint_dir= configs.model_path+'/training'  
    print('checkpoint_dir:',checkpoint_dir)
    checkpoint_path = checkpoint_dir+"/cp-{epoch:04d}.ckpt"

    if CONT_F==False:

        if False:
            keras.utils.get_file(
            os.path.join(os.getcwd(), "data.tar.gz"),
            "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
            extract=True,
            archive_format="tar",
            cache_dir=".",
            )

        # lstm_sound_to_text のデータを使う。
        #saveto = "./datasets/LJSpeech-1.1"
        saveto = "../lstm_sound_to_text/Datasets/LJSpeech-1.1"

        if USE_TEST_DATA_ORG == True:
            wavs = glob("{}/**/*.wav".format(saveto), recursive=True)

            id_to_text = {}
            with open(os.path.join(saveto, "metadata.csv"), encoding="utf-8") as f:
                for line in f:
                    id = line.strip().split("|")[0]
                    text = line.strip().split("|")[2]
                    id_to_text[id] = text

            if False:
                for i,k in enumerate(id_to_text.keys()):
                    print(k,':',id_to_text[k])
                    if i > 2:
                        break

            #max_target_len = 200  # all transcripts in out data are < 200 characters
            max_target_len = 189
            data = get_data(wavs[:100], id_to_text, max_target_len)

            #vectorizer = VectorizeChar(max_target_len)
            vectorizer.max_len=max_target_len

            print("vocab size", len(vectorizer.get_vocabulary()))
            
            print('len(data):',len(data))
            split = int(len(data) * 0.99)
            train_data = data[:split]
            test_data = data[split:]
            print('data[:2]',data[:2])
            
            ds = create_tf_dataset(train_data, bs=8)   # bs=batch size
            val_ds = create_tf_dataset(test_data, bs=4)

            batch = next(iter(val_ds))

            # The vocabulary to convert predicted indices into characters
            idx_to_char = vectorizer.get_vocabulary()

            print(f"Size of the training set: {len(ds)}")
            print(f"Size of the training set: {len(val_ds)}")

            configs.max_text_length = max_target_len
            configs.save()

        if USE_TEST_DATA_MLTU == True:
            #----------------------------------------
            # pandas を使ってみる。
            # https://keras.io/examples/audio/ctc_asr/
            #----------------------------------------
            
            #dataset_path = "Datasets/LJSpeech-1.1"
            dataset_path = saveto
            metadata_path = dataset_path + "/metadata.csv"
            wavs_path = dataset_path + "/wavs/"

            # Read metadata file and parse it
            metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
            metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
            metadata_df = metadata_df[["file_name", "normalized_transcription"]]

            # structure the dataset where each row is a list of [wav_file_path, sound transcription]
            #dataset = [[f"Datasets/LJSpeech-1.1/wavs/{file}.wav", label.lower()] for file, label in metadata_df.values.tolist()]
            dataset = [[wavs_path+f"{file}.wav", label.lower()] for file, label in metadata_df[:100].values.tolist()]

            max_text_length, max_spectrogram_length = 0, 0
            
            for file_path, label in tqdm(dataset):
                spectrogram = WavReader.get_spectrogram(file_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
                valid_label = [c for c in label if c in configs.vocab]
                max_text_length = max(max_text_length, len(valid_label))
                max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])
                configs.input_shape = [max_spectrogram_length, spectrogram.shape[1]]

            configs.max_spectrogram_length = max_spectrogram_length
            configs.max_text_length = max_text_length
            configs.save()
            
            # check wav spectrogram
            if False:
                for file_path, label in tqdm(dataset):
                    spectrogram = WavReader.get_spectrogram(file_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
                    plot_spectrogram(spectrogram,title=label)
                    
                    break
                sys.exit()
                

            #metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)        # randam data
            #print('--------')
            #print(metadata_df.head(3))

            # Create a data provider for the dataset
            #data_provider = DataProvider(
            data_provider = DataProviderAsr(
                dataset=dataset,
                skip_validation=True,
                batch_size=configs.batch_size,
                data_preprocessors=[
                    WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                    ],
                transformers=[
                    #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    LabelIndexer_my(configs.vocab),
                    #LabelIndexer_jp(configs.vocab),
                    LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
                    ],
            )

            # Split the dataset into training and validation sets
            train_data_provider, val_data_provider = data_provider.split(split = 0.9)
            
            print('train_data_provider.__len__():',train_data_provider.__len__())

            # Save training and validation datasets as csv files
            train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
            val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
            


        #split = int(len(metadata_df) * 0.90)
        #df_train = metadata_df[:split]
        #df_val = metadata_df[split:]


        #sys.exit()
    else:
        if USE_TEST_DATA_MLTU == True:
            #metadata_df = pd.read_csv(metadata_path, sep="|")        
            # get kanji vocaab
            #vectorizer=kanj_to_vocab(metadata_df)
            dataset_train = pd.read_csv(configs.model_path+"/train.csv").values.tolist()
            dataset_val = pd.read_csv(configs.model_path+"/val.csv").values.tolist()
            #train_data_provider = DataProvider(
            train_data_provider = DataProviderAsr(
                dataset=dataset_train,
                skip_validation=True,
                batch_size=configs.batch_size,
                data_preprocessors=[
                    WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                    ],
                transformers=[
                    #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    #LabelIndexer(configs.vocab),
                    LabelIndexer_my(configs.vocab),
                    LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
                    ],
            )
            #val_data_provider = DataProvider(
            val_data_provider = DataProviderAsr(
                dataset=dataset_val,
                skip_validation=True,
                batch_size=configs.batch_size,
                data_preprocessors=[
                    WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                    ],
                transformers=[
                    SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    #LabelIndexer(configs.vocab),
                    LabelIndexer_my(configs.vocab),
                    LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
                    ],
            )
        
    # train_data, test_data をファイルに保存する
    # https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
    if USE_TEST_DATA_ORG==True:
        import pickle
        if CONT_F== False:
            print('len(data):',len(data))
            split = int(len(data) * 0.99)
            train_data = data[:split]
            test_data = data[split:]
            #print('data[:2]',data[:2])
            with open("data/train_data", "wb") as fp:   #Pickling
                pickle.dump(train_data, fp)

            with open("data/test_data", "wb") as fp:   #Pickling
                pickle.dump(test_data, fp)

        else:
            with open("data/train_data", "rb") as fp:   # Unpickling
                train_data = pickle.load(fp)    
            with open("data/test_data", "rb") as fp:   # Unpickling
                test_data = pickle.load(fp)    

        #ds = create_tf_dataset(train_data, bs=64)   # bs=batch size
        ds = create_tf_dataset(train_data, bs=configs.batch_size)   # bs=batch size
        val_ds = create_tf_dataset(test_data, bs=4)
        #sys.exit()


    """
    ## Create & train the end-to-end model
    """
    def label_to_str(lable):
        ss=''
        for i in lable:
            #print('i:',i)
            if i < len(configs.vocab):
                c=configs.vocab[i]
                ss += c
            else:
                ss += '.'
        return ss

    # original dataprovider
    if USE_TEST_DATA_ORG==True:
        #dsx=list(ds)
        for val in ds:
            #print(type(val))
            # <class 'dict'>
            #print(val.keys())
            # dict_keys(['source', 'target'])
            
            source=val['source']        # input
            target=val['target']        # label
            print('source.shape:',source.shape)
            # source.shape: (8, 2754, 193)
            print('target.shape:',target.shape)
            # target.shape: (8, 200)
            
            print('type(source)',type(source))
            # type(source) <class 'tensorflow.python.framework.ops.EagerTensor'>
            print('type(target)',type(target))
            # type(target) <class 'tensorflow.python.framework.ops.EagerTensor'>

            label = label_to_str(target[0]._numpy())
            print('label:',label)
            plot_spectrogram(source[0]._numpy(),title=label)

            break

    if USE_TEST_DATA_MLTU==True:
        for val in train_data_provider:
            print(type(val))
            # <class 'list'>
            if False:
                source ,target=val
                print('source.shape:',source.shape)
                # source.shape: (8, 1371, 193)
                print('target.shape:',target.shape)
                # target.shape: (8, 166)
            else:
                source=val[0][0]        # source input
                target=val[0][1]        # target label
                print('source.shape:',source.shape)
                # source.shape: (8, 1371, 193)
                print('target.shape:',target.shape)
                # target.shape: (8, 166)

                print('type(source)',type(source))
                # type(source) <class 'numpy.ndarray'>
                print('type(target)',type(target))
                # type(target) <class 'numpy.ndarray'>

                label = label_to_str(target[0])
                print('label:',label)
                plot_spectrogram(source[0],title=label)
                
            break
            
