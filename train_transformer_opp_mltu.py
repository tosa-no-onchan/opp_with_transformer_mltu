"""
ROS2 ML Path Planner with Transformer

Title: Obstacle Path Planner with Transformer and mltu

Prog: train_transformer_opp_mltu.py

1. how to train
$ python train_transformer_opp_mltu.py

tensorflow==2.16.1  with cuda
keras==3.3.3
mlut==1.2.5
python 3.10.12

**References:**

- [Attention is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Very Deep Self-Attention Networks for End-to-End Speech Recognition](https://arxiv.org/pdf/1904.13377.pdf)
- [Speech Transformers](https://ieeexplore.ieee.org/document/8462506)
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
"""
import os
import sys

from glob import glob
import tensorflow as tf

try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]

# add by nishi 2024.6.6
os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = "enable"

import random

import keras
from keras import layers

from k3_transformer_opp_model import Transformer,TokenEmbedding,SpeechFeatureEmbedding,TransformerEncoder,TransformerDecoder
#from transformer_asr_model import Transformer,TokenEmbedding,SpeechFeatureEmbedding,TransformerEncoder,TransformerDecoder
from tools_mltu import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard,LearningRateScheduler

import pandas as pd
from tqdm import tqdm

import numpy as np
from mltu.preprocessors import WavReader
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from configs import ModelConfigs

#keras.initializers.Initializer()
#initializer = tf.keras.initializers.GlorotNormal()

import yaml
from pathlib import Path

"""
## Callbacks to display predictions
"""
class DisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28,d_provider=0):
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
        self.d_provider = d_provider    # add by nishi 0:original 1:mltu provider

    def on_epoch_end(self, epoch, logs=None):
        #if epoch % 5 != 0:
        if epoch % 10 != 0:
            return
        if self.d_provider==0:
            source = self.batch["source"]
            target = self.batch["target"].numpy()
        else:
            #print('type(self.batch)',type(self.batch))
            bt=self.batch.__getitem__(1)
            #print('type(bt)',type(bt))
            # type(bt) <class 'list'>
            #source = bt[0][0]
            # changed by nishi 2024.10.2
            source = bt[0]
            #print('type(source)',type(source))
            # type(source) <class 'numpy.ndarray'>
            #print('np.shape(source)',np.shape(source))
            # np.shape(source) (8, 1392, 193)
            #print('tf.shape(source)',tf.shape(source))
            # tf.shape(source) tf.Tensor([   8 1392  193], shape=(3,), dtype=int32)
            #target = bt[0][1]
            # changed by nishi 2024.10.2
            target = bt[1]
            #print('type(target)',type(target))
            # type(target) <class 'numpy.ndarray'>
            #print('np.shape(target)',np.shape(target))
            # np.shape(target) (8, 186)

        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        #print('tf.shape(preds)',tf.shape(preds)) 
        # tf.shape(preds) tf.Tensor([  8 186], shape=(2,), dtype=int32)       
        preds = preds.numpy()
        for i in range(bs):
            #target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            target_text = ""
            for c in target[i, :]:
                if c < len(self.idx_to_char):
                    target_text += self.idx_to_char[c]
            prediction = ""
            for idx in preds[i, :]:
                if idx < len(self.idx_to_char):
                    prediction += self.idx_to_char[idx]
                #if idx == self.target_end_token_idx:
                #    break
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

    # add by nishi 2024.6.2
    def get_config(self):
        #base_config = super().get_config()
        config = {
                "init_lr": self.init_lr,
                "lr_after_warmup": self.lr_after_warmup,
                "final_lr": self.final_lr,
                "warmup_epochs": self.warmup_epochs,
                "decay_epochs": self.decay_epochs,
                "steps_per_epoch": self.steps_per_epoch,
        }
        #return dict(list(base_config.items()) + list(config.items()))
        return config

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

    CONT_F=False
    epoch_num=10000
    #epoch_num=500
    #epoch_num=1
    test_date="test_opp"

    USE_TEST_DATA_OPP = True    # Obstacle Path Planning
    USE_TEST_DATA_MLTU = False

    USE_lr_MY=False
    initial_epoch=0             # start 0

    # 初期乱数作成
    set_seed(0)

    if CONT_F==False:
        # Create a ModelConfigs object to store model configurations
        configs = ModelConfigs()
        configs.vocab =  vectorizer.vocab
        configs.char_to_idx = vectorizer.char_to_idx     
        configs.model_path = os.path.join("Models/", test_date)   
    else:
        configs = ModelConfigs.load("Models/"+test_date+"/configs.yaml")

    checkpoint_dir= configs.model_path+'/training'
    print('checkpoint_dir:',checkpoint_dir)

    if CONT_F==True:
        latest=latest_checkpoint(checkpoint_dir)
        initial_epoch =latest_checkno(latest)

    print('initial_epoch:',initial_epoch)
    checkpoint_path = checkpoint_dir+"/cp-{epoch:04d}.weights.h5"

    if CONT_F==False:
        if USE_TEST_DATA_OPP == True:
            imgs = glob("{}/*.yaml".format(configs.opp_path), recursive=False)
            datasetx=[]
            for s in imgs:
                #print("s:",s)
                with open(s, 'r') as yml:
                    config = yaml.safe_load(yml)
                    #print("image:",config['image'])
                    #print("data:",config['data'])
                    datasetx.append([config['image'],config['data']])

            dataset = random.sample(datasetx, len(datasetx))
            dt_len=len(dataset)
            train_l=int(dt_len*0.9)

            print("train_l:",train_l)

            #print("dataset[1]:",dataset[1])
            #max_spectrogram_length=800
            max_spectrogram_length=600

            input_shape_cols=max_spectrogram_length
            #input_shape_rows=120
            input_shape_rows=122

            configs.input_shape = [max_spectrogram_length, input_shape_rows]

            x_interval=4
            #x_interval=2
            max_text_length= int(max_spectrogram_length/x_interval)
            print("max_text_length:",max_text_length)
            configs.max_spectrogram_length = max_spectrogram_length
            configs.max_text_length = max_text_length
            configs.save()

            train_data_provider = DataProviderAsr(
                dataset=dataset[:train_l],
                skip_validation=True,
                batch_size=configs.batch_size,
                data_preprocessors=[
                    #WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                    OppReader(opp_path=configs.opp_path, input_shape=configs.input_shape),
                    ],
                transformers=[
                    #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    #LabelIndexer(configs.vocab),
                    LabelIndexer_my(configs.vocab,configs.char_to_idx),
                    LabelPadding(max_word_length=configs.max_text_length, padding_value=49),        # 'B'
                    #LabelPadding(max_word_length=configs.max_text_length, padding_value=50),       # '.'
                    #LabelPadding(max_word_length=configs.max_text_length, padding_value=24),        # center
                    ],
            )

            val_data_provider = DataProviderAsr(
                dataset=dataset[train_l:],
                skip_validation=True,
                batch_size=4,
                data_preprocessors=[
                    #WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                    OppReader(opp_path=configs.opp_path,input_shape=configs.input_shape),
                    ],
                transformers=[
                    #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    #LabelIndexer(configs.vocab),
                    LabelIndexer_my(configs.vocab,configs.char_to_idx),
                    LabelPadding(max_word_length=configs.max_text_length, padding_value=49),    # 'B'
                    #LabelPadding(max_word_length=configs.max_text_length, padding_value=50),   # '.'
                    #LabelPadding(max_word_length=configs.max_text_length, padding_value=24),    # center
                    ],
            )

            # Save training and validation datasets as csv files
            train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
            val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

            # Split the dataset into training and validation sets
            #train_data_provider, val_data_provider = data_provider.split(split = 0.9)
            print('train_data_provider.__len__():',train_data_provider.__len__())

            ds=train_data_provider
            val_ds=val_data_provider

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
                    LabelIndexer_my(configs.vocab,configs.char_to_idx),
                    LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
                    ],
            )
            #val_data_provider = DataProvider(
            val_data_provider = DataProviderAsr(
                dataset=dataset_val,
                skip_validation=True,
                #batch_size=configs.batch_size,
                # changed by nishi 2024.10.2
                batch_size=4,
                data_preprocessors=[
                    WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
                    ],
                transformers=[
                    #SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
                    #LabelIndexer(configs.vocab),
                    LabelIndexer_my(configs.vocab,configs.char_to_idx),
                    LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
                    ],
            )
            ds=train_data_provider
            val_ds=val_data_provider

            #latest = tf.train.latest_checkpoint(checkpoint_dir)
            #basename_without_ext = os.path.splitext(os.path.basename(latest))[0]
            #initial_epoch=int(basename_without_ext.split('-')[1])
            #print('initial_epoch:',initial_epoch)
        
    max_target_len = configs.max_text_length
    #data = get_data(wavs, id_to_text, max_target_len)

    print('max_target_len:',max_target_len)
    #vectorizer = VectorizeChar(max_target_len)
    print("vocab size", len(vectorizer.get_vocabulary()))
    # vocab size 34

    #sys.exit()

    d_provider=1

    """
    ## Create & train the end-to-end model
    """
    if USE_TEST_DATA_MLTU==True:
        idx_to_char = configs.vocab
        display_cb = DisplayOutputs(
            val_ds,
            # changed by nishi 2024.10.2
            #batch, 
            idx_to_char, target_start_token_idx=2, target_end_token_idx=3,d_provider=d_provider
        )  # set the arguments as per vocabulary index for '<' and '>'

    if USE_TEST_DATA_OPP==True:
        idx_to_char = configs.vocab
        display_cb = DisplayOutputs(
            val_ds,
            # changed by nishi 2024.10.2
            #batch, 
            #idx_to_char, target_start_token_idx=0,d_provider=d_provider
            idx_to_char, target_start_token_idx=1,d_provider=d_provider
        )  # set the arguments as per vocabulary index for '<' and '>'


    print("configs.model_path:",configs.model_path)

    """
    /home/nishi/kivy_env/lib/python3.10/site-packages/keras/src/callbacks/model_checkpoint.py

    # Alternatively, one could checkpoint just the model weights as -
    checkpoint_filepath = '/tmp/ckpt/checkpoint.weights.h5'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    """
    #checkpoint_filepath = '/tmp/ckpt/checkpoint.weights.h5'
    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        #f"{configs.model_path}/a.model.keras", 
        monitor="loss",
        #monitor="CER",
        #monitor="val_CER",
        verbose=1, 
        save_best_only=True,
        save_weights_only=True,
        #save_freq=20*batch_size,
        save_freq=20*configs.batch_size,
        mode="min")

    # tensorboard 
    # https://teratail.com/questions/97nyrumr5iix6d
    tb_callback = TensorBoard(checkpoint_dir+"/logs", update_freq=1)

    #earlystopper = EarlyStopping(monitor="val_loss", patience=3, verbose=1, mode="min")
    #earlystopper = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="min")
    earlystopper = EarlyStopping(monitor="loss", patience=15, verbose=1, mode="min")
    #earlystopper = EarlyStopping(monitor="loss", patience=20, verbose=1, mode="min")
    #earlystopper = EarlyStopping(monitor="loss", patience=30, verbose=1, mode="min")

    print("passed:#3 ")

    # 入力画像サイズ  (122,600) -> Transponse (600,122)
    model = Transformer(
        num_hid=200,            # MultiHeadAttentionの key_dim
        #num_hid=122,            # MultiHeadAttentionの key_dim
        num_head=2,             # MultiHeadAttentionの num_heads
        #num_head=6,             # MultiHeadAttentionの num_heads
        #num_feed_forward=400,      # 
        #num_feed_forward=244,       # 122*2=244 1度に取り込む、 データ数か ? -> 1 vocab 分のデータか by nishi 2024.10.12
        num_feed_forward=488,       # 122*4=244 1度に取り込む、 データ数か ? -> 1 vocab 分のデータか by nishi 2024.12.2
        source_maxlen=max_target_len,
        target_maxlen=max_target_len,   # 600 / 1 vocab rows(4) = 150 points(vocab) を出力か?
        num_layers_enc=4,       # default
        #num_layers_enc=2,       # ちょっと減らしてみる by nishi 2024.10.12
        #num_layers_enc=6,       # ちょっと増やしてみる by nishi 2024.10.30
        #num_layers_dec=1,       # default
        num_layers_dec=2,       # ちょっと増やしてみる by nishi 2024.10.29
        #num_layers_dec=3,       # ちょっと増やしてみる by nishi 2024.10.30
        #num_classes=34,
        num_classes=50,     # 最終出力の class numbers  -> 49:'B'
        #num_classes=51,     # 最終出力の class numbers
        #num_classes=52,     # 最終出力の class numbers
        d_provider=d_provider
    )

    #sys.exit()
    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=0.1,
    )
    learning_rate = CustomSchedule(
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        #decay_epochs=50,
        steps_per_epoch=len(ds),
    )

    #print("len(ds):",len(ds))
    #sys.exit()
    if USE_lr_MY==False:
        optimizer = keras.optimizers.Adam(learning_rate)
        if USE_TEST_DATA_MLTU == True:
            #callbacks_list=[cp_callback, earlystopper, tb_callback]
            callbacks_list=[display_cb, cp_callback, earlystopper, tb_callback]
        elif  USE_TEST_DATA_OPP == True:
            callbacks_list=[cp_callback, earlystopper, tb_callback]
        else:
            callbacks_list=[display_cb, cp_callback, earlystopper, tb_callback]
        #callbacks_list=[cp_callback, earlystopper, tb_callback]
    else:
        initial_lr=lr_schedul(initial_epoch)
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr)
        #callbacks_list=[display_cb, cp_callback, earlystopper, tb_callback,lr_decay]

        if USE_TEST_DATA_MLTU == True:
            #callbacks_list=[cp_callback, earlystopper, tb_callback]
            callbacks_list=[display_cb, cp_callback, earlystopper, tb_callback]

        elif  USE_TEST_DATA_OPP == True:
            callbacks_list=[cp_callback, earlystopper, tb_callback]
        else:
            callbacks_list=[display_cb, cp_callback, earlystopper, tb_callback]
        #callbacks_list=[cp_callback, earlystopper, tb_callback,lr_decay]

    model.compile(optimizer=optimizer, loss=loss_fn)
    print("passed:#4 ")
    #sys.exit()
    if CONT_F == True:
        #latest = tf.train.latest_checkpoint(checkpoint_dir)
        latest=latest_checkpoint(checkpoint_dir)
        print('latest:',latest)       # Models/test1/training/cp-0002.weights.h5
        #sys.exit()
        model.load_weights(latest)
        if True:
            # reset model.optimizer.adam
            if os.path.exists(configs.model_path+'/conf_adam.yaml'):
                print('> reset optimizer.adam')
                conf_adam = yaml.safe_load(Path(configs.model_path+'/conf_adam.yaml').read_text())
                conf_adam['learning_rate']=learning_rate
                #optimizer.from_config(conf_adam)
                model.optimizer.from_config(conf_adam)

    #model.summary(line_length=110,expand_nested=True)
    print('>>>passed:#11')
    #sys.exit()
    #epoch_num=100

    history = model.fit(ds, 
                validation_data=val_ds,
                epochs=initial_epoch+epoch_num,
                initial_epoch=initial_epoch,
                #callbacks=[display_cb, cp_callback, earlystopper, tb_callback], 
                #callbacks=[display_cb, cp_callback, earlystopper], 
                #callbacks=callbacks_list,
                #callbacks=[display_cb, cp_callback],
                #callbacks=[cp_callback,tb_callback], 
                #callbacks=[tb_callback],
                callbacks=[display_cb, earlystopper],
                )
    print('passed: #99')
    '''
    In practice, you should train for around 100 epochs or more.

    Some of the predicted text at or around epoch 35 may look as follows:

    target:     <as they sat in the car, frazier asked oswald where his lunch was>
    prediction: <as they sat in the car frazier his lunch ware mis lunch was>

    target:     <under the entry for may one, nineteen sixty,>
    prediction: <under the introus for may monee, nin the sixty,>
    '''
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History
    #print(history.history.keys())
    # dict_keys(['loss', 'val_loss'])
    MODEL_DIR='./model_sav'
    epo=history.params['epochs']

    model.save(configs.model_path+'/a.model.keras')
    print('passed: #100')

    if False:
        import tf2onnx
        model.output_names = ['output']
        input_signature = [tf.TensorSpec(shape=(None,600, 122),dtype=tf.float32, name='digit')]
        tf2onnx.convert.from_keras(model,input_signature, output_path=configs.model_path+'/model.onnx')

    print('passed: #101')
    # https://www.tensorflow.org/guide/saved_model?hl=ja
    # カスタムモデルの保存
    # tf.saved_model.save を使う。
    tf.saved_model.save(model,configs.model_path+'/a.model')
    model_tf = tf.saved_model.load(configs.model_path+'/a.model')

    # 
    #conf_adam = optimizer.get_config()
    conf_adam = model.optimizer.get_config()
    #print("conf_adam:",conf_adam)
    # save to yaml faile
    # https://stackoverflow.com/questions/12470665/how-can-i-write-data-in-yaml-format-in-a-file
    with open(configs.model_path+'/conf_adam.yaml', 'w') as outfile:
        #yaml.dump(conf_adam, outfile, default_flow_style=False)
        yaml.dump(conf_adam, outfile, default_flow_style=True)

    if False:
        # read test
        yaml_dict = yaml.safe_load(Path(configs.model_path+'/conf_adam.yaml').read_text())
        print("yaml_dict:",yaml_dict)
