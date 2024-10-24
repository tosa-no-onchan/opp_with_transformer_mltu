'''
k3_transformer_opp_model.py

'''

import tensorflow as tf


import keras
from keras import layers


"""
## Define the Transformer Input Layer

When processing past target tokens for the decoder, we compute the sum of
position embeddings and token embeddings.

When processing audio features, we apply convolutional layers to downsample
them (via convolution strides) and process local relationships.
"""

#------------
# reffer https://keras.io/examples/keras_recipes/antirectifier/
# https://keras.io/guides/making_new_layers_and_models_via_subclassing/
#------------
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.num_vocab=num_vocab
        self.maxlen = maxlen
        self.num_hid=num_hid

        #self.emb = keras.layers.Embedding(num_vocab, num_hid)
        #self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    # add by nishi
    def build(self, input_shape):
        self.emb = tf.keras.layers.Embedding(self.num_vocab, self.num_hid)
        self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    ### Start 追加されたコード
    def get_config(self):
        config = {
            "num_vocab": self.num_vocab,
            "maxlen": self.maxlen,
            "num_hid": self.num_hid,
            "emb": self.emb,
            "pos_emb": self.pos_emb,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    ###  End  

class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.num_hid=num_hid
        self.maxlen=maxlen
        #self.conv1 = keras.layers.Conv1D(
        #    num_hid, 11, strides=2, padding="same", activation="relu"
        #)
        #self.conv2 = keras.layers.Conv1D(
        #    num_hid, 11, strides=2, padding="same", activation="relu"
        #)
        #self.conv3 = keras.layers.Conv1D(
        #    num_hid, 11, strides=2, padding="same", activation="relu"
        #)

    # add by nishi
    def build(self, input_shape):
        self.conv1 = keras.layers.Conv1D(
            self.num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = keras.layers.Conv1D(
            self.num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = keras.layers.Conv1D(
            self.num_hid, 11, strides=2, padding="same", activation="relu"
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)

    ### Start 追加されたコード
    def get_config(self):
        config = {
            "num_hid": self.num_hid,
            "maxlen": self.maxlen,
            "conv1": self.conv1,
            "conv2": self.conv2,
            "conv3": self.conv3,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    ###  End  


"""
## Transformer Encoder Layer

https://keras.io/examples/nlp/text_classification_with_transformer/

"""
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.feed_forward_dim=feed_forward_dim
        self.rate=rate

        #self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        #self.ffn = keras.Sequential(
        #    [
        #        layers.Dense(feed_forward_dim, activation="relu"),
        #        layers.Dense(embed_dim),
        #    ]
        #)
        #self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        #self.dropout1 = layers.Dropout(rate)
        #self.dropout2 = layers.Dropout(rate)

    # add by nishi
    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.feed_forward_dim, activation="relu"),
                layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)


    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    ### Start 追加されたコード
    def get_config(self):
        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "rate": self.rate,
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    ###  End  

"""
## Transformer Decoder Layer
"""
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.feed_forward_dim=feed_forward_dim
        self.dropout_rate=dropout_rate

        #self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        #self.self_att = layers.MultiHeadAttention(
        #    num_heads=num_heads, key_dim=embed_dim
        #)
        #self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        #self.self_dropout = layers.Dropout(0.5)
        #self.enc_dropout = layers.Dropout(0.1)
        #self.ffn_dropout = layers.Dropout(0.1)
        #self.ffn = keras.Sequential(
        #    [
        #        layers.Dense(feed_forward_dim, activation="relu"),
        #        layers.Dense(embed_dim),
        #    ]
        #)

    # add by nishi
    def build(self, input_shape):
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.feed_forward_dim, activation="relu"),
                layers.Dense(self.embed_dim),
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

    ### Start 追加されたコード
    def get_config(self):
        config = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "dropout_rate": self.dropout_rate,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "layernorm3": self.layernorm3,
            "self_att": self.self_att,
            "enc_att": self.enc_att,
            "self_dropout": self.self_dropout,
            "enc_dropout": self.enc_dropout,
            "ffn_dropout": self.ffn_dropout,
            "ffn": self.ffn,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    ###  End  


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
        d_provider=0,    # add by nishi 0:original 1:mltu provider
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

        # out put class
        self.classifier = layers.Dense(num_classes)
        # test by nishi 2024.10.15
        #self.classifier = layers.Dense(num_classes, activation="sigmoid")

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    # add by nishi
    # https://github.com/keras-team/keras/issues/18823
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "classifier": self.classifier,
                #"classifier": self.classifier.get_config(),
            }
        )
        return config
    
    # add by nishi
    # https://discuss.pytorch.org/t/attributeerror-resnet50-object-has-no-attribute-conv1/72072/4
    #def build_from_config(self, config):
    #    print('build_from_config() called ,config=',config)
    #    super().build_from_config(config)
    #    
    #    self.classifier=config["classifier"]
    def build(self,input_shape):
        # out put class
        self.classifier = layers.Dense(self.num_classes)
        # test by nishi 2024.10.15
        #self.classifier = layers.Dense(self.num_classes, activation="sigmoid")


    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        if self.d_provider==0:
            source = batch["source"]
            target = batch["target"]
        else:
            bt=batch
            source = bt[0]
            target = bt[1]

        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            #loss = model.compute_loss(None, one_hot, preds, sample_weight=mask)
            loss = super().compute_loss(None, one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        #print('test_step():#1')
        if self.d_provider==0:
            source = batch["source"]
            target = batch["target"]
        else:
            #bt=batch[0]
            bt=batch
            source = bt[0]
            target = bt[1]

        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        #loss = model.compute_loss(None, one_hot, preds, sample_weight=mask)
        loss = super().compute_loss(None, one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]

        if self.d_provider==0:
            #print('type(source):',type(source))
            # type(source): <class 'tensorflow.python.framework.ops.EagerTensor'>
            #print('tf.shape(source)',tf.shape(source))
            # tf.shape(source) tf.Tensor([   4 2754  193], shape=(3,), dtype=int32)
            enc = self.encoder(source)

        else:
            #print('type(source):',type(source))
            # type(source): <class 'numpy.ndarray'>
            #print('tf.shape(source)',tf.shape(source))
            # tf.shape(source) tf.Tensor([1392  193], shape=(2,), dtype=int32)
            # batch 軸が必要みたい。 by nishi 2024.10.2
            #sourcex=source[np.newaxis,:, :]
            #sourcex=np.expand_dims(source,axis=0)
            #print('tf.shape(sourcex)',tf.shape(sourcex))
            enc = self.encoder(source)

        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            # ここで、 warnigs か?
            # /home/nishi/kivy_env/lib/python3.10/site-packages/keras/src/ops/nn.py:545: UserWarning:
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input
