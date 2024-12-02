import os
import sys
import typing
import numpy as np

import tensorflow as tf
from mltu.inferenceModel import OnnxInferenceModel
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

#from train import WavReaderMel
from configs import ModelConfigs

#from keras.models import load_model
#from tensorflow.keras.models import load_model
import keras

#keras.saving.load_model
#from keras.saving import load_model
from keras.config import enable_unsafe_deserialization
from mltu.tensorflow.metrics import CERMetric

#from mltu.tensorflow.losses import CTCloss
#from losses import CTCloss

import mltu.tensorflow.losses
from tools_mltu import *
from train_transformer_opp_mltu import CustomSchedule
import cv2

class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, data: np.ndarray):
        data_pred = np.expand_dims(data, axis=0)
        preds = self.model.run(None, {self.input_name: data_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

import matplotlib.pyplot as plt

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

if False:
    def pred_my(self,model, source):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]

        enc = self.encoder(source)

        dec_input = tf.ones((bs, 1), dtype=tf.int32)
        dec_logits = []
        print("self.target_maxlen:",self.target_maxlen)
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


#----------------
# https://github.com/leimao/Frozen-Graph-TensorFlow/blob/master/TensorFlow_v2/utils.py
#----------------
def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    test_date="test_opp"

    model_dir="Models/"+test_date
    configs = ModelConfigs.load("Models/"+test_date+"/configs.yaml")

    print("configs.model_path:",configs.model_path)

    LOAD_2=False        # Keras Saved model   --> NG
    LOAD_3=False         # Tensorflow Saved model    --> OK
    LOAD_4=True        # load frozen model    --> OK

    if LOAD_2==True:
        print("LOAD_2")
        #loss_fn=CTCloss()
        cer_m=CERMetric(configs.vocab)

        objs_x={#"CTCloss":CTCloss(),
                #"CERMetric":CERMetric(configs.vocab),
                "CustomSchedule":CustomSchedule
                }

        #model = load_model(configs.model_path+'/a.model.keras',safe_mode=False)
        model = keras.saving.load_model(configs.model_path+'/a.model.keras',custom_objects=objs_x,safe_mode=False)
        #model = load_model(configs.model_path+'/a.model.hdf5',custom_objects=objs_x,safe_mode=False)

    if LOAD_3==True:
        print("LOAD_3")
        # https://www.tensorflow.org/guide/saved_model?hl=ja
        # カスタムモデルの読み込みと使用
        model = tf.saved_model.load(configs.model_path+'/a.model')

        if False:
            concrete_func = model.signatures[
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

            # Get frozen ConcreteFunction
            from tensorflow.python.framework import convert_to_constants
            constantGraph = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

            #constantGraph.graph.as_graph_def()
            #layers = [op.name for op in constantGraph.graph.get_operations()]
            #print(layers)

            print("Frozen model inputs: ")
            print(constantGraph.inputs)
            # [<tf.Tensor 'inputs:0' shape=(None, 600, 122) dtype=float32>]
            print("Frozen model outputs: ")
            print(constantGraph.outputs)
            # [<tf.Tensor 'Identity:0' shape=(None, 300, 53) dtype=float32>]

    # use Frozen model
    if LOAD_4==True:
        # frozen model を試す。
        print("LOAD_4")
        # https://github.com/leimao/Frozen-Graph-TensorFlow/blob/master/TensorFlow_v2/example_1.py

        # Load frozen graph using TensorFlow 1.x functions
        with tf.io.gfile.GFile(configs.model_path+"/a.model_frozen.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions
        frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=["source:0"],
                                        outputs=["Identity:0"],
                                        print_graph=False)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        # [<tf.Tensor 'source:0' shape=(1, 600, 122) dtype=float32>]
        print("Frozen model outputs: ")
        print(frozen_func.outputs)
        # [<tf.Tensor 'Identity:0' shape=(1, 150) dtype=int32>]

        #for s in frozen_func.__dict__:
        #    print(s)

    #sys.exit(0)

    #df = pd.read_csv("Models/05_sound_to_text/202306191412/val.csv").values.tolist()
    dataset_val = pd.read_csv(model_dir+"/val.csv").values.tolist()
    op_reader=OppReader(opp_path=configs.opp_path, input_shape=configs.input_shape)
    sp_pad=SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0)

    print("configs.max_spectrogram_length:",configs.max_spectrogram_length)
    idx_to_char = configs.vocab
    DISP_F=True

    for img_path, label in dataset_val[:30]:
        print("img_path:",img_path)
        dt = op_reader.get_opp_data(img_path)
        img=dt.T
        if DISP_F==True:
            cv2.imshow('Source image', img)
            cv2.waitKey(300)

        w=dt.shape[0]
        #print("dt.shape:",dt.shape)
        #print("type(dt):",type(dt))
        dt_in,label=sp_pad(dt,None)
        dt_in = np.expand_dims(dt_in,axis=0)

        print("go pred!!")
        if LOAD_2==True:
            #text = model.generate(tf.constant(dt_in),1)
            #text = model.generate(dt_in,1)
            #print('text:',text)
            #text = model.pred(dt_in)
            #text = model.pred(tf.constant(dt_in))
            text = model(dt_in)
        if LOAD_3==True:
            #text = model.pred(dt_in)
            text = model(dt_in)

        if LOAD_4==True:
            # Get predictions for test images
            #frozen_graph_predictions = frozen_func(x=tf.constant(test_images))[0]
            text = frozen_func(source=tf.constant(dt_in))[0]

        #print('text:',text)
        labels = text[0]
        prediction = ""
        x=4
        for idx in labels[1:]:
            idx=int(idx)
            #print("idx:",idx)
            y=12+idx*2
            #if idx < len(idx_to_char):
            if idx < 49:
                prediction += idx_to_char[idx]
                if x < w:
                    cv2.circle(img,(x,y),2,(128),-1)
            x+=4
        #cv2.imwrite('./work/v_'+img_path, img)
        print("prediction:",prediction)
        if DISP_F==True:
            cv2.imshow('Detect image', img)
            cv2.waitKey(0)
