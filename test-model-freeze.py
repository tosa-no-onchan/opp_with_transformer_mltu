"""
test-model-freeze.py

https://jimmy-shen.medium.com/how-to-freeze-graph-in-tensorflow-2-x-3a3238c70f19

https://www.tensorflow.org/guide/saved_model?hl=ja


convert frozen.pb to model.onnx

$ python -m tf2onnx.convert --input ./Models/test_opp/a.model_frozen.pb --output a.model.onnx --outputs Identity:0 --inputs source:0

"""

import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import keras

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from configs import ModelConfigs
from tensorflow.python.framework import convert_to_constants

test_date="test_opp"
model_dir="Models/"+test_date
configs = ModelConfigs.load("Models/"+test_date+"/configs.yaml")

input_model = "Models/test_opp/a.model"
output_model = "Models/test_opp/a.model.freeze.pb"

#to tensorflow lite
#converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
#tflite_quant_model = converter.convert()

LOAD_3=False             # use Keras saved model --> NG
                        #  ,but trained model with model.py use model = Model(inputs=inputs, outputs=output)
LOAD_4=False
LOAD_5=True            # use TF saved model  -->  OK
                        #  ,but trained model with model.py use model = Model(inputs=inputs, outputs=output)
                        # and 
                        # Frozen model inputs: [<tf.Tensor 'inputs:0' shape=(None, 600, 122) dtype=float32>]
                        # Frozen model outputs: [<tf.Tensor 'Identity:0' shape=(None, 300, 53) dtype=float32>]

base_dir='Models/test_opp'

if LOAD_3==True:
        # Keras saved model
        # https://www.tensorflow.org/lite/convert/concrete_function?hl=ja
        print("LOAD_3")

        from train_transformer_opp_mltu import CustomSchedule
        objs_x={
                #"CTCloss":CTCloss(),
                #"CERMetric":CERMetric(configs.vocab),
                "CustomSchedule":CustomSchedule
        }
        model = keras.models.load_model(configs.model_path+'/a.model.keras',custom_objects=objs_x,safe_mode=False)

        # 具象関数を Keras モデルから取得
        run_model = tf.function(lambda x : model(x))

        # 具象関数を保存
        concrete_func = run_model.get_concrete_function(
                tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        #tf.saved_model.save(model, configs.model_path+'/a.model_frozen.pb', concrete_func)

        # Get frozen ConcreteFunction    
        constantGraph = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

        print("Frozen model inputs: ")
        print(constantGraph.inputs)
        # [<tf.Tensor 'x:0' shape=(None, 600, 122) dtype=float32>]
        print("Frozen model outputs: ")
        print(constantGraph.outputs)
        # [<tf.Tensor 'Identity:0' shape=(None, 300, 53) dtype=float32>]

        tf.io.write_graph(graph_or_graph_def=constantGraph.graph, logdir=configs.model_path, name="a.model_frozen.pb",as_text=False) 

        # Get frozen ConcreteFunction    
        #frozen_func = convert_variables_to_constants_v2(full_model)    
        #frozen_func.graph.as_graph_def()
       
        #concrete_func.inputs[0].set_shape([1, 600, 122])
        #converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        #tflite_model = converter.convert()

if LOAD_4==True:
        print("LOAD_4")


if LOAD_5==True:        
        # Tensorflow saved model ->  OK
        print("LOAD_5")
        # https://www.tensorflow.org/lite/convert/concrete_function?hl=ja

        # https://github.com/leimao/Frozen-Graph-TensorFlow/tree/master/TensorFlow_v2
        #  example_1.py

        model = tf.saved_model.load(configs.model_path+'/a.model')
        concrete_func = model.signatures[
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        # Get frozen ConcreteFunction    
        constantGraph = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

        print("Frozen model inputs: ")
        print(constantGraph.inputs)
        # [<tf.Tensor 'source:0' shape=(1, 600, 122) dtype=float32>]
        print("Frozen model outputs: ")
        print(constantGraph.outputs)
        # [<tf.Tensor 'Identity:0' shape=(1, 150) dtype=int32>]
        tf.io.write_graph(graph_or_graph_def=constantGraph.graph, logdir=configs.model_path, name="a.model_frozen.pb",as_text=False) 

