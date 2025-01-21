# opp_with_transformer_mltu/inferencModel_tf-lite.py
# https://www.tensorflow.org/lite/guide/inference?hl=ja

import os
import sys
import typing
import numpy as np

from configs import ModelConfigs
import pandas as pd
from tools_mltu import *

from scipy.special import softmax
import cv2
import tensorflow as tf

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    test_date="test_opp"
    model_dir="Models/"+test_date
    #configs = BaseModelConfigs.load(model_dir+"/configs.yaml")
    configs = ModelConfigs.load("Models/"+test_date+"/configs.yaml")

    print("configs.model_path:",configs.model_path)
    if os.path.exists("./work")==False:
        os.mkdir("./work")

    #sys.exit(0)

    LOAD_1=True    # tf-lite

    if LOAD_1==True:

        # Load the TFLite model and allocate tensors.
        try:
            # Load the TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path="a.model.tflite")
            interpreter.allocate_tensors()

        except Exception as e:
            print(e)
            sys.exit(0)

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    #print("model.__dict__:",model.__dict__)
    #sys.exit(0)

    #df = pd.read_csv("Models/05_sound_to_text/202306191412/val.csv").values.tolist()
    dataset_val = pd.read_csv(model_dir+"/val.csv").values.tolist()

    op_reader=OppReader(opp_path=configs.opp_path, input_shape=configs.input_shape)

    sp_pad=SpectrogramPadding_my(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0)

    print("configs.max_spectrogram_length:",configs.max_spectrogram_length)
    idx_to_char = configs.vocab

    DISP_F=True

    for img_path, label in dataset_val:
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
        #print("dt_in.shape:",dt_in.shape)
        #print("type(dt_in):",type(dt_in))
        #print("dt_in.dtype:",dt_in.dtype)
        print("go pred!!")

        if LOAD_1==True:
            # Test the model on random input data.
            input_shape = input_details[0]['shape']
            interpreter.set_tensor(input_details[0]['index'], dt_in)

            interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            preds = interpreter.get_tensor(output_details[0]['index'])

            print("preds.shape:",preds.shape)
            # preds.shape: (1, 150)
            #sys.exit(0)
        x=4
        prediction = ""
        for idx in preds[0][1:]:
            #print("idx:",idx)
            y=12+idx*2
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



    
    