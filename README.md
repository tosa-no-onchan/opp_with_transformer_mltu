# opp_with_transformer_mltu  
  
ROS2 ML Path Planner with Transformer  
  
Obstacle Path Planner with Transformer and mltu  

#### 1. How to trainning.  

    $ python train_transformer_opp_mltu.py  

#### 2. model freeze  

    $ python test-model-freeze.py  

#### 3. Convert a.model_frozen.pb to ONNX  

    $ python -m tf2onnx.convert --input ./Models/test_opp/a.model_frozen.pb --output a.model.onnx --outputs Identity:0 --inputs source:0  

````
2024-12-02 17:56:15,974 - INFO - Using tensorflow=2.16.2, onnx=1.17.0, tf2onnx=1.16.1/15c810
2024-12-02 17:56:15,975 - INFO - Using opset <onnx, 15>
2024-12-02 17:56:26,784 - INFO - Computed 0 values for constant folding
2024-12-02 17:57:02,417 - INFO - Optimizing ONNX model
2024-12-02 18:01:05,781 - INFO - After optimization: Add -592 (7494->6902), Cast -2695 (3288->593), Const -18743 (19479->736), GlobalAveragePool +1802 (0->1802), Identity -903 (903->0), MatMul -592 (4346->3754), ReduceMean -1802 (1802->0), Reshape -738 (2547->1809), Transpose -605 (2999->2394)
2024-12-02 18:01:07,629 - INFO - 
2024-12-02 18:01:07,630 - INFO - Successfully converted TensorFlow model ./Models/test_opp/a.model_frozen.pb to ONNX
2024-12-02 18:01:07,630 - INFO - Model inputs: ['source:0']
2024-12-02 18:01:07,630 - INFO - Model outputs: ['Identity:0']
2024-12-02 18:01:07,630 - INFO - ONNX model is saved at a.model.onnx
````

  [Frozen Graph TensorFlow 2.x](https://github.com/leimao/Frozen-Graph-TensorFlow/tree/master/TensorFlow_v2)  


#### 4. predict  

    $ python inferencModel.py  
    unzip a.model.onnx.zip
    use a.model.onnx  
    train loss > 0.7  
  
#### 5. Env.  

   Ubuntu Mate 22.04  
   Virtual Env  
   tensorflow==2.16.1  with cuda  
   keras==3.3.3  
   mlut==1.2.5  
   python 3.10.12  


#### 6. reffrence page.    

  [ROS2 自作 Turtlebot3 による 草刈りロボット開発。#8 Transformer で経路計画をする。](http://www.netosa.com/blog/2024/09/ros2-turtlebot3-8-thetastarplanner.html)  

  [Trainning Data](https://huggingface.co/datasets/tosa-no-onchan/opp)  

  [How to make Trainning Data.](https://github.com/tosa-no-onchan/opp_with_transformer_cpp)  
