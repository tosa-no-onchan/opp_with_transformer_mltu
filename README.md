# opp_with_transformer_mltu  
  
ROS2 ML Path Planner with Transformer  
  
Obstacle Path Planner with Transformer and mltu  

#### 1. How to trainning.  

    $ python train_transformer_opp_mltu.py  

#### 2. model freeze  

    $ python test-model-freeze.py  

#### 3. predict  

    $ python inferencModel.py  
    unzip a.model_frozen.pb.zip
    use Models/test_opp/a.model_frozen.pb  
    train loss > 0.7  
  
#### 3. Env.  

   tensorflow==2.16.1  with cuda  
   keras==3.3.3  
   mlut==1.2.5  
   python 3.10.12  


#### reffrence page.    

  [ROS2 自作 Turtlebot3 による 草刈りロボット開発。#8 Transformer で経路計画をする。](http://www.netosa.com/blog/2024/09/ros2-turtlebot3-8-thetastarplanner.html)
