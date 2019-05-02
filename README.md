# ImageClassification
1. Download this repository
2. Add extension to 3D Slicer
3. Add Slicer openCV, SlicerIGT and Sequences extension from Slicer extensions manager
4. Install Anaconda (https://www.anaconda.com/distribution/)
- ensure to add to path
5. Run setup_tensorflow_env.bat in command prompt
- \<Path to repository\>/setup_tensorflow_env.bat \<path to repository\> 
- e.g. c:/Users/ImageClassification/setup_tensorflow_env.bat c:/Users/ImageClassification
7. Collect training photos
- Open Collect_Training_Photos module in 3D Slicer
- Start Plus Config file
- Select existing model or create a new model
- Select image class or create new classes
- Keeping the object that you are trying to recognize in the image frame click Start Image Collection
  - For best results introduce as much variety in orientation and background conditions as possible
- Click Stop Image Collection
8. Retrain the network
- run StartRetrain.bat from command prompt
  - \<Path to repository\>/StartRetrain.bat \<Path to repository\> <\Model name\> \<Number of training steps\> \<Batch size\> 
  - e.g. c:/Users/ImageClassification/StartRetrain.bat c:/Users/ImageClassification SampleName 500 100
- To visualize training:
  - Open command prompt
  - Execute the following command:
    - tensorboard --logdir \<Path to retrainContainer\>/\<Model_Name\>/trained_model/retrain_logs
  - Navigate in browser to \<host_name\>:6006
11. Run Classifier
  Activate the anaconda environment:
     \>\>\> conda activate \<path to repository\>/Env
- Start classifier
     \>\>\> python \<path to repository\>/Models/classifier.py --model_name=\<Model name\>
- Open CNN_Image_Classifier module in 3D Slicer
- Select model
- Click Start
