# ImageClassification
1. Download this repository
2. Add extension to 3D Slicer
3. Add Slicer openCV, SlicerIGT and Sequences extension from Slicer extensions manager
4. Install Anaconda (https://www.anaconda.com/distribution/)
5. Run setup_tensorflow_env.bat in command prompt
- <path to repository>/setup_tensorflow_env.bat <path to repository> 
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
- Click Retrain
- This may take up to 20min
- To visualize training:
  - Open command prompt
  - Execute the following command:
    - tensorboard --logdir \<Path to retrainContainer\>/\<Model_Name\>/trained_model/retrain_logs
  - Navigate in browser to \<host_name\>:6006
11. Run Classifier
- Open CNN_Image_Classifier module in 3D Slicer
- Select model
- Click Start
