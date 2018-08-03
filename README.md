# ImageClassification
1. Download this repository
2. Add extension to 3D Slicer
3. Install Docker Community Edition
- https://store.docker.com/editions/community/docker-ce-desktop-windows
- Need to create an account
- Install using linux containers
4. Install tensorflow (optional)
- Only needed to visualize training metrics
-  https://www.tensorflow.org/versions/r1.8/install/
5. Build Retrain Container
- Open command prompt
- Build container using the following command:
  - docker build -t retrainimage \<Path to ImageClassification Directory\>/Models/retrainContainer
6. Collect training photos
- Open Collect_Training_Photos module in 3D Slicer
- Start Plus Config file
- Select existing model or create a new model
- Select image class or create new classes
- Keeping the object that you are trying to recognize in the image frame click Start Image Collection
  - For best results introduce as much variety in orientation and background conditions as possible
- Click Stop Image Collection
7. Retrain the network
- Click Retrain
- This may take up to 20min
- To visualize training:
  - Open command prompt
  - Execute the following command:
    - tensorboard --logdir \<Path to retrainContainer\>/\<Model_Name\>/trained_model/retrain_logs
  - Navigate in browser to \<host_name\>:6006
8. Copy trained_model folder from retrainContainer directory to classifierContainer directory
9. Build Classifier Container
- Open command prompt
- Build container using the following command:
  - docker build -t classifierimage \<Path to ImageClassification Directory\>/Models/classifierContainer
10. Run Classifier
- Open CNN_Image_Classifier module in 3D Slicer
- Select model
- Click Start
