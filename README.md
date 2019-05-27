# ImageClassification
1. Download this repository
2. Download latest version of Slicer (Nightly)
3. Add extension to 3D Slicer
  - edit -\> Application Settings -\> modules
  - add paths to RetrainCNN, CNN_Image_Classifier and Collect_Training_Images
4. Add Slicer openCV, SlicerIGT and Sequences extension from Slicer extensions manager
5. Install opencv, tensorflow and tensorflow_hub using pip
  -\>\>\> import pip
  -\>\>\> from pip.\_internal import main as pipmain
  -\>\>\> pipmain(['install','tensorflow'])
  -\>\>\> pipmain(['install','tensorflow_hub'])
  -\>\>\> pipmain(['install','opencv-python'])
  - May need to run Slicer as administrator in Windows 
6. Collect training photos
- Open Collect_Training_Photos module in 3D Slicer
- Start Plus Config file
- Select existing model or create a new model
- Select image class or create new classes
- Keeping the object that you are trying to recognize in the image frame click Start Image Collection
  - For best results introduce as much variety in orientation and background conditions as possible
- Click Stop Image Collection
7. Retrain the network
- Open RetrainCNN module in 3D Slicer
- Select the Model that you would like to retrain
- Select all training parameters in the Advanced Options tab
- Click Retrain
8. Run Classifier
- Open CNN_Image_Classifier module in 3D Slicer
- Start Plus Config file
- Select model
- Click Start
