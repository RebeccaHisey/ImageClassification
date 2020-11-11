cmake_minimum_required(VERSION 3.5)

project(ImageClassification)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/ImageClassification")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "John Doe (AnyWare Corp.)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/ImageClassification.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/ImageClassification/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(RecordTrainingImages)
add_subdirectory(TrainClassifier)
add_subdirectory(ClassifyImages)
add_subdirectory(CNN_Image_Classifier)
add_subdirectory(Collect_Training_Images)
add_subdirectory(RetrainCNN)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
