Analysis and classification of medical image processing
It is a binary classification model that classifies 2 benign liver lesions namely HEMANGIOMA (HEM) and FOCAL NODULAR HYPERPLASIA (FNH).
Ultrasound images of both the classes were taken from a publicly available dataset present at ultrasoundcases.com. 
Although the main classification is not applied over the complete ultrasound image, rather ROI's are extracted from the images using GUI designed in Python. (ROI's stand for Region of interest)
Texture properties were applied on these Roi's , which resulted in few excel files with values of different texture properties of them. 
These excel files were fed to a SVM model for their binary classification.
THe model is written in MATLAB , while the UI for ROI cropping is done using Python.
This model also utilises an external library called LIBSVM, which performs Faster and Efficient Hyperparameter tuning of the model. 
Resultin in  ->
