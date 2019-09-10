# LAP Lifestyle Patterns Classification
A classifier based on convolutional neural networks to analyze egocentric images based on lifestyle patterns.

### CITATION:

P. Herruzo, L. Portell, A. Soto, B. Remeseiro (2019). Towards Objective Description of Eating, Socializing and Sedentary Lifestyle Patterns in Egocentric Images.

### USAGE:

- models: In this folder, we can find the code of the experiments we realized.
	- train_InceptionV3.py: This is the main file. We can find the code of the model used, the training part of the network, and the code for generating the output images, such as the confusion matrices. To run different models, such as multiclass, separated classifiers (food, seated or social), and powerset (training all 12 different combinations), we just need to uncomment the choosing data set to run. We can also decide between run the code in GPU or CPU (there is a flag for doing this). There is also a parameter FLAG_DEBUG that, if True, you will not run the whole experiment, just a few samples.
	- OtherClassifiers.ipynb: In this file is the code of obtaining the data of the output of the fully-connected layer and the traditional classifiers. You can choose a data set to run, and run the classifiers, doing grid-search over the k-fold cross validation. 
	- OtherClassifiers_PCA.ipynb: This file contains the code for non-deep-learning approaches. We can find Incremental PCA to reduce the dimensionality of the data set and the traditional classifiers.
    - grad_cam.py: code for computing gradCam maps.
    - outputs/weights: Folder to store the weights obtained by training the CNN. 
    - outputs/confusion_matrices: Folder to store the confusion matrices obtained after applying softmax during the CNN evaluation, and also the confusion matrices obtained with the traditional machine learning classifiers.

-scripts/dataset.py: This file contains the class that allow us to load data and perform different augmentation methods like flipping, adding Gaussian noise, and rotations with different
degrees in the range [−30,30].

-data/datasets: A folder containing the information of each data set. We have for each data set a folder, where we can find for instance the file classesID.txt, where we can see the meaning of each label. We also have training, validation and test files, where we can see all the links to the images that correspond to these splits and its corresponding label.

### DOWNLOAD:

You can download a [zip file of the source code](https://github.com/lauraportell/LAP-LifestylePatterns-Classification/archive/master.zip) directly.

Alternatively, you can clone it from GitHub as follows:

``` sh
$ git clone https://github.com/lauraportell/LAP-LifestylePatterns-Classification.git
```
