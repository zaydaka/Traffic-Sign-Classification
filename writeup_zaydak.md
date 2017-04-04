#**Traffic Sign Recognition** 

##Andrew Zaydak

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report (this)


[//]: # (Image References)

[image1]: ./writeup_images/train_image1.jpg "Training Image 1"
[image2]: ./writeup_images/train_image2.jpg "Training Image 2"
[image3]: ./writeup_images/train_image3.jpg "Training Image 3"
[image4]: ./writeup_images/training_hist.jpg "Testing Class Distribution"
[image5]: ./writeup_images/validation_hist.jpg "Validation Class Distribution"
[image6]: ./writeup_images/network_architecture.jpg "Network Architecture"
[image7]: ./writeup_images/traffic_1.jpg "Traffic Image 1"
[image8]: ./writeup_images/traffic_2.jpg "Traffic Image 2"
[image9]: ./writeup_images/traffic_3.jpg "Traffic Image 3"
[image10]: ./writeup_images/traffic_4 "Traffic Image 4"
[image11]: ./writeup_images/traffic_5 "Traffic Image 5"
[image12]: ./writeup_images/traffic_6 "Traffic Image 6"


---

####1. Overview

This project is to build a classifier for German traffic sign images using a convolutional neural network.  It was coded in Python using Tensorflow.
The following files are included:

* Traffic_Sign_classifier.ipynb	(classifier code)
* README.md (readme file for project)
* writeup_zaydak.md (this)
* data/ (includes example images used to test the network)
* writeup_images/ (images for this writeup)

###1. Data Set Summary & Exploration

The German Traffic Sign data was broken into three subsets:

* Training - Used to train the network (34799 images)
* Validation - A Validation set	(4410 images)
* Testing - A testing set (12630 images)

The above statistics were computed using the numpy's array shape member.  Matplot lib was also used to visiualize some of the data.

In this data set there are 43 different classes (labels) of images.  Each image is a 32 x 32 pixel images with 3 color channels (RGB). Example images of this data set are shown below.

![alt text][image1]
![alt text][image2]
![alt text][image3]

The following figure show the distributions of the classes for both the training and validation data sets.  It is interesting to note that the distributions are nearly identical between the two.  This is by design and shows that the validation set is a good representation of the training data set.  That being said, classes within these subsets are not equally represented.  For example, calss 43 representing X, has around 250 examples where classes have close to 2000.

![alt text][image4]
![alt text][image5]


###2.Design and Test a Model Architecture

####2.1 Preprocessing

Image preprocessing code can be found in the fourth code cell of the IPython notebook and includes the following steps:

* First, conversion to grayscale
* Second, image standardization
* Third, gaussian blurring

Grayscale conversion was used because it seemed to improve image classification.  Although some of the color information is destroyed in this process, I believe that it helps create a color invariance that prevents missclassification of images that are shadows and other color distortions.  Although most of the images in the training data already are somewhate blurry, performming Gaussian blurring helped remove high frequency information in the images and seemed to slightly help training.  Finally, image standardization was performed on the images to give them a zero mean and unity norm.

The preprocessing steps were implemented to operations on the Tensorflow graph for fast computation.

Grayscale conversion was done using tf.image.rgb_to_grayscale function.  Gaussian blurring was done by using the tf.nn.conv2d with a constant 3x3 kernal that represents the Gaussian. Finally, tf.map_fn was used to batch map the tf.image.per_image_standardization function to each image in the data set.

The actual execution of this part of the Tensorflow graph was in the IPython notebook code cell 8.


####2.2 Data Set Build Up

As described in Section 1, the data set was split into 34799 training images and 4410 validation images.  Both had a similar distribution of classes.  The data sets were pre-split into these two sets however were shuffled at the start of the training and before each epoch.  Additional data was not added to the data set nor was any image augmentation used to expand the data set.  It is however expected that performance would have been improved had small random rotations and other affine projections of the images had been used to expance the data set.


###3. Network Architecture

The network architecture is coded in the fifth code cell of the IPython notebook.  The architecture design used the LeNet network as a starting point.The figure below graphically shows the network architecture.

![alt text][image5]

* The inital 32x32x1 preprocessed image first passes through a convolution layer consisting of 20 5x5 kernel filters.  followed by a ReLu activation function and a max pooling downsampling operation.
* The second convolutional layer consisted of 40 5x5 kernel filters followed by a ReLu activation function and max pooling downsampling.  The output is a 5x5x40 tensor.
* Next, the tensor was flattened to a 1000x1 vector and passed through a full connected layer with a ReLu activation function.  This layer outputs a 400x1 vector.
* Another fully connected layer adn ReLu activation function compresses the information down to a 100x1 vector.
* Finally, a fully connected layer outputs a 43x1 vectore of logits.

The 5x5 kernel filters all used 1x1x1x1 strides, similar to that of the LeNet network.  The second convolution layer expanded from LeNet which provided better validation results.  To compensate for the lager number of outputs of the second convolution layer, the fully connected layers of the original network were also expanded.


####4. Training

Network training code is located in ipython notebook code cells 9 and 10.  Then network output is a 43x1 vectore of logits.  The chosen loss function is the softmax cross entropy between the logits and the training labels (which are encoded a one hot vector).  The training optimizer algorithm was chosen to be the Adam algorithm.  Reading literature and from trial-and-error this seemed to be a good approach.

Training was computed using a Nividia GTX 1080 GPU which could finish each epoch in less than a second using batch sizes of 128.  Because of the advantage of using a GPU for processing, it was decided to increase the number of epochs to 20 and prevent over fitting using a dropout operation after each layer with a 40% chance of dropout and a small learning rate of 0.001.  The training data was shuffled each epoch and the accuracy of the validation classification was outputed.


####5. Results

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 96.8%
* test set accuracy of 95.1%

As discussed above, the starting architecture was based on the LeNet classifier with the exception of the preprocessing operations.  This is a good starting point because the LeNet network performs well on the MNIST data set.  For most image classification problems a convolution network is a good approach since it presernves some spatial information in the data.  For the most part, a trial and error appraoch was taken to adjust the network and it was found that expanding the layers to include more nodes (in the fully connected layers) and more kernel filters (in the convolution layers) improved the model.  This is expected as there are more classes in the German traffic sign data set compaired to the MNIST data set.  Image preprocessing descisions seemed to have a higher impact on the validation accuracy than network archetectur changes.  The final network performed better on the training set than the validation set.  This is expected however the training accuracy was about 3% better than validation indicating that the network could be slightly over fitting to the data dispite using dropout.


###Test a Model on New Images

Six German traffic signs were found on the internet.  The images were downloaded and croped into 32x32 RGB color images.  In general, the image quality of these images are far better than the training images.  The edges are sharper and the colors are brighter.  The preprocessing standardization and blurring helps bring these images closer to what the network expects.


![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] 

The pedestrian image is much darker than the others and the road work image has a very structered backgroun therefore if might be expected that these may be misclassified by the network.

The network, however, correctly classified all six of the images except for the pedestrian image as seen in code block thirteen.  After further investigation, the network may have had trouble with this image because of the skewness of the image.  Data augmentation during the training process to expand that data set might aid the network in cases such as this.

With five out of the six example images correctly calssified, the network performed with an accuracy of 83%.  This is far less than the testing accuracy however not much can be determined from such a small data set.

The misclcassified pedestrian image was classified as a 'No passing for vehicles over 3.5 metric tons' sign.  Interestingly enough, the correct prediction was not even in the top five softmax probabilities.


Here a summary results of the predictions:

| Image			        |     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| Speed Llimit 120      	| Speed Llimit 120    				| 
| Bummpy Road     		| Bummpy Road  					|
| No Entry			| No Entry					|
| Pedestrians	      		| No passing for vehicles over 3.5 metric tons	|
| Priority Road			| Priority Road      				|
| Road Work			| Road Work    					|




####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 