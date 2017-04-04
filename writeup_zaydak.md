#**Traffic Sign Recognition** 

##Andrew Zaydak

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report (this)



[//]: # (Image References)

[image1]: ./writeup_images/train_image_1.jpg "Training Image 1"
[image2]: ./writeup_images/train_image_2.jpg "Training Image 2"
[image3]: ./writeup_images/train_image_3.jpg "Training Image 3"
[image4]: ./writeup_images/training_hist.jpg "Testing Class Distribution"
[image5]: ./writeup_images/validation_hist.jpg "Validation Class Distribution"
[image6]: ./writeup_images/network_architecture.JPG "Network Architecture"
[image7]: ./writeup_images/traffic_1.jpg "Traffic Image 1"
[image8]: ./writeup_images/traffic_2.jpg "Traffic Image 2"
[image9]: ./writeup_images/traffic_3.jpg "Traffic Image 3"
[image10]: ./writeup_images/traffic_4.jpg "Traffic Image 4"
[image11]: ./writeup_images/traffic_5.jpg "Traffic Image 5"
[image12]: ./writeup_images/traffic_6.jpg "Traffic Image 6"
[image13]: ./writeup_images/vis.jpg "Conv-1 Feature Map"


---

#### Overview

This project is to build a classifier for German traffic sign images using a convolutional neural network.  It was coded in Python using Tensorflow.
The following files are included:

* Traffic_Sign_classifier.ipynb	(classifier code)
* README.md (readme file for project)
* writeup_zaydak.md (this)
* data/ (includes example images used to test the network)
* writeup_images/ (images for this write-up)


### 1. Data Set Summary & Exploration

The German Traffic Sign data was broken into three subsets:

* Training - Used to train the network (34799 images)
* Validation - A Validation set	(4410 images)
* Testing - A testing set (12630 images)

The above statistics were computed using the numpy's array shape member.  Matplot lib was also used to visualize some of the data.

In this data set there are 43 different classes (labels) of images.  Each image is a 32 x 32 pixel images with 3 color channels (RGB). Example images of this data set are shown below.

![alt text][image1]
![alt text][image2]
![alt text][image3]

The following figure shows the distributions of the classes for both the training and validation data sets.  It is interesting to note that the distributions are nearly identical between the two.  This is by design and shows that the validation set is a good representation of the training data set.  That being said, classes within these subsets are not equally represented.  For example, class 43 representing X, has around 250 examples where classes have close to 2000.

![alt text][image4]
![alt text][image5]



### 2.Design and Test a Model Architecture

#### 2.1 Preprocessing

Image preprocessing code can be found in the fourth code cell of the IPython notebook and includes the following steps:

* First, conversion to grayscale
* Second, image standardization
* Third, Gaussian blurring

Grayscale conversion was used because it seemed to improve image classification.  Although some of the color information is destroyed in this process, I believe that it helps create a color invariance that prevents misclassification of images that are shadows and other color distortions.  Although most of the images in the training data already are somewhat blurry, performing Gaussian blurring helped remove high frequency information in the images and seemed to slightly help training.  Finally, image standardization was performed on the images to give them a zero mean and unity norm.

The preprocessing steps were implemented to operations on the Tensorflow graph for fast computation.

Grayscale conversion was done using tf.image.rgb_to_grayscale function.  Gaussian blurring was done by using the tf.nn.conv2d with a constant 3x3 kernel that represents the Gaussian. Finally, tf.map_fn was used to batch map the tf.image.per_image_standardization function to each image in the data set.

The actual execution of this part of the Tensorflow graph was in the IPython notebook code cell 8.


#### 2.2 Data Set Build Up

As described in Section 1, the data set was split into 34799 training images and 4410 validation images.  Both had a similar distribution of classes.  The data sets were pre-split into these two sets however were shuffled at the start of the training and before each epoch.  Additional data was not added to the data set nor was any image augmentation used to expand the data set.  It is however expected that performance would have been improved had small random rotations and other affine projections of the images had been used to expand the data set.


### 3. Network Architecture

The network architecture is coded in the fifth code cell of the IPython notebook.  The architecture design used the LeNet network as a starting point. The figure below graphically shows the network architecture.

![alt text][image6]

* The initial 32x32x1 preprocessed image first passes through a convolution layer consisting of 20 5x5 kernel filters followed by a ReLu activation function and a max pooling down sampling operation.
* The second convolutional layer consisted of 40 5x5 kernel filters followed by a ReLu activation function and max pooling down sampling.  The output is a 5x5x40 tensor.
* Next, the tensor was flattened to a 1000x1 vector and passed through a full connected layer with a ReLu activation function.  This layer outputs a 400x1 vector.
* Another fully connected layer and ReLu activation function compresses the information down to a 100x1 vector.
* Finally, a fully connected layer outputs a 43x1 vector of logits.

The 5x5 kernel filters all used 1x1x1x1 strides, similar to that of the LeNet network.  The second convolution layer expanded from LeNet which provided better validation results.  To compensate for the lager number of outputs of the second convolution layer, the fully connected layers of the original network were also expanded.


#### 4. Training

Network training code is located in IPython notebook code cells 9 and 10.  Then network output is a 43x1 vector of logits.  The chosen loss function is the softmax cross entropy between the logits and the training labels (which are encoded a one hot vector).  The training optimizer algorithm was chosen to be the Adam algorithm.  Reading literature and from trial-and-error this seemed to be a good approach.

Training was computed using a NVidia GTX 1080 GPU which could finish each epoch in less than a second using batch sizes of 128.  Because of the advantage of using a GPU for processing, it was decided to increase the number of epochs to 20 and prevent over fitting using a dropout operation after each layer with a 40% chance of dropout and a small learning rate of 0.001.  The training data was shuffled each epoch and the accuracy of the validation classification was outputted.


#### 5. Results

The code for calculating the accuracy of the model is located in the ninth cell of the IPython notebook.

My final model results were:
* Training set accuracy of 99.7%
* Validation set accuracy of 96.8%
* Test set accuracy of 95.1%

As discussed above, the starting architecture was based on the LeNet classifier with the exception of the preprocessing operations.  This is a good starting point because the LeNet network performs well on the MNIST data set.  For most image classification problems a convolution network is a good approach since it preserves some spatial information in the data.  For the most part, a trial and error approach was taken to adjust the network and it was found that expanding the layers to include more nodes (in the fully connected layers) and more kernel filters (in the convolution layers) improved the model.  This is expected as there are more classes in the German traffic sign data set compared to the MNIST data set.  Image preprocessing decisions seemed to have a higher impact on the validation accuracy than network architecture changes.  The final network performed better on the training set than the validation set.  This is expected however the training accuracy was about 3% better than validation indicating that the network could be slightly over fitting to the data despite using dropout.


### Test a Model on New Images

Six German traffic signs were found on the internet.  The images were downloaded and cropped into 32x32 RGB color images.  In general, the image quality of these images is far better than the training images.  The edges are sharper and the colors are brighter.  The preprocessing standardization and blurring helps bring these images closer to what the network expects.


![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] 

The pedestrian image is much darker than the others and the road work image has a very structured background therefore if might be expected that these may be misclassified by the network.

The network, however, correctly classified all six of the images except for the pedestrian image as seen in code block thirteen.  After further investigation, the network may have had trouble with this image because of the skewness of the image.  Data augmentation during the training process to expand that data set might aid the network in cases such as this.

With five out of the six example images correctly classified, the network performed with an accuracy of 83%.  This is far less than the testing accuracy however not much can be determined from such a small data set.

The misclassified pedestrian image was classified as a 'Traffic Signals' sign.  Interestingly enough, the correct prediction was third in the list of top five probabilities with a 0.005 probability.


Here a summary results of the predictions:


| Image			        |     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| Speed Llimit 120      	| Speed Llimit 120    				| 
| Bummpy Road     		| Bummpy Road  					|
| No Entry			| No Entry					|
| Pedestrians	      		| Traffic Signals				|
| Priority Road			| Priority Road      				|
| Road Work			| Road Work    					|



The softmax probabilities were computed in code cell 15.  For all example images, the network was very confident in its prediction except for the pedestrians sign (which it got wrong) and the road work sign.  The road work sign example produced the least confidant results.  Below are tables showing the probabilities for each example image.


| Probability         	|     Prediction (Speed Limit 120)	        					| 
|:---------------------:|:---------------------------------------------:| 
| .996         			| Speed Limit 120  									| 
| .023     				| Speed Limit 100 										|
| .003					| Speed Limit 80										|
| .003	      			| Speed Limit 30					 				|
| .003				    | Speed Limit 20     							|


| Probability         	|     Prediction (Bummpy Road)	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Bummpy Road									| 
| .033     				| Bicycles Crossing										|
| .000					| Traffic Signals										|
| .000	      			| Slippery Road					 				|
| .000				    | Childern Corssing    							|

| Probability         	|     Prediction (No Entry)	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Entry									| 
| .000     				| Stop										|
| .000					| End of All Speed and Passing Limits										|
| .000	      			| Turn Left Ahead					 				|
| .000				    | Turn Right Ahead    							|

| Probability         	|     Prediction (Pedestrians)	        					| 
|:---------------------:|:---------------------------------------------:| 
| .681         			| Traffic Signals									| 
| .262     				| General Caution										|
| .051					| Pedestrians										|
| .003	      			| Road Narrows on the Right					 				|
| .001				    | Roundabout Mandatory   							|

| Probability         	|     Prediction (Priority Road)	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority Road									| 
| .000     				| Roundabout Mandatory									|
| .000					| Keep Right									|
| .000	      			| No Passing					 				|
| .000				    | Turn Right Ahead    							|

| Probability         	|     Prediction (Road Work)	        					| 
|:---------------------:|:---------------------------------------------:| 
| .264         			| Road Work									| 
| .262     				| General Caution									|
| .197					| Right-of-way at the Next Intersection									|
| .056	      			| Pedestrian					 				|
| .045				    | Traffic Signals    							|


The bummpy road sign was chosen to display the feature maps of the first convolution layer and is shown below.

![alt text][image13]

The above images show the weight response to an example input image of a 'bumpy road' sign. There were 20 kernels used in this convolution layer. All of the kernels filters were activated for this input.  Most of the other filters activated on diagonal lines or edges. In this case if would be from the diamond shape of the sign itself.  These kernels act as an edge detection.  Some of the kernels, for example kernel 11, not only picked up on the shape of the sign but also the inner contents of the sign.