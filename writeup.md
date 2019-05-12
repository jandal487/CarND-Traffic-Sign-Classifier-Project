# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[img1]: ./test_images/curve_right.png
[img2]: ./test_images/no_passing.png
[img3]: ./test_images/priority_road.png
[img4]: ./test_images/roundabout.png
[img5]: ./test_images/stop.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jandal487/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is = 34799
* The size of the validation set is = 4410
* The size of test set is = 12630
* The shape of a traffic sign image is = (32, 32, 3)
* The number of unique classes/labels in the data set is = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing number of images distributed across each class label:
![](https://github.com/jandal487/CarND-Traffic-Sign-Classifier-Project/blob/master/histogram_imgs/hist_train.png)

![](https://github.com/jandal487/CarND-Traffic-Sign-Classifier-Project/blob/master/histogram_imgs/hist_test.png)

![](https://github.com/jandal487/CarND-Traffic-Sign-Classifier-Project/blob/master/histogram_imgs/hist_valid.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I have 2 step pre-processing:

1. Shuffling: I shuffeled the training set as shown in code cell number 5: `shuffle(X_train, y_train)`
2. Normalization: Then I normalized every image by subtracting 128 from every pixel and then dividing by 128. This operation results in float32 variable for every image.
`n = np.float32(128)`
`X_train=(X_train.astype(np.float32)-n)/n`


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution layer 1     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU	1				|												|
| Max pooling 1	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution layer 2     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU	2				|												|
| Max pooling 2	      	| 2x2 stride,  outputs 5x5x32 				|
| Flatten Layer	    | output 800     									|
| Fully connected layer 1		| input 800, output 512        									|
| RELU	3				|												|
| DROPOUT 1				|						0.75						|
| Fully connected layer 2		| input 512, output 128        									|
| RELU	4				|												|
| DROPOUT 2				|						0.75						|
| Output Layer		| input 128, output 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The hyperparameters are as following:
* Learning rate = 0.01
* EPOCH = 10
* DROPOUT = 0.75
* OPTIMIZATION Algorithm = Adam with default TF params

In order to find suitable hyperparameter values I executed the training process several times untill I was sure not overfitting the training data set. I think drop out really plays a great role in regularization and avoiding overfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.940
* test set accuracy of 0.934

First of all I used the same LeNet architecture that was discussed in the lectures. I just made sure I have the correct input and output dimensions. This resulted in very low validation training. Therefore I tried the following things:
* I increased the EPOCH size to 15 and even further to know how soon I overfit the dataset
* Then I also tweeked the learning rate
* Then I changed the the size of the BATCH
* Then I changed the dimensions of the convolution layers and increased the depth of each layer
* I also increased layers in the network but was easily overfitting the dataset with more than 2 convolution layers.
* Then I added DROPOUT Layers and it resulted in very good training and validation accuracy
* Finally, when I was satisfied with the performance of the model on training and validation sets, I checked model's accuracy on the test set. And it also resulted very good.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the [web](http://www.gettingaroundgermany.info/regeln.shtml):

![](https://github.com/jandal487/CarND-Traffic-Sign-Classifier-Project/blob/master/histogram_imgs/hist_test.png)

![alt text][img1] ![alt text][img2] ![alt text][img3] 
![alt text][img4] ![alt text][img5]

The corresponding actual labels for these images are: `[20, 9, 12, 40, 14]`

I found several images on the internet with different challenges. I think the following difficulties my model can face depending on the image:

* The contrast of image colors and over visibility
* Distortion in the shape of the traffice sign
* In case of the image is very blurry

I think these can really challenge general capability of my model.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

This can be seen in the notebook for details. In short, my model was 100% accurate on the given images and every image was correctly classified.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Dangerous curve to the right      		| Dangerous curve to the right   									| 
| No passing     			| No passing 										|
| Priority road					| Priority road											|
| Roundabout mandatory	      		| Roundabout mandatory					 				|
| Stop			| Stop     							|

However this is not very significant performance, as my test set had only 5 images. Therefore, true generally capability would be know on various amount of unseen and difficult images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

* Image 1: Dangerous curve to the right. The model is confident as there is big difference in 1st and 2nd max probabilities.
  Top 5 probabilities: 
  [  9.99999285e-01   6.62446382e-07   6.24112480e-12   8.29648125e-17    1.01430549e-17]
  Corresponding labels: 
  [20 23 28 41 19]
  
* Image 2: No passing. The model is confident as there is big difference in 1st and 2nd max probabilities.
  Top 5 probabilities: 
  [  9.99908805e-01   9.11617462e-05   3.13678239e-09   9.21112797e-10    4.28456076e-10]
  Corresponding labels: 
  [ 9 41 23 20 35]
  
* Image 3: Priority road. The model is confident as there is big difference in 1st and 2nd max probabilities.
  Top 5 probabilities: 
  [  1.00000000e+00   9.43388722e-23   8.35492761e-24   1.06233551e-24    1.42610755e-25]
  Corresponding labels: 
  [12 11 42 10 41]
  
* Image 4: Roundabout mandatory. The model is confident as there is big difference in 1st and 2nd max probabilities.
  Top 5 probabilities: 
  [  1.00000000e+00   1.39741008e-09   7.28302488e-11   3.95354617e-11    6.40692534e-12]
  Corresponding labels: 
  [40  7 11 42 37]
  
* Image 5: Stop. The model is confident as there is big difference in 1st and 2nd max probabilities.
  Top 5 probabilities: 
  [  1.00000000e+00   1.39741008e-09   7.28302488e-11   3.95354617e-11    6.40692534e-12]
  Corresponding labels: 
  [40  7 11 42 37]

As it can be seen that for these 5 images the model confidentently classified them correctly. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


