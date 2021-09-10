# **Traffic Sign Recognition** 

## Writeup


[//]: # (Image References)

[image1]: ./examples/image1.jpg "Visualization"
[image2]: ./examples/image2.jpg "More Visualisation"
[image3]: ./examples/image3.jpg "More Visualisation"
[image4]: ./examples/image4.png "Lenet Model"
[image5]: ./examples/image5.png "Before Preprocessing"
[image6]: ./examples/image6.png "after preprocessing"
[image7]: ./examples/image7.png "Better Model Architecture"
[image8]: ./examples/image8.png "Testing on new images"
[image9]: ./examples/image9.png "Softmax Possibilities"
 
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 Training examples
* The size of the validation set is 4,410 Validation examples
* The size of test set is 12,630 Testing examples
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how evenly the various classes are distributed. We used the Matplotlib's hist method.

![alt text][image1]

Further visulaization is also done where we can see the images being represented far more frequently than others.
The class labeled 0 ("Speed limit (20km/h)") in the dataset is the represented, with only 180 examples.
![alt text][image2]

On the other hand, the class labeled 2 ("Speed limit (50km/h)") has 1980 examples, which is over an order of magnitude difference!
![alt text][image3]

As shown further, we find that classes that are less represented tend to be misclassified more often.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
Prior to this project, we built a Lenet-5 implementation,  a  simple model capable of performing very well on the MNIST handwriting sample dataset. This used the following model architecture.

![alt text][image4]

Training the data on this model yielded ~89% accuracy on the validation set and a little over 90% on the test set.

### Preprocessing:

Below is a random sampling of images before preprocessing. We can see that some of the images are very dark and would be challenging to classify. So, the first step is to improve the result through preprocess the data, an attempt to amplify the important features in the image.

![alt text][image5]

I experimented with a number of ways to process and normalize the data and obtained the best experimental performance with histogram equalization and normalizing the image values from -0.5 to 0.5. 

By applying histogram equalization to the images, handled by OpenCV's equalizeHist method, we can correct very dark and very bright lighting conditions. With just the preprocessing step, the model was able to achieve ~92% accuracy.

![alt text][image6]

###  Better Model Architecture:

To achieve better performance, we needed to build a better CNN model. After some research and experimentation, we decided to build the model having feed-forward model like LeNet-5 in the convolved, pooled layers are branched off and fed into a flattened, fully connected layer, noting that each of these convolutions passed first through an additional max-pooling proportional to their layer size. In order to prevent overfitting to the training data,we applied a 50% dropout to each of the fully connected layers.

![alt text][image7]

### Training Model:

We initiated a simple learning rate of 0.001 with 100 epochs, using a batch size of 256. The weights were initialized with TensorFlow's truncated_normal method, having a mean of 0 and a standard deviation of 0.1. The loss was calculated using a softmax cross entropy function, by comparing the predicted classes with the validation set. This is optimized using tf.train.AdamOptimizer. This enables the model to use a large step size and move quickly to convergence without a lot of tuning.

The accuracy obtained is :
 validation accuracy = 0.982
 test accuracy: 0.962
 
### Testing model on new images:

Using Google image search, we selected  a few good representations of traffic signs to run through my model. It was more challenging to find interesting test cases. Every image had good lighting conditions. Here we cropped and resized to 32x32 px, a best effect to suit to to train and test in new images.

![alt text][image8]

### Softmax Probabilities:

For better understand whether the model is predicting and to find the wrong gettings the output the top 5 predictions are derived.

The model predicted nearly 100% certainty for every single sign, showing how clear the images are.


Firstly the 100 km/h Speed Limit sign seems to have has sharp contrast and very simple shapes without almost no noise, and the number 100 couldn't be clearer. The model guesses it is a 80 km/h Speed Limit sigh. 

While the model was still 99.99% certain of its guess, this was the least confident of any of the guesses. The next most likely guess at 0.04% likelihood is a speed limit of 50kmph, and behind that, at only 0.0001% likelihood is the correct prediction that is 100kmph.

This is because When we look into the data, we notice that the 50 km/h Speed Limit sign has 2010 examples in the training set, which  is the most represented class in the entire dataset. The 100 km/h Speed Limit sign only has 1,290 examples. So, a possible improvement to prediction accuracy could be by ensuring each class is equally represented.

In conclusion, the model is pretty robust. To get the accuracy above 99%, it is necessary to augment the dataset, by ensuring all the classes are represented equally.

![alt text][image9]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


