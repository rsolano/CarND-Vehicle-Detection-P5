**Vehicle Detection Project**

Ricardo Solano

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog_car.png
[image2]: ./output_images/hog_notcar.png
[image3]: ./output_images/test_bboxes_out1.png
[image4]: ./output_images/test_bboxes_out2.png
[image5]: ./output_images/test_bboxes_out3.png
[image6]: ./output_images/test_bboxes_out4.png
[image7]: ./output_images/test_bboxes_out5.png
[image8]: ./output_images/test_bboxes_out6.png
[image9]: ./output_images/test_out1.png
[image10]: ./output_images/test_out2.png
[image11]: ./output_images/test_out3.png
[image12]: ./output_images/test_out4.png
[image13]: ./output_images/test_out5.png
[image14]: ./output_images/test_out6.png
[image15]: ./output_images/test_heatmap_out1.png
[image16]: ./output_images/test_heatmap_out2.png
[image17]: ./output_images/test_heatmap_out3.png
[image18]: ./output_images/test_heatmap_out4.png
[image19]: ./output_images/test_heatmap_out5.png
[image20]: ./output_images/test_heatmap_out6.png
[image21]: ./output_images/test_labels_out1.png
[image22]: ./output_images/test_labels_out2.png
[image23]: ./output_images/test_labels_out3.png
[image24]: ./output_images/test_labels_out4.png
[image25]: ./output_images/test_labels_out5.png
[image26]: ./output_images/test_labels_out6.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting the HOG features is contained in the `get_hog_features` function, lines 8 through 25 of the file called `lesson_functions.py`. The code leverages the `skimage.hog()` function for computing the gradient histogram. Below is an example for a car and non-car image with their corresponding HOG features extracted:

Car                | Not car
:-----------------:|:-------------------------:
![alt text][image1]| ![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I completed this step mostly by trial and error. I tried several combinations of HOG parameters (color_space, HOG orientations, pixels per cell, cells per block) and chose my final set based on accuracy of the model (see next step) and execution time. The parameters that worked best for me were:

Color space | Orientations | Pixels per cell | Cell per block | HOG channel
:----------:|:------------:|:---------------:|:--------------:|:-----------
YCrCb       |      9       |        8        |        2       |     ALL


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the LinearSVC() function provided by the sklearn library. I used the entire dataset provided for vehicles and non-vehicles for this project (GTI and KITT). The code for this is in the `process_data()` and `train()` functions in `search_classify.py`, lines 81 to 130. My model achieved an accuracy of 98.8%:

```
Extracting training data and training model...
Feature vector length: 8460
27.98 Seconds to train SVC...
Test Accuracy of SVC =  0.9879
labels    : [ 0.  1.  0.  0.  1.  0.  1.  0.  0.  0.]
prediction: [ 0.  1.  0.  0.  1.  0.  1.  0.  0.  0.]
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For this step I used the Hog sub-sampling windows search method, which allows to both extract features and make predictions. I adapted the `find_cars()` function from the class materials and defined it in the `search_classify.py` file, lines 15-78. The idea is to extract the hog features once from a portion of the image and pass the individual subsamples to the classifier. Although this functions allows to be invoked multiple times with varying scale values, I'm making a single call per frame with a 1.5 scale. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched the images using a scale of 1.5 in all channels using YCrCb and spatial binning color and histograms of color. Here are some examples:

![alt text][image5]
![alt text][image6]
![alt text][image8]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Bounding boxes     | Heatmap              |Labels              | Combined boxes
:-----------------:|:--------------------:|:------------------:|:---------------------:
![alt text][image3]| ![alt text][image15] |![alt text][image21]|![alt text][image9]
![alt text][image4]| ![alt text][image16] |![alt text][image22]|![alt text][image10]
![alt text][image5]| ![alt text][image17] |![alt text][image23]|![alt text][image11]
![alt text][image6]| ![alt text][image18] |![alt text][image24]|![alt text][image12]
![alt text][image7]| ![alt text][image19] |![alt text][image25]|![alt text][image13]
![alt text][image8]| ![alt text][image20] |![alt text][image26]|![alt text][image14]


### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image14]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

