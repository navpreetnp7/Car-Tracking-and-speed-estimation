# Vehicle detection and tracking module

Vehicle detection and tracking module is implemented using Pretrained YOLOv4 on COCO dataset for object detection, DeepSort Model for object tracking, and TensorFlow library. We take the output of YOLO (You Only Look Once) and feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in to create a highly accurate object tracker and then estimate the speed (in px/sec) using the Euclidean distances between the bounding boxes of each car.

## Demo of Vehicle Tracking with Speed Estimation

![demo](outputs/demo.gif)



## Running the Module

Install all the dependencies and create conda environment to run the model.

```bash
conda env create -f conda-gpu.yml
conda activate yolov4-g
```

### Download the pretrained weights for YOLOv4

We will use the pre-trained weights for our module which can be downloaded from below url.
 https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

These weights are trained using the **COCO** dataset.
Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

### Executing the module

First we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then we run the object_tracker.py script to run our object tracker with YOLOv4, DeepSort and TensorFlow.

The input video is placed in the location *./data/video/* and the output video in *./outputs/*

```bash
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on video
python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4
```


## YOLO (You Only Look Once)

Its a real-time object detection algorithm, which is one of the most effective object detection algorithms. Object detection is a critical capability of autonomous vehicle technology. YOLO achieves high accuracy while also being able to run in real-time. The algorithm “only looks once” at the image in the sense that it requires  only one forward propagation pass through the neural network to make  predictions. After non-max suppression (which makes sure the object  detection algorithm only detects each object once), it then outputs  recognized objects together with the bounding boxes.

With YOLO, a single CNN simultaneously predicts multiple bounding boxes  and class probabilities for those boxes. YOLO trains on full images and  directly optimizes detection performance.



## Deep SORT

Deep SORT is an extension to SORT (Simple Real time Tracker). It is the most popular and one of the most widely used elegant object tracking framework. The Concepts used by Deep Sort are :

#### The Kalman filter

Kalman filter is a crucial component in deep SORT. The variables have only absolute position and velocity factors, since we are assuming a simple linear velocity model. The Kalman filter helps us factor in the noise in detection and uses prior state in predicting a  good fit for bounding boxes.

For each detection, we create a “Track”, that has all the necessary  state information. It also has a parameter to track and delete tracks  that had their last successful detection long back, as those objects  would have left the scene.

Also, to eliminate duplicate tracks, there is a minimum number of detections threshold for the first few frames. 

#### Distance metric

We use the squared Mahalanobis distance (effective metric when dealing with distributions)  to incorporate the uncertainties from the Kalman  filter. Thresholding this distance can give us a very good idea on the  actual associations. This metric is more accurate than say, euclidean  distance as we are effectively measuring distance between 2  distributions 

#### Hungarian algorithm

We use the standard Hungarian algorithm, which is very effective and a simple data association problem. We use it using a a single line import on sklearn.

#### Appearance feature vector

The idea to obtain a vector that can describe all the features of a  given image is quite simple. We first build a classifier over our  dataset, train it till it achieves a reasonably good accuracy, and then  strip the final classification layer. Assuming a classical architecture, we will be left with a dense layer producing a single feature vector,  waiting to be classified.



```
A simple distance metric, combined with a powerful deep learning technique is all it takes for deep SORT to be an elegant Object trackers.
```



### References  

https://github.com/hunglc007/tensorflow-yolov4-tflite

https://github.com/nwojke/deep_sort

https://github.com/theAIGuysCode/yolov4-deepsort

https://medium.com/@riteshkanjee/deepsort-deep-learning-applied-to-object-tracking-924f59f99104

https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088



