## Object Detection with YOLOv8
A YOLO object detection model, specifically YOLOv8 with an FPN architecture set to 2, was employed for the task, utilizing pretrained weights from the COCO dataset.
This was achieved through the application of transfer learning techniques using the TensorFlow and keras_cv libraries.<br>
The dataset, comprising images and their corresponding labels in text files, underwent preprocessing to resize images to 640x640 pixels with a scaling factor between 0.8 and 1.25. The data pipeline was constructed to format the input for the model as a combination of images, class labels, and bounding boxes.<br>
Model training was conducted on a university hub node equipped with an Nvidia GTX 1080 GPU, after verifying the implementation's functionality on a smaller batch locally.<br>
#### Methodology 
* The foundational architecture for this task was YOLOv8, chosen for its robustness and efficiency in processing complex image data for object detection tasks.<br>
* The optimization of the model was conducted using the Adam optimizer, configured with a learning rate of 0.007 and a weight decay parameter of 0.00009. This choice was done because of the optimizer's capability to dynamically adjust learning rates for each parameter, enhancing the model's convergence speed and performance stability.<br>
* I did run the training using different l_r such as 0.001, 0.0009, 0.005 ect. However in my opinion I did not observe any significant difference by changing the learning rate.
For the loss functions, I used binary cross-entropy for classification tasks, for the task of identifying if there is an object in the box or not, and CIoU (Complete Intersection over Union) for bounding box regression.<br>
* In my experiments, I tried using SparseCategorical Cross entropy loss because of a multi class classification task. However, after running a few iterations and doing some research, I realized that was not the correct choice.<br>
* The model training was conducted with a batch size of 32, a decision influenced by the need to balance computational efficiency and the model's ability to generalize from the training data. Increasing the batch size was cursing the memory overflow errors and reducing it led to showing down the training process.<br>
* The comprehensive architecture of the YOLOv8 model under this configuration encompassed a total of 185 layers.
My experimental strategy included two distinct training regimes: first, freezing the top layer and selectively training the last 20-25 layers, which collectively comprise approximately 13,000 parameters; second, training the entire network, encompassing around 3 million parameters. The latter approach was found to yield superior results in terms of model accuracy and object detection performance. Consequently, I adopted this full-network training strategy for the final model implementation, leveraging the full depth and complexity of the YOLOv8 architecture to achieve optimal detection outcomes.

##### Instructions for running the code.
```
>>> Python3 object_detection.py
```
I have included simple instructions using command line input() from the user. During the execution you will be asked following questions:<br>
* If you want to train the model or just make inference using already trained model?[y/n]
* Do you want to load the existing weights from previous training ?[y/n]
* Do you want to save the current model weights?[y/n]
* Do you want to make predictions and generate results after training?[y/n]<br>

References:<br>
https://keras.io/examples/vision/yolov8/
