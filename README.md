# Object Detection - Building your custom object detection model

We can perform multiple tasks on images using Convolutiuonal Neural Networks (CNNs). These tasks are arranged in the following hierarchy - 
1. Image Classification - Based on the pixel values, predict the class of the image. This is a supervised learning technique.
2. Localization - When there is only 1 object of a particular class in the image, we perform localization. Localization is the activity of classification along with predicting the bounding box of the predicted object. We estimate 4 co-ordinates of the bounding box.
3. Object Detection - When there are multiple objects to be detected in 1 image, we cannot put a common label on the complete image. We need to label every object in the image. Along with this, we perform localization on each object i.e. we draw a bounding box for each image.
4. Segmentation - Object detection has a disadvantage that it only gives us bounding boxes for the objects. It does not tell us the shape of the objects. Segmentation creates a pixel-wise mask for the detected objects.

In this article, I will discuss Object Detection using Faster-RCNN. 

## Prerequisites :-
1. I was using `Ubuntu 18.04 LTS` and `Python 3.6.8` for this project.
2. Make sure you have downloaded Tensorflow using sudo so that you do not struggle with file permissions. For editting certain files, you may have to use sudo to grant permissions.
3. We will be dealing with the object_detection folder. For me, the path for this folder was `/usr/local/lib/python3.6/dist-packages/tensorflow/models/research/object_detection`. Make sure that the path after tensorflow/ is the same.
4. I am assuming that the objects in each image have been labelled using LabelImg. After labelling, an xml file is generated for each image which contains the co-ordinates of the bounding boxes and label associated with each bounding box. 
5. The file structure is important for this project. Preferably, use the following file structure.
-home/aishwarya/
   -Project1
       -train_images
       -test_images
       -train_labels.csv
       -test_labels.csv
       -xml_to_csv.py
       -generate_tfrecords.py
       -train.record
       -test.record
       -labelmap.pbtxt
The remaining files will be used from tensorflow/models/research/object_detection/.

## Procedure for object detection :-

1. We will convert the annotations in the xml files into csv format by running `xml_to_csv.py`. This will generate 2 files - `train_labels.csv` and `test_labels.csv`.
If you are using this for your own project, make sure to change the file names and their paths in the main function.
```
python xml_to_csv.py
```

2. The next step is to generate TFRecords of these csv files. This will basically map a numerical label to the string object labels. This mapping can be specified in `generate_tfrecords.py`. After running this python code, we will get 2 files - train.record and test.record. To run `generate_tfrecords.py` use the following commands.
```
python generate_tfrecord.py --csv_input=path/train_labels.csv --image_dir=path/train_images --output_path=path/train.record
python generate_tfrecord.py --csv_input=path/test_labels.csv --image_dir=path/test_images --output_path=path/test.record
```
Replace "path" with the path of your csv files, train or test image directories and .record files. 

3. Create a label map for the names of your objects and store this mapping in labelmap.pbtxt. This is a  Tensorflow Graph Text file. The file will contain entries as follows -
```
item {
    id: 1
    name: 'Object name 1'
}item {
    id: 2
    name: 'Object name 2'
}
```
and the rest.

4. Now we will use a configuration file for our training. I have used faster_rcnn_inception_v2_pets.config file which can be found in your `tensorflow/models/research/object_detection/samples/configs` folder. Copy this file from `tensorflow/models/research/object_detection/samples/configs` to `path/` , where "path" is the file which contains your .record files. In reference to point 5 of Prerequisites, the destination folder will be home/aishwarya/Project1. Use sudo cp command.

After this step, in the copied config file, change the following parameters -

	1. Line 9 - Set the number of objects to detect by changing the num_classes variable.
	2. Line 106 - Set the path for the model.ckpt checkpoint file by replacing "PATH_TO_BE_CONFIGURED".
	3. Line 123 - Set the path of the train.record TFRecord file by replacing "PATH_TO_BE_CONFIGURED".
	4. Line 125 - Set the path of label_map.pbtxt Tensorflow Graph Text file by replacing "PATH_TO_BE_CONFIGURED".
	5. Line 130 - Set "num_examples" to the number of images in your test folder.
	6. Line 135 - Set the path of the test.record TFRecord file by replacing "PATH_TO_BE_CONFIGURED".
	7. Line 137 - Again set the path of label_map.pbtxt Tensorflow Graph Text file by replacing "PATH_TO_BE_CONFIGURED".
	
Save these changes.

5. To begin training, go to `tensorflow/models/research/object_detection` folder and run the `model_main.py` file using the command -
```
python model_main.py --logtostderr --model_dir=path/ --pipeline_config_path=path/faster_rcnn_inception_v2_pets.config
```
Replace "path" with your own path.

6. To visualize your training process, navigate to Tensorboard by running-
```
tensorboard --logdir=path
```
Replace "path" with your own path. I used `/home/aishwarya/Project1`. If Tensorboard does not open automatically then you can open `localhost:6006` on your web browser and you will be able to access Tensorboard.

7. To run this model, we will first export the inference graph using `export_inference_graph.py` file present in `tensorflow/models/research/object_detection/` folder. Run it using-
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path path/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix path/model.ckpt-XXXX --output_directory path/inference_graph
```
Again, replace path with your path. XXXX is the latest saved model checkpoint which will be the highest number. Replace XXXX with your latest checkpoint number.  

8. Download `object_detection_tutorial.ipynb` from this repository. This code accepts an image and displays it with the bounding boxes and labels for a limited time span. You may uncomment parts of the code to use your webcam for object detection. Make the following changes for object detection on a single image -

	1. The variable "MODEL_NAME"
	`MODEL_NAME = 'path/inference_graph'`	
	2. The variable "PATH_TO_LABELS"
	`PATH_TO_LABELS = 'path/labelmap.pbtxt'`
	Replace "path" with your own path.
	Now, move to the last cell of the notebook. There, change -
	3. The variable "image"
	`image = cv2.imread('path/image_name.PNG')`
	Again, replace "path" with the path of the image directory on which you want to test this model. Replace "image_name.PNG" with the name of your image.
	
Viola! Now you finally have your model ready.


Reference article for generating your own custom object detector :- https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85
