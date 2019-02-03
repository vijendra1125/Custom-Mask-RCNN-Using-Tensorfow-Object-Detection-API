# Custom Mask RCNN using Tensorfow Object detection API
A sample project to build a custom Mask RCNN model using Tensorflow object detection API

## Folder Structure
- Tensorflow_API-Custom_Mask_RCNN
  - pre_trained_models
    - *downloaded files for the choosen pre-trained model will come here* 
  - dataset
    - Annotations
        - *maskss for training images will come here*
    - JPEGImages
      - *all of images for training will come here*
    - testImages
      - *all images for testing will come here*
    - lable.pbtxt
    - train.record
   - IG
     - *inference graph of the trained model will be saved here*
   - CP
     - *checkpoints of the trained model will be saved here*
   - eval.ipynb
   - train.ipynb
   - *config file for the choosen model*


## Steps

#### Create folders
Create the folders following the structure given above (You could use a different name for any of the folders)

#### Prepare train and test images
This repository contains train and test images for detection of "UE Roll" blue bluetooth speaker and a cup but I will highly recommend you to create your own dataset. Pick up objects you want to detect and take some pics of it with varying backgrounds, angles and distances.  Some of the sample images used in this sample project are given below:

![train_images1](https://user-images.githubusercontent.com/5885636/47269602-faca7280-d57d-11e8-9e99-5fcbb3e8a633.jpg)
![train_images2 jpg](https://user-images.githubusercontent.com/5885636/47269615-264d5d00-d57e-11e8-953d-9820da967dca.jpg)

Once you have captured images, transfer it to your PC and resize it to a smaller size (given images have the size of 512 x 384) so that your training will go smoothly without running out of memory. Now rename (for better referencing later) and divide your captured images into two chunks, one chunk for training(80%) and other for testing(20%). Finally, move training images into *JPEGImages* folder and testing images into *testImages* folder.


#### Label the data
Now its time to label the training data. We will be doing it using the [Pixel Annotation Tool](https://github.com/abreheret/PixelAnnotationTool). 
You could find my tutorial on pixel annotaion tool [here](https://www.youtube.com/watch?v=tX-xcg5wY4U). 
This tool will generate three files in the image folder
  - IMAGENAME_color_mask.png
  - IMAGENAME_mask.png
  - IMAGENAME_watershed_mask.png. 

You need to take all IMAGENAME_color_mask.png and place it in the dataset/Annotations folder and then rename it from IMAGENAME_color_mask.png to IMAGENAME.png

Color mask will look like this:

![image17](https://user-images.githubusercontent.com/5885636/47310389-96c6ad80-d654-11e8-9516-054566d947d9.png)

#### Setup Tensorflow models repository 
Now it's time when we will start using Tensorflow object detection API so go ahead and clone it using the following command
```
git clone https://github.com/tensorflow/models.git
```
Once you have cloned this repository, change your present working directory to models/reserarch/ and add it to your python path. If you want to add it permanently then you will have to make the change in your .bashrc file or you could add it temporarily for current session using the following command:
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
You also need to run following  command in order to get rid of the *string_int_label_map_pb2* issue (more details [HERE](https://github.com/tensorflow/models/issues/1595))
```
protoc object_detection/protos/*.proto --python_out=.
```
Now your Environment is all set to use Tensorlow object detection API


#### Convert the data to Tensorflow record format
In order to use Tensorflow API, you need to feed data in Tensorflow record format. I have modified the script *create_pet_tf_record.py* given by Tensorflow and I have placed the same in this repository inside the folder named as *extra*. Name of the modified file is given as *create_mask_rcnn_tf_record.py*. All you need to do is to take this script and place it in the models/research/object_detection/dataset_tools.

_Note: create_mask_rcnn_tf_record.py is modified in such a way that given a mask image, it should found bounding box around objects on it owns and hence you don't need to spend extra time annotating bounding boxes but it comes at a cost, if mask image has multiple objects of same class then it will not be able to find bounding box for each object of the same class rather it will take a bounding box encompassing all objects of that class. If you have multiple objects of the same class in some images then use some tool like labelImg to generate xml files with bounding boxes and then modify create_mask_rcnn_tf_record.py to take bounding box from xml file instead of trying to find it from mask image._

One additional thing you need to do it to edit the dictionary in the script at line 57. You need to the give name of the classes as key and the value of pixel for the colour of mask you have chosen for respective class while masking the classobjects using pisxelAnnotationTool as value.
After doing above, one last thing is still remaining before we get our Tensorflow record file. You need to create  a file for label map, in this repo its *label.pbtxt*, with the dictionary of the label and the id of objects. Check *label.pbtxt* given in the repository to understand the format, its pretty simple (Note: name of the label should be same as class names you had given in the dictionary). Now it time to create record file. From models/research as present working directory run the following command to create Tensorflow record:
```
python object_detection/dataset_tools/create_mask_rcnn_tf_record.py --data_dir=<path_to_your_dataset_directory> --annotations_dir=<name_of_annotations_directory> --image_dir=<name_of_image_directory> --output_dir=<path_where_you_want_record_file_to_be_saved> --label_map_path=<path_of_label_map_file>
```
For more help run the following command:
```
python object_detection/dataset_tools/create_pascal_tf_record.py -h
```
An example will be:
```
Python object_detection/dataset_tools/create_mask_rcnn_tf_record.py --data_dir=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/dataset --annotations_dir=Annotations --image_dir=JPEGImages --output_dir=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/dataset/train.record --label_map_path=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/dataset/label.pbtxt
```


#### Training
Now that we have data in the right format to feed, we could go ahead with training our model. The first thing you need to do is to select the pre-trained model you would like to use. You could check and download a pret-rained model from [Tensorflow detection model zoo Github page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Once downloaded, extract all file to the folder you had created for saving the pre-trained model files. Next you need to copy *models/research/object_detection/sample/configs/<your_model_name.config>* and paste it in the project repo. You need to configure 5 paths in this file. Just open this file and search for PATH_TO_BE_CONFIGURED and replace it with the required path. I used pre-trained mask RCNN which is trained with inception V2 as feature extracter and I have added modified config file (along with PATH_TO_BE_CONFIGURED as comment above lines which has been modified) for same in this repo. You could also play with other hyperparameters if you want. Now you are all set to train your model, just run th following command with models/research as present working directory
```
python object_detection/legacy/train.py --train_dir=<path_to_the folder_for_saving_checkpoints> --pipeline_config_path=<path_to_config_file>
```
An example will be
```
python object_detection/legacy/train.py --train_dir=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/CP --pipeline_config_path=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/mask_rcnn_inception_v2_coco.config
```
Let it train till loss will be below 0.2 or even lesser. once you see that loss is as low as you want then give keyboard interrupt. Checkpoints will be saved in CP folder. Now its time to generate inference graph from saved checkpoints
```
python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=<path_to_config_file> --trained_checkpoint_prefix=<path to saved checkpoint> --output_directory=<path_to_the_folder_for_saving_inference_graph>
```
An example will be
```
python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/mask_rcnn_inception_v2_coco.config --trained_checkpoint_prefix=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/CP/model.ckpt-2000 --output_directory=/Users/vijendra1125/Documents/tensorflow/object_detection/multi_object_mask/IG
```
**Bonus: If you want to train your model using Google Colab then check out the *train.ipynb* file**

#### Test the trained model
Finally, it's time to check the result of all the hard work you did. All you need to do is to copy model/research/object_detection/object_detection_tutorial.ipynb and modify it to work with you inference graph. A modified file is already given as eval.ipynb with this repo, you just need to change the path, number of classes and the number of images you have given as test image. Below is the result of the model trained for detecting the "UE Roll" blue bluetooth speaker and a cup.

![multi_mask2](https://user-images.githubusercontent.com/5885636/47269891-8a255500-d581-11e8-8dc7-26bf7fc3a013.png)
![multi_mask1](https://user-images.githubusercontent.com/5885636/47269890-8a255500-d581-11e8-9e81-f7eba4505961.png)
![multi_mask3](https://user-images.githubusercontent.com/5885636/47269892-8abdeb80-d581-11e8-93e4-01e6dd61ea3f.png)
![multi_mask5](https://user-images.githubusercontent.com/5885636/47269894-8abdeb80-d581-11e8-87b0-cf7ed32d0faa.png)
![multi_mask6](https://user-images.githubusercontent.com/5885636/47269895-8abdeb80-d581-11e8-923f-f8f567d0710c.png)
