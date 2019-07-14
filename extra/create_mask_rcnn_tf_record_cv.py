# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re
import cv2

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Path to root directory to dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('image_dir', 'JPEGImages', 'Name of the directory contatining images')
flags.DEFINE_string('annotations_dir', 'Annotations', 'Name of the directory contatining Annotations')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')
FLAGS = flags.FLAGS

# mask_pixel: dictionary containing class name and value for pixels belog to mask of each class
# change as per your classes and labeling
mask_pixel = {'balloon':[119,76,194,117,84]}

def dict_to_tf_example(filename,
                       mask_path,
                       label_map_dict,
                       img_path):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    filename: name of the image 
    mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      dataset directory holding the actual image data.


  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by filename is not a valid JPEG
  """
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  width = np.asarray(image).shape[1]
  height = np.asarray(image).shape[0]
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  with tf.gfile.GFile(mask_path, 'rb') as fid:
    encoded_mask_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_mask_png)
  mask = PIL.Image.open(encoded_png_io)
  mask_np = np.asarray(mask.convert('L'))
  if mask.format != 'PNG':
    raise ValueError('Mask format not PNG')

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  masks = []

  cv2.imshow("origin", mask_np)
  cv2.imwrite('origin.png', mask_np)

  for k in list(mask_pixel.keys()):
      class_name = k

      pixel_vals = mask_pixel[class_name]

      for pixel_val in pixel_vals:     
        print('for pixel val#:', pixel_val) 
        
        mask_copy = mask_np.copy()
        mask_copy[mask_np == pixel_val] = 255
        ret,thresh = cv2.threshold(mask_copy, 254, 255, cv2.THRESH_BINARY)
        (_, conts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        index = 0
        if conts != None:
          for c in conts:
            #rect = cv2.boundingRect(c)
            x, y, w, h = cv2.boundingRect(c)
            xmin = float(x)
            xmax = float(x+w)
            ymin = float(y)
            ymax = float(y+h)
            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)
            print(filename, 'bounding box for', class_name,  xmin, xmax, ymin, ymax)

            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])

            mask_np_black = mask_np*0
            cv2.drawContours(mask_np_black, [c], -1, (255,255,255), cv2.FILLED)

            mask_remapped = (mask_np_black == 255).astype(np.uint8)
            masks.append(mask_remapped)

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }

  encoded_mask_png_list = []
  for mask in masks:
    img = PIL.Image.fromarray(mask)
    output = io.BytesIO()
    img.save(output, format='PNG')
    encoded_mask_png_list.append(output.getvalue())
  feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
      mask_path = os.path.join(annotations_dir, example + '.png')
      image_path = os.path.join(image_dir, example + '.jpg')

      try:
        tf_example = dict_to_tf_example(example,
                                        mask_path,
                                        label_map_dict,
                                        image_path)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
          print("done")
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', xml_path)

def main(_):
  data_dir = FLAGS.data_dir
  train_output_path = FLAGS.output_dir
  image_dir = os.path.join(data_dir, FLAGS.image_dir)
  annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from dataset.')
  examples_list = os.listdir(image_dir) 
  for el in examples_list:
    if el[-3:] !='jpg':
      del examples_list[examples_list.index(el)]
  for el in examples_list:  
    examples_list[examples_list.index(el)] = el[0:-4]

  create_tf_record(train_output_path,
                  FLAGS.num_shards,
                  label_map_dict,
                  annotations_dir,
                  image_dir,
                  examples_list)


if __name__ == '__main__':
  tf.app.run()
