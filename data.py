from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from sklearn.model_selection import train_test_split 
import matplotlib.image as mpimg
from helpers import * 
from keras.preprocessing.image import img_to_array, array_to_img
from scipy.ndimage import rotate
import random
from PIL import Image, ImageEnhance
from skimage.util import random_noise
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

def augment_dataset(batch_size, aug_dict, images, masks, img_path, msk_path, max_iters, seed = 1):

  img_array = np.array([img_to_array(img) for img in images])
  msk_array = np.array([img_to_array(img) for img in masks])

  image_datagen = ImageDataGenerator(**aug_dict, fill_mode = "reflect")
  mask_datagen = ImageDataGenerator(**aug_dict, fill_mode = "reflect")
  image_datagen.fit(img_array, augment = True, seed = seed)
  mask_datagen.fit(msk_array, augment = True, seed = seed)
  
  image_generator = image_datagen.flow(img_array,
                                      batch_size = batch_size,
                                      save_to_dir = img_path,
                                      save_format = "png",
                                      save_prefix = "aug_img",
                                      seed = seed)

  mask_generator = mask_datagen.flow(msk_array,
                                    batch_size = batch_size,
                                    save_to_dir = msk_path,
                                    save_format = "png",
                                    save_prefix = "aug_msk",
                                    seed = seed)

  train_generator = zip(image_generator, mask_generator)

  i = 0
  for batch in train_generator:
    i += 1
    if i > max_iters: break


def preprocessing(batch_size, train_path, aug_dict, patches = False, random_state = 0, 
split_ratio = 0.2, foreground_threshold = 0.25, max_iters = 20):

  images_path = os.path.join(train_path, 'images/')
  masks_path = os.path.join(train_path, 'groundtruth/')   
  images, masks = load_images(images_path, masks_path)
  #Augment the dataset
  augment_dataset(batch_size, aug_dict, images, masks, images_path, masks_path, max_iters)
  files_img_aug = os.listdir(images_path)
  
  images_aug = np.asarray([load_image(images_path + files_img_aug[i]) for i in range(len(files_img_aug))])
  files_mask_aug = os.listdir(masks_path)
  
  mask_aug = np.asarray([load_image(masks_path + files_mask_aug[i]) for i in range(len(files_mask_aug))])

  X = images_aug
  y = np.expand_dims(mask_aug, axis = 3)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state = random_state)

  return X_train, X_test, y_train, y_test
