## Data
The data is obtained from the AIcrowd Road Segmentation Challenge. It consists of 100 training images with their groudntruths and 50 test images without groundgruths. 

## Data Preprocessing
All data preprocessing and the augmentations done to enlarge our dataset can be found data.py. 

## Helper Files
In order to make submissions we use the following .py files:
- mask_to_submission.py
- submission_to_mask.py
These files can be found in the submission folder.

## Models
### 1) CNN 
The CNN model's architecture can be found in CNN_model.py
### 2) U-Net
The U-Net model's architecture can be found in Unet_model.py
