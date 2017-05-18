import pylab as pl
import matplotlib.cm as cm
import numpy as np
import cv2
import glob

# for image_path in all_files:
# data_image_files = set(glob.glob('../Dropbox (MIT)/LearningData/data/cam*/*/*.tif'))
# ground_truth_image_files = set(glob.glob('../Dropbox (MIT)/LearningData/ground_truth/*/*.tif'))
# all_files = data_image_files | ground_truth_image_files

image_path = 'LearningData/data/cam8/1/223.tif'
# 0 is grayscale
im = cv2.imread(image_path, 0)
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
# crops a 64 x 64 px image from top left corner
im_crop = img[0:32, 0:32]
new_image_path = image_path.replace("LearningData", "Learning_Data_Scaled")
im_array = np.array(im_crop)
cv2.imwrite(new_image_path,im_crop)