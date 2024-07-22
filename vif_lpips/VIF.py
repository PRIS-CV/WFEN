from vif_utils import vif
from PIL import Image
import numpy as np
import os
import sys
import cv2 as cv

HR_path =  "/home/liwenjie/wenjieli/Face_SR/datasets/test_datasets/Helen50/HR/" # CelebA1000  Helen50
SR_path =  "/home/liwenjie/wenjieli/Face_SR/WFEN/results_helen/trans_LAAT/" # results_CelebA  results_helen



fileList = os.listdir(HR_path)
sum = 0
count = 0
for image in fileList:
    count += 1
    img1 = np.array(Image.open(HR_path  + image).convert('L'))
    img2 = np.array(Image.open(SR_path  + image).convert('L'))
    x = vif(img1, img2)
    sum = sum + x
print(sum/count)
print(count)