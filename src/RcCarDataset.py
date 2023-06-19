"""
Self-driving car image pair Dataset.

@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

from torch.utils import data

import cv2
import numpy as np

# # use skimage if you do not have cv2 installed
# from skimage import io
y_bias = 186
gray_min = np.array([0, 0, 46])
gray_max = np.array([180, 43, 255])


def augment(dataroot, imgName, angle):
    """Data augmentation."""
    name = dataroot  + imgName
    
    current_image = cv2.imread(name)
    #srcimg2 = current_image[134:200,0:200]
    hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    #mask_gray = cv2.inRange(hsv, gray_min, gray_max)
    #retval, dst = cv2.threshold(mask_gray, 30, 255, cv2.THRESH_BINARY)
    # 膨胀，白区域变大
    #dst = cv2.dilate(dst, None, iterations=2)
    # # 腐蚀，白区域变小
    #dst = cv2.erode(dst, None, iterations=6)
    #矩阵切片，把需要的东西提取出来
    #hawk = dst[134:200,0:200]  #200*66
    #current_image = io.imread(name)
    '''
    img2 = np.zeros_like(srcimg2)
    img2[:,:,0] = hawk
    img2[:,:,1] = hawk
    img2[:,:,2] = hawk
    '''
    #current_image = current_image[65:-25, :, :]
    #if np.random.rand() < 0.5:
        #current_image = cv2.flip(current_image, 1)
        # current_image = np.flipud(current_image)
        #angle = angle * -1.0

    return gray, angle


class TripletDataset(data.Dataset):
    """Image pair dataset."""

    def __init__(self, dataroot, samples, transform=None):
        """Initialization."""
        self.samples = samples
        self.dataroot = dataroot
        self.transform = transform

    def __getitem__(self, index):
        """Get image."""
        batch_samples  = self.samples[index]
        steering_angle = float(batch_samples[1])

        center_img, steering_angle_center = augment(self.dataroot, batch_samples[2], steering_angle)
        #left_img, steering_angle_left     = augment(self.dataroot, batch_samples[1], steering_angle + 0.4)
        #right_img, steering_angle_right   = augment(self.dataroot, batch_samples[2], steering_angle - 0.4)

        center_img = self.transform(center_img)
        #left_img   = self.transform(left_img)
        #right_img  = self.transform(right_img)
        #print(batch_samples[2], steering_angle)
        return (center_img, steering_angle_center) #, (left_img, steering_angle_left), (right_img, steering_angle_right)

    def __len__(self):
        """Length of dataset."""
        return len(self.samples)
