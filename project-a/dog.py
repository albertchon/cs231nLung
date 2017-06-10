import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os
from skimage.feature import hog
import cv2


INPUT_FOLDER = './npy3d-data/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
patient_images = {}

#storing images
for p in range(len(patients)):
    path = INPUT_FOLDER + patients[p]
    image = np.load(path)
    print image.shape
    ax_slice = np.float32(image[32])
    print ax_slice.shape
    # plt.imshow(ax_slice, cmap='gray')
    # fd, hog_image = hog(ax_slice, pixels_per_cell=(6,6), visualise=True)
    # hog_image /= np.linalg.norm(hog_image)
    # orb = cv2.ORB_create()
    ax_slice = ax_slice.astype(np.uint8, copy=False)
    # kp = orb.detect(ax_slice, None) 
    # kp, des = orb.compute(ax_slice, kp)
    # print kp
    # img3 = np.zeros((128, 128))
    # img2 = cv2.drawKeypoints(ax_slice,kp,outImage = ax_slice, color=(0,255,0), flags=0)
    # plt.imshow(img2)
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(ax_slice)
    im_with_keypoints = cv2.drawKeypoints(ax_slice, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(im_with_keypoints)
    # plt.show()
    # plt.imshow(hog_image, cmap='gray')
    plt.show()

    break