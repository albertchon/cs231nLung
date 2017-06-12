import numpy as np # linear algebra
import os
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np
import SimpleITK as sitk

output = np.load("masks/1.3.6.1.4.1.14519.5.2.1.6279.6001.262736997975960398949912434623_nodule_mask.npy")
print(output.shape)
print(np.sum(output))
print(np.sum(output > 0.001))
count = 0
for i in range(len(output)):
	if np.sum(output[i]) > 0.001:
		count += 1
print(count)
#plt.imshow(output[160], cmap="gray")
#plt.show()