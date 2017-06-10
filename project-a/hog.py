import numpy as np
from skimage.feature import hog
from matplotlib import pyplot as plt
import os

def get_hog():
	print('getting hog')	
	INPUT_FOLDER = './stage2-pixels-npy/'
	patients = os.listdir(INPUT_FOLDER)
	path = './stage2-hog-npy/'
	for p in range(len(patients)):
		image = np.load(INPUT_FOLDER + patients[p])    	
		image = np.transpose(image, [2, 0, 1])
		image_hog = np.zeros((64, 128, 128), dtype=np.float32)
		for i in range(32, 64):
			layer = image[i]
			plt.imshow(np.float32(layer), cmap='gray')
			plt.show()
			_, layer_hog = hog(layer, visualise = True)
			plt.imshow(layer_hog, cmap='gray')
			plt.show()
			image_hog[i] = layer_hog
			break
		break
		image_hog = np.float16(np.transpose(image_hog, [1, 2, 0]))
		np.save(path + patients[p], image_hog)
		print p+1

if __name__ == '__main__':
	get_hog()