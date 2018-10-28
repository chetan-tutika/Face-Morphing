import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from numpy.linalg import inv
from PIL import Image
from cpselect import cpselect

def click_correspondences(im1, im2):

	im1_pts, im2_pts = cpselect(im1, im2)
	#im1_pts = np.flip(im1_pts,1)
	#im2_pts = np.flip(im2_pts,1)
	im1_pts = np.clip(np.around(im1_pts),0, img1.shape[:2])
	im2_pts = np.clip(np.around(im2_pts),0, img2.shape[:2])
	return im1_pts,im2_pts





if __name__ == "__main__":

	import time
	start_time = time.time()
	img1 = np.array(Image.open('Face.jpg').convert('RGB'))
	img2 = np.array(Image.open('Damon.jpg').convert('RGB'))

	im1_pts, im2_pts = click_correspondences(img1, img2)
	im1_pts = np.clip(np.around(im1_pts),0, img1.shape[:2])
	im2_pts = np.clip(np.around(im2_pts),0, img2.shape[:2])


	print(len(im1_pts),len(im2_pts))
	np.save('im1_cor',im1_pts)
	np.save('im2_cor',im2_pts)


	
	print(im1_pts, im2_pts)

