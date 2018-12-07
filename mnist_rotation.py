# We wish to explore rotating MNIST images. To a human, number classification is invariant under small rotations in that
# we still call a 6 a six, even if it's written slightly wonkily (but only "slightly" - too much and it becomes a nine). So 
# by rotating our labeled images through small angles, we can obtain new images with correct labels and so greatly expand our 
# labeled data. 

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

#Import MNIST data. 

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='datasets/')
y = pd.Series(mnist.target).astype('int').astype('category')
x = pd.DataFrame(mnist.data)

def image_mat(n):
	return x.loc[n].values.reshape(28,28)

# A function that plots the desired image in our MNIST data. 

def image_plot(n):
	plt.close()
	plt.imshow(image_mat(n), cmap='gray_r')
	plt.show()
    

# Since the images have already been centred, we can rotate about the middle of each image. Rotating these images presents two 
# challenges: the first is that unless we're rotating by mutliples of pi/2, in rotating a square image onto a square image, we'll have
# to "shave off" the corners; the second is that the MNIST images are low resolution (28x28) and again, unless we're rotating by multiples
# of pi/2 it's unclear how to rotate a square pixel onto a square pixel. 

# To the first problem, the fact that our images are already centred means that we can assume that the corners to be shaved off 
# under rotation are blank. This leads us to the following tactic:  generate a blank image  ((28,28) zero matrix); for each pixel in
# the original image, find it's location under a rotation about the centre of the image; if this pixel is rotated onto a point on the
# blank image (i.e. if it isn't part of the shaved off corner), let the nearest pixel to this point on the blank image take the value 
# of the original pixel. In this way we generate our rotated image matrix. We derive the following functions through geometric 
# considerations. 

def rotate_coordinates(i,j,angle,n):
	R = np.reshape([np.cos(angle),-np.sin(angle),np.cos(angle),np.sin(angle)],(2,2))
	a = np.reshape([1,1],(2,1))
	b = np.reshape([j,i],(2,1))
	return np.floor(0.5*n*a-(0.5*n-0.5)*np.matmul(R,a)+np.matmul(R,b)).astype(int) 

def image_rotate(matrix,angle):
	n = np.shape(matrix)[0]
	Z = np.zeros((n,n))
	for i in range(0,n):
		for j in range(0,n):
			v = rotate_coordinates(i,j,angle,n)
			k = v[0][0]
			l = v[1][0]
			if 0<=l<n and 0<=k<n:
				Z[l,k]=matrix[i,j]
	return Z

# To quickly see the rotated image, we call the following function.

def image_rotate_plot(n,angle):
	plt.close()
	plt.imshow(image_rotate(image_mat(n),angle), cmap='gray_r')
	plt.show()
    
# We see that with our rotation function we sometime encounter a problem. For example, try "image_rotate_plot(0,0.2)". There
# are some pixels on the image that are conspicuously blank; what's happened here is that this pixel has never quite been the "nearest"
# to any of the orignal image's pixels i.e. blankness due to rounding. Here encounter the second of our challenges. 

# If you rotate a pixel onto a grid, it may land on several squares, but with our current method only one of the squares takes the 
# pixel's value. An intuitive way to overcome this "all-or-nothing" method would be to include some kind of averaging between nearby 
# pixels. One way of averaging pixels in an image would be through decreasing the resolution of the image. This leads to the following
# method: increase the image's resolution; rotate the image using our rotation function; finally decrease the rotated image's resolution 
# to that of the original image. 

# Here we define a function to take an nxn matrix and return an (n*m)x(n*m) matrix, where each entry in the original matrix becomes
# an mxm block in the corresponding position in the final matrix. This corresponds to increasing the image resolution, increasing the 
# the number of pixels by a factor of m**2. 
    
def smooth(A,m):
	n = np.shape(A)[0]
	B = np.zeros((m*n,m*n))
	for i in range(0,n):
		for j in range(0,n):
			a = float(A[i][j])
			for l in range(0,m):
				for k in range(0,m):
					B[m*i+l][m*j+k] = a
	return B

# Here we take an (n*m)x(n*m) matrix and return an nxn matrix, where each entry in the final matrix takes the mean value of 
# of the mxm block in the corresponding position of the original matrix. 

def coarse(B,m):
    n = int(np.shape(B)[0]/m)
    A = np.zeros((n,n))
    summ = 0
    for i in range(0,n):
        for j in range(0,n):
            for l in range(0,m):
                for k in range(0,m):
                    summ+=B[m*i+l][m*j+k]
            A[i][j] = float(summ)/(m**2)
            summ =0
    return A

# As described, we rotate our image matrix through a specified angle by increasing the resolution (by a factor of m), rotating and
# then decreasing the resolution of image. 

def new_image_rotate(matrix,angle,m):
	return coarse(image_rotate(smooth(matrix,m),angle),m)

# To quickly see the rotated image, we call the following function.

def new_image_rotate_plot(n,angle,m):
	plt.close()
	plt.imshow(new_image_rotate(image_mat(n),angle,m),cmap='gray_r')

# We can quickly call four plots of an image for comparison. From left to right: the original image, the image rotated
# through specified angle without resolution changes, the image rotated through the same angle but with a factor 2 resolution
# change, the image rotated through the same angle but with a factor 3 resolution change. 

def compare_image_plot(n,angle):
	plt.close()
	fig, axes = plt.subplots(1,4,sharey=True)
	a = image_mat(n)
	b = image_rotate(a,angle)
	c = new_image_rotate(a,angle,2)
	d = new_image_rotate(a,angle,3)
	axes[0].imshow(a,cmap='gray_r')
	axes[1].imshow(b,cmap='gray_r')
	axes[2].imshow(c,cmap='gray_r')
	axes[3].imshow(d,cmap='gray_r')
	plt.tight_layout()
	plt.show()

# Playing about, we see that in general there is an improvement in the quality of the rotation from left to right; however the 
# quality increases only marginally as the resolution factor increases, while the computation cost increases at least by the square
# of the resolution factor 

#Example:

compare_image_plot(23456,1)

