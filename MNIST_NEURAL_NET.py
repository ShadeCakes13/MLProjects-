# Import relevant stuff 
# Import the MNIST data and relevant packages. 
import os
import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='datasets/')

# Import labels for MNIST data (70000 label (28,28) pixel images) 
y = mnist.target

# Import corresponding image information for MNIST data 
x = mnist.data/255.0

# This programme will generate and train a fully connected neural net, with dimensions specified by "layers" (since we're dealing with MNIST data, the first entry in layers will be 784, the last will be 10). Below, we have one hidden layer of size 100. "n" is the number of images used in our training data. 

layers = [784, 100, 10]
m = len(layers) 
n = 1000

#Initialise Weights: 

W = ['x']*(m-2)
for i in range(0, m-2): 
	np.random.seed(0)
	W[i] = 0.001*np.random.rand(layers[i],layers[i+1])

np.random.seed(0)
V = 0.01*np.random.rand(layers[m-2],layers[m-1])


# Generate training data:

sample = np.random.RandomState(seed=0).permutation(60000)[0:n]
X = x[sample,:]
Y = y[sample]

# Put Y into correct form:

Z = np.zeros((n,10))
for i in range(0,n):
	j = int(Y[i])
	Z[i][j] = 1 
Y = Z

# Define activation function:
 
def t(x):
	return np.tanh(x)
t = np.vectorize(t)

# Vectorise exponential (to obtain non-negative numbers in final layer): 
 
def s(x):
	return np.exp(x)
s = np.vectorize(s)


#Feed forward. Inputs: "W" is a list of matrices, each matrix representing the weights between layers; "V" is a matrix representing the weights between the pen- and ultimate layer; X represents the training data. Outputs: "H" is a list of matrices, each matrix representing the value at a layer when a particular instance of the training data is inputted; P is a matrix representing the digit probabilities associated with each instance of the training data; L represents the loss function functions, the total distance between our training data probabilities and our labels. 


def ff(W,V,X):
	H = ['x']*(m-1)
	H[0] = X
	for i in range(1,m-1):
		H[i] = t(np.matmul(H[i-1],W[i-1]))
	G = s(np.matmul(H[m-2],V))	
	sums = np.array([1./np.sum(G[i,:]) for i in range(0,n)])
	D = np.diag(sums)
	P = np.matmul(D,G)
	L = np.linalg.norm(P-Y, ord ='fro')**2
	return H, P, L


#Back Propagation. Inputs (as above). Outputs: "LW" is a list of matrices representing the derivative of L with respect to each of the weight matrices; "LV" is a matrix representing the derivative of L with respect to the weight matrix, V. 

def bp(W,V,H,P):
	LP = 2*(P-Y)
	LW = ['x']*(m-2)
	G = np.multiply(LP,P)
	LH = np.matmul(G,np.transpose(V))
	LV = np.matmul(np.transpose(H[m-2]),G)
	for i in range(1,m-1):		
		G = np.multiply(1-np.multiply(H[m-i-1],H[m-i-1]),LH)
		LH = np.matmul(G,np.transpose(W[m-2-i]))
		LW[m-2-i] = np.matmul(np.transpose(H[m-2-i]),G)
	return LW, LV


# Genearate test data. "p" is the test sample size. 

p=400
test = np.random.RandomState(seed=0).permutation(60000)[60000-p:60000]
XT = x[test,:]
YT = y[test]

# Variation on our feed forward function. When evaluating the effectiveness of our neural network, we our just interest in the final calculating the "P" in the function "ff". 

def gff(W,V,X):
	n = np.shape(X)[0]
	H = ['x']*(m-1)
	H[0] = X
	for i in range(1,m-1):
		H[i] = t(np.matmul(H[i-1],W[i-1]))
	G = s(np.matmul(H[m-2],V))	
	sums = np.array([1./np.sum(G[i,:]) for i in range(0,n)])
	D = np.diag(sums)
	P = np.matmul(D,G)
	return P

# Here we calculate the error rate (%) of our neural net on the training data:

def rate(W,V,XT):
	P = gff(W,V,XT)
	p = np.shape(XT)[0]
	count = 0 
	for i in range(0,p):
		a = np.argmax(P[i,:])
		b = YT[i]
		if a != b:	
			count += 1 
	return 100.0*count/p
	


# We run the following to see our neural network as it's being trained. At each iteration, we print the iteration number as well the loss per training image (so that we can compare this ratio to that of an untrained neural net, which we'd expect would be ~0.9). Every tenth iteration, we print the percentage error of our current model on the test data. 
 
count = 0 
while 0 < 1:
	H = ff(W,V,X)[0]
	P = ff(W,V,X)[1]
	L1 = ff(W,V,X)[2]
	L2 = L1 +1 
	print("Loss/n = " + str(L1/n) +"     " + str(count))
	LW = bp(W,V,H,P)[0]
	LV = bp(W,V,H,P)[1]
	thing = 1
	while L2>L1:
		WW = [W[i] - (0.1**thing)*LW[i] for i in range(0,m-2)]
		VV = V - (0.1**thing)*LV
		L2 = ff(WW,VV,X)[2]
		thing +=1
	W = WW
	V = VV
	count +=1
	if count%10 ==0:
		print("Error rate on test data = " + str(rate(W,V,XT)) +"%")



# For example, if we take the following, layers = [784, 100, 10], n = 1000 and p = 400 (i.e. running the above), then we find that after 170 iterations, the error rate is 12.5%.  





