# Import relevant stuff 
# Import the MNIST data and relevant packages.

import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='datasets/')

# Import labels for MNIST data
 
y = mnist.target

# Import corresponding image information for MNIST data (scaled)
x = mnist.data/255.0

# This programme will generate and train fully connected MNIST neural networks with specified shapes. 

# We initialise the neural network.
#Inputs: "layers" is a list defining the shape of the neural networks (since we're dealing with MNIST data, the first entry will be 784 and
# 	the last will be 10, so that "layers = [784,100,100,10]" corresponds to a neural network with two hidden layers of 100 nodes); 
# 	"n" is the number of images to include in our training data; "p" is the number of images to include in our test data; "w" sets 
# 	the maximum value of entries in all but the final of our initial weight matrices; "v" does the same but for our final intial 
# 	weight matrix. 
#Outputs: "W" is a list consisting of the initial weight matrix between all but the final two layers; "V" is the initial weight
#	matrix between the final two layers; "X" is a matrix of training images; "Y" is a matrix of training labels; "XT" is a matrix of 
# 	test images; "YT" is a matrix of test labels; "m" is the number of layers in the neural net. 

def init(layers,n,p,w,v):
	m = len(layers) 
	
	W = ['x']*(m-2)
	for i in range(0, m-2): 
		np.random.seed(1)
		W[i] = w*np.random.rand(layers[i],layers[i+1])

	np.random.seed(1)
	V = v*np.random.rand(layers[m-2],layers[m-1])

	sample = np.random.RandomState(seed=0).permutation(60000)[0:n]
	X = x[sample,:]
	Y = y[sample]

	Z = np.zeros((n,10))
	for i in range(0,n):
		j = int(Y[i])
		Z[i][j] = 1 	
	Y = Z

	test = np.random.RandomState(seed=0).permutation(60000)[60000-p:60000]
	XT = x[test,:]
	YT = y[test]
	
	return W,V,X,Y,XT,YT, m 


# Define activation function.

def t(x):
	return np.tanh(x)
t = np.vectorize(t)

# Vectorise exponential (to obtain non-negative numbers in final layer).

def s(x):
	return np.exp(x)
s = np.vectorize(s)

#Feed forwards. 
#Inputs: as described above. 
#Outputs: "H" is a list of matrices, each matrix representing the value at a particular layer when particular data is inputted 
#	  into our neural network; P is a matrix representing the digit probabilities associated with each instance of our inputted 
# 	  data; L represents the loss that the inputted data incurs in our neural network.

def ff(W,V,X,Y,m):
	n = np.shape(X)[0]
	H = ['x']*(m-1)
	H[0] = X
	for i in range(1,m-1):
		H[i] = t(np.matmul(H[i-1],W[i-1]))
	G = s(np.matmul(H[m-2],V))	
	sums = np.array([1./np.sum(G[i,:]) for i in range(0,n)])
	D = np.diag(sums)
	P = np.matmul(D,G)
	L = np.linalg.norm(P-Y)**2
	return H, P, L

# Back Propagation. 
# Inputs: as above. 
# Outputs: "LW" is a list of matrices representing the derivative of L with respect to each of the weight matrices; "LV" is a matrix 		  representing the derivative of L with respect to the weight matrix "V". 


def bp(W,V,Y,H,P,m):
	LW = ['x']*(m-2)
	LP = 2*(P-Y)
	G = np.multiply(LP,P)
	HH = H[m-2]
	
	LV = np.matmul(np.transpose(HH),G)
	I = np.ones((10,10))
	K = np.matmul(G,I)
	K = np.multiply(K,P)
	LV -= np.matmul(np.transpose(HH),K)
	
	LH = np.matmul(G,np.transpose(V))
	Q = np.matmul(P,np.transpose(V))
	I = np.ones((10,np.shape(HH)[1]))
	K = np.matmul(G,I)
	LH -= np.multiply(K,Q)

	for i in range(1,m-1):		
		G = np.multiply(1-np.multiply(H[m-i-1],H[m-i-1]),LH)
		LH = np.matmul(G,np.transpose(W[m-2-i]))
		LW[m-2-i] = np.matmul(np.transpose(H[m-2-i]),G)
	return LW, LV

# Variation on our feed forwards function: when evaluating the effectiveness of our neural network, we're just interest in 
# the final calculation of "P" in the function "ff". (Inputs as above)

def gff(W,V,X,m):
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

# Calculate the error rate (%) of our neural network on the training data. (Inputs as above)

def rate(W,V,XT,YT,m):
	P = gff(W,V,XT,m)
	p = np.shape(XT)[0]
	count = 0 
	for i in range(0,p):
		a = np.argmax(P[i,:])
		b = YT[i]
		if a != b:	
			count += 1 
	return 100.0*count/p

# We run the following to see our neural network being trained in real time. At each iteration, we print the iteration number 
# and the loss per training image (so that we can compare this ratio to that of an untrained neural network, for which we'd 
# expect this value to be ~0.9). Every tenth iteration, we print the percentage error of our current model on the test data. 
# We output the best performing weights when we encounter some degree of convergence (either no appreciable change in the loss 
# or the percentage error is persistently not decreasing). 

def train1(layers,n,p,w,v):
	
	st = init(layers,n,p,w,v)
	W = st[0]
	V = st[1]
	X = st[2]
	Y = st[3]
	XT = st[4]
	YT = st[5]
	m = st[6]	
	 
	count = 0 
	thang = 0 
	R2 =  1000 
	fail1 = 0 
	fail2 = 0

	P = gff(W,V,X,m)
	L1 = np.linalg.norm(P-Y)**2
	L2 = L1 +1 
	R1 = R2 
	thing = 1

	while fail1< 200 and fail2 == 0:
		print("Loss/n = " + str(L1/n) +"     " + str(count))
		H = ff(W,V,X,Y,m)[0]
		LW = bp(W,V,Y,H,P,m)[0]
		LV = bp(W,V,Y,H,P,m)[1]
		if count%10 ==0:
			t1 = t2 
			t2 = time.time()
			print(t2-t1)
			R2 = rate(W,V,XT,YT,m)
			if R2 < R1: 
				output_W = W
				output_V = V
			else:
				fail1 += 1
			print("Error rate on test data = " + str(R2) +"%")
		while L2>=L1:
			WW = [W[i] - (0.1**(thing-thang))*LW[i] for i in range(0,m-2)]
			VV = V - (0.1**(thing-thang))*LV
			P = gff(WW,VV,X,m)
			L2 = np.linalg.norm(P-Y)**2
			thing +=1
			if thing == 200:
				fail2 = 1				
				break
		if thing == 2:
			thang +=1
		else:
			thang -=1 
		W = WW
		V = VV
		L1 = L2 
		L2 = L1 + 1 
		R1 = R2 
		thing = 1
		count +=1

	return output_W, output_V

# Though it has the advantage of making relatively the iterative process going on in our training, calling lots of external 
# functions in "train1" leads to many redundant and expensive calculations. As such, we create the less clear, but more 
# computationally efficient, "train2".


def train2(layers, n, p, w, v): 
	
	m = len(layers)
	
	W = ['x']*(m-2)
	for i in range(0, m-2): 
		np.random.seed(1)
		W[i] = w*np.random.rand(layers[i],layers[i+1])
	np.random.seed(1)
	V = v*np.random.rand(layers[m-2],layers[m-1])

	s = np.vectorize(np.exp)
	t = np.vectorize(np.tanh)

	sample = np.random.RandomState(seed=0).permutation(60000)[0:n]
	X = x[sample,:]
	Y = y[sample]
	Z = np.zeros((n,10))
	for i in range(0,n):
		j = int(Y[i])
		Z[i][j] = 1 
	Y = Z
	
	test = np.random.RandomState(seed=0).permutation(60000)[60000-p:60000]
	XT = x[test,:]
	YT = y[test]

	count = 0 
	thang = 0 
	R2 =  1000 
	fail1 = 0 
	fail2 = 0

	H = ['x']*(m-1)
	HT = ['x']*(m-1)
	H[0] = X
	LW = ['x']*(m-2)
	I1 = np.ones((10,10))
	I2 = np.ones((10,layers[m-2]))
	
	for i in range(1,m-1):
		H[i] = t(np.matmul(H[i-1],W[i-1]))
	G = s(np.matmul(H[m-2],V))	
	sums = np.array([1./np.sum(G[i,:]) for i in range(0,n)])
	P = np.matmul(np.diag(sums),G)
	L1 = np.linalg.norm(P-Y)**2
	L2 = L1 +1 
	R1 = R2 
	thing = 1

	while fail1 < 400 and fail2 == 0:
		
		print("Loss/n = " + str(L1/n) +"     " + str(count))

		HH = H[m-2]
		LV = np.matmul(np.transpose(HH),np.multiply(2*(P-Y),P))-np.matmul(np.transpose(HH),np.multiply(np.matmul(np.multiply(2*(P-Y),P),I1),P))
		LH = np.matmul(np.multiply(2*(P-Y),P),np.transpose(V))-np.multiply(np.matmul(np.multiply(2*(P-Y),P),I2),np.matmul(P,np.transpose(V)))
		
		for i in range(1,m-1):		
			G = np.multiply(1-np.multiply(H[m-i-1],H[m-i-1]),LH)
			LH = np.matmul(G,np.transpose(W[m-2-i]))
			LW[m-2-i] = np.matmul(np.transpose(H[m-2-i]),G)

		if count%10 ==0:		
			HT = XT
			for i in range(1,m-1):
				HT = t(np.matmul(HT,W[i-1]))
			G = s(np.matmul(HT,V))	
			sums = np.array([1./np.sum(G[i,:]) for i in range(0,p)])
			D = np.diag(sums)
			P = np.matmul(D,G)
			Count = 0 
			for i in range(0,p):
				a = np.argmax(P[i,:])
				b = YT[i]
				if a != b:	
					Count += 1 
			R2 = 100.0*Count/p
			if R2 < R1: 
				output_W = W
				output_V = V
			else:
				fail1 += 1
			print("Error rate on test data = " + str(R2) +"%")
		while L2>=L1:
			WW = [W[i] - (0.1**(thing-thang))*LW[i] for i in range(0,m-2)]
			VV = V - (0.1**(thing-thang))*LV
			for i in range(1,m-1):
				H[i] = t(np.matmul(H[i-1],WW[i-1]))
			G = s(np.matmul(H[m-2],VV))	
			sums = np.array([1./np.sum(G[i,:]) for i in range(0,n)])
			P = np.matmul(np.diag(sums),G)
			L2 = np.linalg.norm(P-Y)**2
			thing +=1
			if thing == 200:
				fail2 = 1				
				break 
		if thing == 2:
			thang +=1
		else:
			thang -=1 
		W = WW
		V = VV
		for i in range(1,m-1):
			H[i] = t(np.matmul(H[i-1],W[i-1]))
		L1 = L2 
		L2 = L1 + 1 
		R1 = R2 
		thing = 1
		count +=1
	
	return output_W, output_V

# For example, if we run the "train2([784, 100, 10],20000,500,0.0001,0.0000001)", after ~8hrs the function terminates yields:  
#"Error rate on test data = 3.4%
#Loss/n = 0.018052078097475083     2871"

# A note on use:
# Since we're dealing with large matrices and exponential functions, the programme has a habit of breaking down in early iterations
# due to overflow. It's for this reason that we've introduced the coefficients "w" and "v" to control the size of the values in the
# initial matrices and so prevent overflow; as the exponential function is only used in the final layer, the value of "v" is 
# especially important in this regard. Of course, the values of these coefficients will also affect the behaviour of the algorithm 
# more generally, and suitable values for fast convergence are found by trial and error and depend on the shape of the neural 
# network and the number of images in the training data. 
#	It's also worth explaining the convergence criteria of the algorthm. The algorithm can terminate for one of two reasons.
# The first is that after even after decreasing our iterative step-size by a specfied factor, there is no decrease in the loss 
# (in above, the factor is 10**"200" - this is not really a sensible number, since it's much greater than machine precision). 
# This corresponds to having arrived at a stationary point of the loss function. The second is that the lowest recorded error 
# rate hasn't decreased in a specified number of iterations (in the above, that's 400 * 10 = 4000 iterations). These numbers are
# quite arbitrary and should be changed freely.  			
	
