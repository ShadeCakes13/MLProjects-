import urllib.request
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, multivariate_normal

# Import and process data. This consists of 178 bottles of wine, each with one of three labels and with an associated 13-D feature vector. 
data = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data').readlines()
data = [x.decode() for x in data]
data = [x.split(',') for x in data]
featurenames = ['Label','Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
df = pd.DataFrame(data,columns = featurenames)
df = df.apply(pd.to_numeric)

# We will be building a Gaussian Generative Model to be able to label bottles of wine. First, we randomnly divide the data into training (130 bottles) and test data (48 bottles).

perm = np.random.permutation(178)
 
trainx = df.loc[perm[0:130],'Alcohol':'Proline']
trainy = df.loc[perm[0:130],'Label']

testx = df.loc[perm[130:178],'Alcohol':'Proline']
testy = df.loc[perm[130:178],'Label']
 
# We now calculate some probabilities necessary for our Bayesian model, as well finding the indices for of training bottles according to their labels. 

indices = ['x']*3
pis = ['x']*3
for i in range(1,4):
	pis[i-1] = sum(trainy==i)/130.0
	indices[i-1] = np.where(trainy==i)

# Assuming the data follows a multivariate Gaussian distributions (one for each wine label), we construct a classifier by estimating these distributions using our training data. In the function below, we input a list of variables to include in our model ('feat_lst') (e.g. ['Alcohol','Proline'] and the corresponding mean and covariance of the multivariate Gaussian for each label are outputted. 

def gaussians(feat_lst):
	mu = ['x']*3
	c = ['x']*3
	for i in range(0,3):
		data = trainx.iloc[indices[i]][feat_lst]
		mu[i] = np.mean(data).values
		c[i] = np.cov(np.vstack(data.values),rowvar=0)
	return mu, c

# We are now ready to classify bottles. In the first function, the features on which we base our classification are inputter ('feat_lst') along with the vector to be classified. The second function classifies an entry in our test data (specified by 'index') and prints the entry's actual label alongside our label. 

def classifier(x,feat_lst):
	value = 0
	g = gaussians(feat_lst)
	for i in range(0,3):
		mu, c = g[0][i], g[1][i]
		p = pis[i]*multivariate_normal.pdf(x,mean =mu, cov = c)
		if p > value:
			label = i + 1
			value = p
	return label

def test_classifier(index,feature_lst):
    x = testx.iloc[index][feature_lst]
    predicted_label = classifier(x, feature_lst)
    actual_label = testy.iloc[index]
    print("Actual Label: " + str(actual_label))
    print("Predicted Label: " + str(predicted_label))


# We test the success of our classification model. In the function we input the features on which we wish to base our classification, this model is applied to label the test data and the error (as percentage) is outputted. 
 
def test_classifier_error(feature_lst):
	n = np.shape(testx)[0]
	error = 0
	mu, c = gaussians(feature_lst) 
	for j in range(0,n):
		x = testx.iloc[j][feature_lst]
		value = 0
		for l in range(0,3):
			pdf = multivariate_normal.pdf(x,mean=mu[l],cov=c[l])
			p = pis[l]*pdf
			if p > value:
				label = l+1
				value = p
		if label != testy.iloc[j]:
			error += 1
	error_rate = 100.0*error/n
	print('Test Size: ' + str(n))
	print('No. Label Errors: ' + str(error))
	print('Error Rate (%): ' + str(error_rate))

# We would suspect that the more features that are included in our classification, the lower the error rate. Below we plot the error rate against the number of features included (going from left to right in our 'featurenames' list). First, we redefine the function above to return the error rate rather than printing anything. 

def test_classifier_error1(feature_lst):
	n = np.shape(testx)[0]
	error = 0
	mu, c = gaussians(feature_lst) 
	for j in range(0,n):
		x = testx.iloc[j][feature_lst]
		value = 0
		for l in range(0,3):
			pdf = multivariate_normal.pdf(x,mean=mu[l],cov=c[l])
			p = pis[l]*pdf
			if p > value:
				label = l+1
				value = p
		if label != testy.iloc[j]:
			error += 1
	return 100.0*error/n 

def error_plot():
	error = ['x']*(len(featurenames)-1)
	ran = range(1,len(featurenames))
	for i in ran:
		error[i-1]=test_classifier_error1(featurenames[1:i+1])
	plt.plot(ran,error)
	plt.title('Error Rate versus No. Features in Model') 
	plt.xlabel('No. Features')
	plt.ylabel('Error Rate (%)')
	plt.show()

# To visualise the classifier, we restrict ourselves to 2-D. Based on the two features inputted, we construct plot, showing the test data as well as the the decision boundary of our model.  


def plot_model(feature1,feature2):

	feature_lst = [feature1,feature2] 
	mu, c = gaussians(feature_lst)
	E = np.identity(2)
	
# Here we calculate the range for our plot axes. 

	std = ['x']*2
	for i in range(0,2):		
		E[i][0] = min(testx[[feature_lst[i]]].values)[0]
		E[i][1] = max(testx[[feature_lst[i]]].values)[0]
		std[i] = np.var(testx[[feature_lst[i]]].values)**0.5

	x = np.linspace(E[0][0]-std[0],E[0][1]+std[0],1000)
	y = np.linspace(E[1][0]-std[1],E[1][1]+std[1],1000)

# Here we plot the decision boundary between the labels. 

	X,Y = np.meshgrid(x,y)
	mesh = np.dstack((X,Y))
	N = ['x']*3
	Z = ['x']*3
	for i in range(0,3):
		N[i] = multivariate_normal(mu[i],c[i])
		Z[i] = 	pis[i]*N[i].pdf(mesh)
	U = np.minimum(Z[0]-Z[1],Z[0]-Z[2])
	V = np.minimum(Z[1]-Z[0],Z[1]-Z[2])
	plt.contour(X,Y,U,[0],color='black')
	plt.contour(X,Y,V,[0],color='black')

# Here we plot the test data, coloured according to their actual label. 

	for i in range(0,3):
		plt.scatter(testx.iloc[np.where(testy == i+1)][feature_lst[0]].values,testx.iloc[np.where(testy == i+1)][feature_lst[1]].values,color='C'+str(i))
	
	plt.title(str(feature_lst[0]) + ' versus ' + str(feature_lst[1]))
	plt.xlabel(str(feature_lst[0]))
	plt.ylabel(str(feature_lst[1]))
	plt.show()


#Example:

plot_model('Total phenols','Color intensity')







































 
     
