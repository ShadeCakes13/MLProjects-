import urllib.request
from urllib.request import urlopen
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Import and process data.

data = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
data = data.readlines()
data = [x.decode() for x in data]
data = [x.split(',') for x in data]
data = data[0:150]
df = pd.DataFrame(data)
df.iloc[:,0:4] = df.iloc[:,0:4].apply(pd.to_numeric)
columns = ['Sepal Length','Sepal Width','Petal Length','Petal Width','Iris Class'] 
df.columns = columns
iris_classes = list(set(df['Iris Class']))


#Select two variables for comparison.

def variable_select(variable1,variable2):
	return df.iloc[:,[variable1,variable2,4]]

#Plot two variables for comparison. Points are coloured according to 'Iris Class'.  

def scatter_plot(variable1,variable2):
	df = variable_select(variable1,variable2)
	for i in range(0,3):
		colour = "C"+str(i)
		new_df = df.loc[df['Iris Class']==iris_classes[i]]
		a = new_df.iloc[:,0].values
		b = new_df.iloc[:,1].values
		plt.scatter(a,b,color=colour,s=20, label =iris_classes[i])
	title = columns[variable1] + ' versus ' + columns[variable2]
	xlabel = columns[variable1] + ' (cm)'
	ylabel = columns[variable2] + ' (cm)'
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(loc =4 , prop={'size': 6})
	plt.show()


#Lloyd's Algorithm:

#The Lloyd's Algorithm works by iteratively classifying points according to which cluster mean is closest (here, we use the 2-norm). 
#In the function below, 'vector' refers to the point we wish to classify and 'vectorlst' refers to a list of cluster means. 

def nearest(vector,vectorlst):
	distance = np.linalg.norm(vector - vectorlst[0],2)
	label = 0 	
	for i in range(0,len(vectorlst)):
		new_distance = np.linalg.norm(vector - vectorlst[i],2)
		if new_distance < distance:
			distance = new_distance 
			label = i
	return label 


#The function below defines the iterative step in Lloyd's algorithm: we take our dataframe representing the current data classification 
#('df') and a current list of cluster means ('vectorlst') and use our 'nearest' function to reclassify the points (each dataframe in #'dataframes' corresponds to a cluster) and calculate a list of new cluster means ('new_vectorlst'). 

def label(df,vectorlst):
	if np.shape(df)[1] > 2:
		df = df.drop(df.columns[2], axis = 1)
	
	n = np.shape(df)[0]
	m = len(vectorlst)
	lst = ['x']*n
	for i in range(0,n):
		lst[i] = nearest(df.loc[i].values,vectorlst)
	new_label = pd.DataFrame({'label':lst})
	new_df = pd.concat([df,new_label],axis=1)
	
	new_vectorlst = ['x']*m
	dataframes = ['x']*m
	for i in range(0,m):
		sub_df = df.loc[new_df['label']==i]
		new_vectorlst[i] = sub_df.mean(axis=0).values
		dataframes[i] = sub_df
	return new_vectorlst, dataframes 


#For the purposes of visualising the algorithm, we want to plot the decision boundary between clusters at each iteration. 
#The 'nearest neighbour' nature of the classification means that we'll end up with so-called Voronoi cells. In the fuction below, 
#the 'vectorlst' refers to the points that generate each cell, the axes determine what rectangle of the 2-D plane is plotted i.e.
#xaxis = [0,1] means we'll plot in the range 0<x<1. 

def voronoi(vectorlst,xaxis,yaxis):
	n = len(vectorlst)
	x = np.linspace(xaxis[0],xaxis[1],300)
	y = np.linspace(yaxis[0],yaxis[1],300)
	X,Y = np.meshgrid(x,y)
	function_lst = ['x']*n
	W = np.ones((300,300))*100000
	for i in range(0,n):
		function_lst[i] = (X-vectorlst[i][0])**2+(Y-vectorlst[i][1])**2
	for i in range(0,n):
		for j in range(0,n):
			if j != i:		
				W = np.minimum(W,function_lst[j])
		Z = function_lst[i] - W
		plt.contour(X,Y,Z,[0])

#Putting it all together, we create an animation to visualise each iteration in Lloyd's Algorithm. In the function below, 'df' 
#refers to a dataframe that represents the 2-D data we wish to cluster, 'vectorlst' gives us initial cluster centres and 'time' 
#refers to time in seconds between the frames in our animation. Points will be coloured according to which cluster they belong 
#to at each iteration, the lines demarcate the decision boundary between clusters and the crosses mark the position of the 
#cluster meas at each iteration. Llyod's Algorithm terminates when the cluster means don't change between iterations; we use the 
#variable 'truth_lst' to determined when this is the case.  

def animate(df,vectorlst,time):
	xaxis = [np.min(df.iloc[:,0]),np.max(df.iloc[:,0])]
	yaxis = [np.min(df.iloc[:,1]),np.max(df.iloc[:,1])]
	title = df.columns[0] + ' versus ' + df.columns[1]
	xlabel = df.columns[0] + ' (cm)'
	ylabel = df.columns[1] + ' (cm)'
	n = len(vectorlst)
	truth_lst = [False]

	a,b = df.iloc[:,0].values, df.iloc[:,1].values
	plt.scatter(a,b)
	plt.title(title) 
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.pause(time)

	vf = pd.DataFrame(vectorlst)
	a,b = vf.iloc[:,0].values,vf.iloc[:,1].values
	plt.scatter(a,b, marker = 'x', c = 'black', s =200)
	plt.pause(time)

	while not all(truth_lst): 
		plt.close()
		fig, ax = plt.subplots()
		old_vectorlst = vectorlst
		vectorlst = label(df,vectorlst)[0]
		vf = pd.DataFrame(vectorlst)
		dataframes = label(df,vectorlst)[1]
		for i in range(0,n):
			colour = "C"+str(i)
			a,b = dataframes[i].iloc[:,0].values, dataframes[i].iloc[:,1].values
			plt.scatter(a,b,color=colour)
		voronoi(vectorlst,xaxis,yaxis)
		plt.title(title) 
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.pause(time)
		a, b = vf.iloc[:,0].values, vf.iloc[:,1].values
		plt.scatter(a,b, marker = 'x', c = 'black', s =200)
		plt.pause(time)
		truth_lst = ['x']*n
		for i in range(0,n):
			truth_lst[i] = old_vectorlst[i] - vectorlst[i]
		truth_lst = [np.linalg.norm(x)==0 for x in truth_lst]	
	plt.show()



#For ease, we define the function below. We specify the indexes of the two variables to compare, as the number of clusters ('n')
#into which we want to divide the data and the time in seconds between the frames of the animations. The initialising cluster 
#points for Lloyd's Algorithm are randomnly sampled from among the data points. Also, after closing the final frame in our 
#animation, we are shown the same scatter graph but with the points instead coloured according to the 'Iris Class'. Since their
#are three iris classes, we would hope that for n=3, this scatter graph would resemble the final frame i.e. LLoyd's algorithm has
#successfully clustered the data into known categories. 

def reanimate(variable_index_1, variable_index_2, n, time):
	df = variable_select(variable_index_1,variable_index_2)
	vectorlst = df.sample(n)	
	vectorlst = vectorlst.drop(df.columns[2], axis = 1)
	vectorlst = [vectorlst.iloc[x].values for x in range(0,n)]
	animate(df,vectorlst,time)
	scatter_plot(variable_index_1,variable_index_2)
	
#Example animation: 
reanimate(1,3,3,1)











