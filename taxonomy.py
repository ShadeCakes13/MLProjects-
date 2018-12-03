import urllib.request
from urllib.request import urlopen
import pandas as pd
import numpy as np

#Import and process data. 

data = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data')
data = data.readlines()
data = [x.decode() for x in data]
data = [x.split(',') for x in data]
attributes = ['hair','feathers','eggs','milk','airborne','aquatic',
              'predator','toothed','backbone','breathes','venemous',
              'fins','legs','tail','domestic']
animal_names = ['x']*101
animal_vectors = ['x']*101
for i in range(0,len(data)):
    animal_names[i] = data[i][0]
    animal_vectors[i] = data[i][1:16]
df = pd.DataFrame(data = animal_vectors, index = animal_names, columns = attributes)
df.iloc[:,0:16] = df.iloc[:,0:16].apply(pd.to_numeric)


# We'll be implementing Lloyd's ALgorithm to cluster our data and simultaneously we'll be calculating the 'total distance' incurred by the clusters (we take the 2-norm distance between a point and the mean of the cluster to which it's assigned, then sum these distances over all points to get our total distance); the total distance is what we're trying to minimise via Lloyd's. Our data consists of 101 animals with 15 attributes (stored as a vector); our programme will therefore organise the animals into groups with similar features. 

# First, we define a function to perform the iterative step: we take a list of the position of  cluster centres ('vectorlst'), use this to cluster the data (return 'clusters'), calculate the new cluster centres ('new_vectorlst') and calculate the 'total_distance' incurred by the new clusters.
 
def cluster(vectorlst):
    n = len(vectorlst)
    cluster_index = ['x']*101
    total_distance = 0
    for i in range(0,101):
        label = 0
        x = df.iloc[i].values
        distance = np.linalg.norm(x-vectorlst[0])
        for j in range(0,n):
            new_distance = np.linalg.norm(x-vectorlst[j])
            if new_distance < distance:
                distance = new_distance
                label = j 
        cluster_index[i] = label
        total_distance+=distance
    labels = np.array(cluster_index)
    clusters = ['x']*n
    for k in range(0,n):
        clusters[k] = np.where(labels==k)
    new_vectorlst = ['x']*n
    for l in range(0,n):
        new_vectorlst[l] = df.iloc[clusters[l]].values.mean(axis=0)
    return new_vectorlst, clusters, total_distance 


# We implement the 'kmeans' algorithm by taking a randomn sample of size n from among our points to be our initial cluster centres, repeatedly perfoming the iterative step above terminating when the total distance doesn't move between iterations (a necessary and sufficient condition for the clusters not to change). The function below prints these clusters, along with a the total distance incurred by this taxonomy. 
 
def kmeans(n):
    vectorlst = cluster(df.sample(n).values)
    new_vectorlst = cluster(vectorlst[0])
    while vectorlst[2] != new_vectorlst[2]:
        vectorlst = new_vectorlst
        new_vectorlst = cluster(new_vectorlst[0])
    final_vectorlst = new_vectorlst[1]
    for i in range(0,n):
        print("Taxon " + str(i) + ": " + ','.join(df.index[final_vectorlst[i]]))
        print("")
    print("Distance: " + str(new_vectorlst[2]))

# For use in further functions, we use create another 'kmeans' function, except one that returns rather than prints the output. Also, in the new kmeans functions, we specify the initial cluster centres rather than randomnly generating them. 
   

def kmeans_no_print(initial_vectorlst):
    vectorlst = cluster(initial_vectorlst)
    new_vectorlst = cluster(vectorlst[0])
    while vectorlst[2] != new_vectorlst[2]:
        vectorlst = new_vectorlst
        new_vectorlst = cluster(new_vectorlst[0])
    return new_vectorlst

# Lloyd's algorithm minimises the clustering problem locally and not, in general, globally. Initialising with different cluster centres can result in different local solutions. We define a fuction that randomnly generates m lists of initialising cluster centres and calculates m local solutions to n-clustering problem; the solution with the least total distance is printed. 

def kmeans_best(n,m):
	samples = ['x']*m
	for i in range(0,m):
		samples[i] = df.sample(n).values
	best = kmeans_no_print(samples[0])
	for j in range(1,m):
        	new = kmeans_no_print(samples[j])
       		if new[2] < best[2]:
         		best = new
	print("")
	for k in range(0,n):
                print("Taxon " + str(k) + ": " + ','.join(df.index[best[1][k]]))
                print(' ')
	print("Distance: " + str(best[2]))

#Example (run the algorithm 10 times to group animals in 7 clusters):  

print(kmeans_best(7,10))

