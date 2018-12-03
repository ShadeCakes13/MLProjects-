import urllib.request
from urllib.request import urlopen
import pandas as pd
import numpy as np
from ete3 import Tree

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

#We'll trying to hierarchically cluster our animal data (101 animals with a vector of 15 attirbutes) to come up with a taxonomy.
#We'll generate a tree by implementing a greedy algorithm that, at each step, pairs the "closest" subtrees/leaves. As a measure 
#of distance between two subtrees, the mean of each subtree is taken (the mean of the elements in a subtree) and the 2-norm 
#distance between these means is calculated. Trees will be first represented as embedded lists and then graphically. 

#Following the above outline, we create a function to pair the "closest" elements in a lst. A list is inputted and the indices 
#('index1' and 'index2') of the "closest" elements are outputted, along with their 2-norm difference ('diff').

def min_diff(lst):
	n = len(lst)
	diff = np.linalg.norm(lst[0]-lst[1],2)
	index1 = 0
	index2 = 1
	for i in range(0,n-1):
		for j in range(i+1,n):
			new_diff = np.linalg.norm(lst[i]-lst[j],2)
			if new_diff < diff:
				diff = new_diff
				index1 = i
				index2 = j
	return diff, index1, index2

# Next, we need to find a way to access all the elements (animal vectors) in a subtree. The following function takes an embedded 
#list (tree) and returns a list of all elements e.g. defoliant1([[a,b],[[b],[c,d,e]]]) = [a,b,b,c,d,e]. The function 'defoliant' 
#extends this process to included leaves i.e. defoliant(a) = [a]. 
 
def defoliant1(lstlst):
	truth_list = [type(x)==np.ndarray for x in lstlst]
	while not all(truth_list):
		new_lstlst = [] 
		for lst in lstlst:
			if type(lst) == np.ndarray:
				new_lstlst.append(lst)
			else:
				new_lstlst += lst
				lstlst = new_lstlst
		truth_list = [type(x)==np.ndarray for x in lstlst]
	return lstlst
 
def defoliant(lstlst):
	if type(lstlst) == np.ndarray:
		lstlst = [lstlst]
	else:
		lstlst = defoliant1(lstlst)
	return lstlst


#Next we define the greedy iterative step to make our tree (embedded list). We input an embedded list of attribute vectors and a 
#corresponding embedded list of animal names (representing the current tree). Two new embedded lists are outputted, representing 
#the tree after that results from pairing the "closest" subtrees.  

def pair(lstlst,lstlstnam):
	lst = [np.mean(defoliant(x),axis=0) for x in lstlst]
	index1 = min_diff(lst)[1]
	index2 = min_diff(lst)[2]
	new_entry = [lstlst[index1], lstlst[index2]]
	new_entry_nam = [lstlstnam[index1],lstlstnam[index2]]
	lstlst.pop(index1)
	lstlst.pop(index2-1)
	lstlst.append(new_entry)
	lstlstnam.pop(index1)
	lstlstnam.pop(index2-1)
	lstlstnam.append(new_entry_nam)
	return lstlst, lstlstnam

#Now we input our original list of animals ('lst_names') and the corresponding list of attributes (lst_vectors) and an embedded
#list of names representing a tree is outputted. 
 
def tree(lst_vectors, lst_names):
	length = len(lst_vectors)
	count = 0
	while count < length-1:
		lst_vectors, lst_names = pair(lst_vectors,lst_names)
		count += 1
	return lst_names

#To visualise our tree, we use the 'Tree' module from the 'ete3' package.

def print_tree(lst_vectors,lst_names):
	x = str(tree(lst_vectors,lst_names)).replace('[','(')
	x = x.replace(']',')')
	print(Tree(str(x)+";"))

#To facilitate retrieving the necessary data from our dataframe, we define the the function below. We input a numerical list of 
#all the animals we want to include in our taxonomy e.g. quick_print([0,2]) would include the first and second animal ('aardvark' 
#and 'bass'), quick_print(range(0,101)) would include all our animals. 

def quick_print(lst):
	print_tree(list(df.iloc[lst,:].values),list(df.index[lst]))

#Example (taxonomy of all the animals):
quick_print(range(0,101))

	


 



 


 
