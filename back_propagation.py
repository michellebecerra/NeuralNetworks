# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np

# ==============Back Propagation Algorithm==================================

def main():

	# load txt file
	f = open("downgesture_train.list", 'r')
	result_matrix = []
	labels = []
	for line in f.readlines():
	    values_as_strings = line.split('\t')
	    result_matrix.append(values_as_strings)
	    file_path = ''.join(values_as_strings)
	    if "down" in file_path:
	    	labels.append(1)
	    else:
	    	labels.append(0)

	train_set = np.array(result_matrix)
	labels = np.array(labels)
	weights = np.random.uniform(low=-1000, high=1000, size=100)
	print "weights", weights
	print "train_set", train_set
	print "labels", labels
	
	weights = back_propagation(train_set, labels, weights)

def back_propagation(x, y, w):
	eta = 0.1
	d = 100
	return w

if __name__ == "__main__":
    main()