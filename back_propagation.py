# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np
import cv2
# ==============Back Propagation Algorithm==================================

def main():

	# load txt file
# load txt file
	f = open("downgesture_train.list", 'r')
	result_matrix = []
	labels = []
	for line in f.readlines():
		values_as_strings = line.split('\t')
		file_path = ''.join(values_as_strings).rstrip('\n')
		img = cv2.imread(file_path, -1)
		result_matrix.append(img.flatten())

		if "down" in file_path:
			labels.append([1])
		else:
			labels.append([0])
	f.close()
	train_set = np.array(result_matrix) #shape = 184x960
	# print train_set
	# print train_set.shape
	labels = np.array(labels) #shape = 184x1
	input_neurons = train_set.shape[1] #960
	hidden_layer_neurons = 100
	output_neuron = 1
	#initialize all w randomly between -1000 to 1000
	weights_hidden_layer = np.random.uniform(low=-1, high=1, size=(input_neurons,hidden_layer_neurons)) #shape 960x100
	weights_output_layer = np.random.uniform(low=-1, high=1, size=(hidden_layer_neurons,output_neuron)) #shape 100x1

	weights_output, weights_hidden = NN(train_set,labels,weights_hidden_layer, weights_output_layer)
	# print "weights_output", weights_output
	# print "weights_hidden", weights_hidden

	#test
	t = open("downgesture_test.list", 'r')
	test_matrix = []
	labels_test = []
	for line in t.readlines():
		values_as_strings = line.split('\t')
		file_path = ''.join(values_as_strings).rstrip('\n')
		img = cv2.imread(file_path, -1)
		test_matrix.append(img.flatten())

		if "down" in file_path:
			labels_test.append([1])
		else:
			labels_test.append([0])

	test_set = np.array(test_matrix) #shape = 184x960
	computed_labels = forward_propagation(test_set,weights_hidden,weights_output,labels_test)
	print "labels", labels_test
	print "computed_labels", computed_labels
	percent_accuracy = accuracy(labels_test,computed_labels)

	print "Accuracy", percent_accuracy

def accuracy(true_labels, predictions):
	correct = 0.0
	for i in range(len(true_labels)):
		print 'predictions[i]',int(predictions[i])
		if true_labels[i][0] == int(predictions[i]):
			print 'correct',correct
			correct += 1.0

	return float((correct/len(true_labels)))*100.0

def forward_propagation(x,wh,wo,y):
	hiddenLayerInputs = np.dot(x,wh) #shape 184x100
	hiddenLayerOutput = sigmoid(np.array(hiddenLayerInputs)) #184x100
	outputLayerInputs = np.dot(hiddenLayerOutput,wo) #shape 184x1
	outputLayerOutput = sigmoid(np.array(outputLayerInputs)) #184x1
	outputLayerOutput = np.array(outputLayerOutput)
	return outputLayerOutput


	
def NN(x, y, wh, wo):
	epochs = 1000
	eta = 0.1
	hiddenLayerInput = []
	hiddenLayerOutput = []
	while(epochs > 0):
		#Forward
		hiddenLayerInputs = np.dot(x,wh) #shape 184x100
		# print "hiddenLayerInputs", hiddenLayerInputs.shape
		hiddenLayerOutput = sigmoid(np.array(hiddenLayerInputs)) #184x100
		# print "hiddenLayerOutput", hiddenLayerOutput.shape
		outputLayerInputs = np.dot(hiddenLayerOutput,wo) #shape 184x1
		outputLayerOutput = sigmoid(np.array(outputLayerInputs)) #184x1
		outputLayerOutput = np.array(outputLayerOutput)
		# print "outputLayerOutput", outputLayerOutput.shape
		# print "y", y.shape

		#Back propagation
		error = squaredError(y, outputLayerOutput)
		# print "error", error.shape
		deltaOutputLayer = der_sig(outputLayerOutput) #184x1
		deltaOutput = 2*error*deltaOutputLayer
		# print "deltaOutput", deltaOutput.shape
		errorHiddenLayer = deltaOutput.dot(wo.T) #184x100
		# print "errorHiddenLayer", errorHiddenLayer.shape
		deltaHiddenLayer = der_sig(hiddenLayerOutput) #184x100
		deltaHidden = errorHiddenLayer * deltaHiddenLayer

		#error		
		wo -= hiddenLayerOutput.T.dot(deltaOutput)*eta
		wh -= x.T.dot(deltaHidden)*eta
		print "wo", wo
		print "wh", wh

		epochs -= 1

	return wo,wh


def squaredError(y, output):
	return (output-y)**2
def sigmoid(s):
	return 1/(1 + np.exp(-s))
def der_sig(s):
	return s*(1-s)

if __name__ == "__main__":
	main()