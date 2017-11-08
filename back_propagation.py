# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np
import cv2
# ==============Back Propagation Algorithm==================================

#Predicts the down gester by training and testing the gesters by a
#feed forward neural network
def main():

	#Train
	# load training txt file
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

	#variables
	train_set = np.array(result_matrix) #shape = 184x960
	labels = np.array(labels) #shape = 184x1
	input_neurons = train_set.shape[1] #960
	hidden_layer_neurons = 100
	output_neuron = 1

	#initialize all w randomly between -1000 to 1000
	weights_hidden_layer = np.random.uniform(low=-1000, high=1000, size=(input_neurons,hidden_layer_neurons)) #shape 960x100
	weights_output_layer = np.random.uniform(low=-1000, high=1000, size=(hidden_layer_neurons,output_neuron)) #shape 100x1

	#train the neural network with train set
	weights_output, weights_hidden = NN(train_set,labels,weights_hidden_layer, weights_output_layer)

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

	#variables
	test_set = np.array(test_matrix) #shape = 184x960

	#test the neural network using the computed weights by training
	computed_labels = forward_propagation(test_set,weights_hidden,weights_output,labels_test)

	#predictions
	print "True labels:\n", labels_test
	print "Predicted labels:\n", computed_labels

	#calculating accuracy of predictions
	percent_accuracy = accuracy(labels_test,computed_labels)
	print "Accuracy:1", percent_accuracy,'%'

#calculates the percentage of correctly predicted labels
def accuracy(true_labels, predictions):
	correct = 0.0
	for i in range(len(true_labels)):
		if true_labels[i][0] == int(predictions[i]):
			correct += 1.0

	return float((correct/len(true_labels)))*100.0

#computes the labels given weights
def forward_propagation(x,wh,wo,y):
	hiddenLayerInputs = np.dot(x,wh) #shape 184x100
	hiddenLayerOutput = sigmoid(np.array(hiddenLayerInputs)) #184x100
	outputLayerInputs = np.dot(hiddenLayerOutput,wo) #shape 184x1
	outputLayerOutput = sigmoid(np.array(outputLayerInputs)) #184x1
	outputLayerOutput = np.array(outputLayerOutput)
	return outputLayerOutput

#Back propagation Algorithm implementation
#Trains neural network using train set with 1000 Epochs
#computes weights of hidden layer and output layer
def NN(x, y, wh, wo):
	epochs = 1000
	eta = 0.1
	hiddenLayerInput = []
	hiddenLayerOutput = []
	while(epochs > 0):
		#1-Forward
		hiddenLayerInputs = np.dot(x,wh) #shape 184x100
		hiddenLayerOutput = sigmoid(np.array(hiddenLayerInputs)) #184x100
		outputLayerInputs = np.dot(hiddenLayerOutput,wo) #shape 184x1
		outputLayerOutput = sigmoid(np.array(outputLayerInputs)) #184x1
		outputLayerOutput = np.array(outputLayerOutput)

		#2-Back propagation
		error = squaredError(y, outputLayerOutput)
		deltaOutputLayer = der_sig(outputLayerOutput) #184x1
		deltaOutput = 2*error*deltaOutputLayer
		errorHiddenLayer = deltaOutput.dot(wo.T) #184x100
		deltaHiddenLayer = der_sig(hiddenLayerOutput) #184x100
		deltaHidden = errorHiddenLayer * deltaHiddenLayer

		#3-error		
		wo = np.subtract(wo,((hiddenLayerOutput.T.dot(deltaOutput))*eta))
		wh = np.subtract(wh,((x.T.dot(deltaHidden))*eta))

		epochs -= 1

	return wo,wh

#(x-y)^2
def squaredError(y, output):
	return (output-y)
#1/1+e^-s
def sigmoid(s):
	return 1/(1 + np.exp(-s))
#s(1-s)
def der_sig(s):
	return s*(1-s)

if __name__ == "__main__":
	main()