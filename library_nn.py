# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

from sklearn.neural_network import MLPClassifier
import random
import numpy as np
import cv2
# ============== Scilearn Neural Network Usage ===================
# Use scikearn neural networks library function to compare against our
# implementation

# Training Data Processing
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
labels = np.array(labels) #shape = 184x1

#Test Data processing
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

#MLP_Classifier
X_train = train_set
y_train = labels
X_test = test_set
y_test = np.array(labels_test)

#sgd => Stochastic Gradient Descent
mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,),learning_rate='constant',learning_rate_init=.1, max_iter=1000)
mlp.fit(X_train, y_train)  
mlp.predict_proba(X_test)
print("Test set score: %f" % mlp.score(X_test, y_test))
                       



