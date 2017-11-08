from sknn.mlp import Classifier, Layer
import random
import numpy as np
import cv2

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
nn = Classifier(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Linear")],
    learning_rate=0.1,
    n_iter=1000)
nn.fit(train_set, labels)

y_valid = nn.predict(test_set)
print y_valid
score = nn.score(test_set, labels)