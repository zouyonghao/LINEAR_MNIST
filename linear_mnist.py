import numpy as np
import struct
import matplotlib.pyplot as plt
import pylab
import pca
import tools
import linear

train_data = tools.get_train_data()
train_label = tools.get_train_label()

dimension = 50

train_data, eigenVectors = pca.PCA(train_data, dimension)
print(np.array(train_data).shape)

num1 = 1
num2 = 9

# 10 classes
classified_data = []
for i in range(10):
    classified_data.append([])

for i in range(len(train_label)):
    # print(train_label[i][0])
    classified_data[train_label[i][0]].append(train_data[i])

train_data1 = classified_data[num1]
train_data2 = classified_data[num2]

linear_classifier = linear.LinearClassification()
linear_classifier.add_train_data(train_data1, 1)
linear_classifier.add_train_data(train_data2, -1)
linear_classifier.learn_speed_start_init(0.1)
linear_classifier.r_init(1000)
linear_classifier.train_count_init(2000)
linear_classifier.min_error_signal_init(0.01)

# 开始训练
linear_classifier.train()

raw_test_data = tools.get_test_data()

# raw_test_data = pca.normalize2D(raw_test_data)
test_data = np.matmul(raw_test_data, eigenVectors)
test_label = tools.get_test_label()

classified_test_data = []
for i in range(10):
    classified_test_data.append([])

for i in range(len(test_label)):
    # print(train_label[i][0])
    classified_test_data[test_label[i][0]].append(test_data[i])

test_data1 = classified_test_data[num1]
test_data2 = classified_test_data[num2]

correct1 = 0
count1 = 0

for i in range(len(test_data1)):
    count1 += 1
    predict = linear_classifier.classify(test_data1[i])
    if predict == 1:
        correct1 += 1

print(correct1 / len(test_data1))

correct2 = 0
count2 = 0

for i in range(len(test_data2)):
    count2 += 1
    predict = linear_classifier.classify(test_data2[i])
    if predict == -1:
        correct2 += 1

print(correct2 / len(test_data2))

print((correct1 + correct2) / (len(test_data2) + len(test_data1)))