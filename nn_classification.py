import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smp
import math
import random
from time import time
from PIL import Image

training_images_file = open('train-images.idx3-ubyte','rb')
training_images = training_images_file.read()
training_images_file.close()

training_labels_file = open('train-labels.idx1-ubyte', 'rb')
training_labels = training_labels_file.read()
training_labels_file.close()

training_images = bytearray(training_images)
training_images = training_images[16:]

training_labels = bytearray(training_labels)
training_labels = training_labels[8:]

training_images = np.array(training_images).reshape(60000, 784)
training_labels = np.array(training_labels)

training_images_1 = training_images[np.where(training_labels == 1)]
training_labels_1 = training_labels[np.where(training_labels == 1)].reshape(training_images_1.shape[0],1)
training_images_1 = np.column_stack((training_images_1, training_labels_1))

training_images_2 = training_images[np.where(training_labels == 2)]
training_labels_2 = training_labels[np.where(training_labels == 2)].reshape(training_images_2.shape[0],1)
training_images_2 = np.column_stack((training_images_2, training_labels_2))

training_images_7 = training_images[np.where(training_labels == 7)]
training_labels_7 = training_labels[np.where(training_labels == 7)].reshape(training_images_7.shape[0],1)
training_images_7 = np.column_stack((training_images_7, training_labels_7))

mask_1 = random.sample(range(training_images_1.shape[0]), 200)
mask_test_1 = random.sample(range(training_images_1.shape[0]), 50)

mask_2 = random.sample(range(training_images_2.shape[0]), 200)
mask_test_2 = random.sample(range(training_images_2.shape[0]), 50)

mask_7 = random.sample(range(training_images_7.shape[0]), 200)
mask_test_7 = random.sample(range(training_images_7.shape[0]), 50)

test_images_1 = training_images_1[mask_test_1]
test_images_2 = training_images_2[mask_test_2]
test_images_7 = training_images_7[mask_test_7]

training_images_1 = training_images_1[mask_1]
training_images_2 = training_images_2[mask_2]
training_images_7 = training_images_7[mask_7]

training_images = np.vstack((training_images_1, training_images_2, training_images_7))
test_images = np.vstack((test_images_1, test_images_2, test_images_7))

def NNClassify(x, k):
    result = np.array([], dtype = 'int8')
    for test_image in x:
        temp_result = []
        for image in range(training_images.shape[0]):
            distance = np.sqrt(np.sum((test_image - training_images[image]**2)))
            temp_result.append((image, distance))
        n = np.array(temp_result, dtype=[('x', int), ('y', float)])
        n.sort(order='y')

        nearest = []
        for i in range(k):
            nearest.append(training_images[n[i][0], 784])
        nearest = np.array(nearest)
        counts = np.bincount(nearest)
        result = np.append(result, np.argmax(counts))
    return result

def get_accuracy(predictions, labels):
    acc = 0
    for i in range(predictions.shape[0]):
        if predictions[i] == labels[i]:
            acc += 1
    return acc/predictions.shape[0]

def get_accuracy2(predictions, labels):
    acc = 0
    for i in range(predictions.shape[0]):
        if predictions[i] == labels[i]:
            acc += 1
        else:
            img = Image.fromarray(test_images[i,:784].reshape((28,28)))
            img.show()
            print("Prediction: {} | Label: {}".format(predictions[i], labels[i]))
    return acc/predictions.shape[0]    

def NN_cross_validate(n, k):
    folds = np.split(training_images, k)
    total_acc = 0
    for i in range(k):
        test_set = np.array(folds[i])
        training_set = None
        for j in range(5):
            if i == j:
                continue
            if training_set is None:
                training_set = np.vstack((folds[j]))
            else:
                training_set = np.vstack((training_set, folds[j]))
        result = NNClassify(test_set, n)
        total_acc += get_accuracy(result, test_set[:,784])
    avg_accuracy = float(total_acc / k)
    return avg_accuracy

def determine_model(nn, k):
    best_n = -1
    best_avg = -1
    for n in nn:
        avg = NN_cross_validate(n, k)
        if avg > best_avg:
            best_avg = avg
            best_n = n
    return best_n, best_avg

nearest_neighors = np.array([1,3,5,7,9])
best_n, best_avg = determine_model(nearest_neighors, 5)

r = NNClassify(test_images, best_n)
a = get_accuracy(r, test_images[:, 784])

t = test_images[:, 784]
print(r)
print(test_images[:, 784])
print(best_avg)
print(a)

# a2 = get_accuracy2(r, test_images[:,784])
equal = np.where(np.equal(r,test_images[:,784]))
not_equal = np.where(np.not_equal(r,test_images[:,784]))

# for i in range(5):
#     print(i)
#     img = Image.fromarray(test_images[equal[0][i],:784].reshape((28,28)))
#     img.show()
#     # print("Prediction: {} | Label: {}".format(r[i], t[i]))

for i in range(5):
    print(i)
    img = Image.fromarray(test_images[not_equal[0][i],:784].reshape((28,28)))
    img.show()
    print("Prediction: {} | Label: {}".format(r[not_equal[0][i]], test_images[not_equal[0][i], 784]))