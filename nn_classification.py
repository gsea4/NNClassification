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
    # result = np.array((0,0), dtype = 'int32')
    result = []
    for image in range(training_images.shape[0]):
        sum = 0
        for pixel in range(28*28):
            sum += math.pow((x[pixel] - training_images[image][pixel]),2)
        distance = math.sqrt(sum)
        # result = np.vstack((result, np.array((image, distance))))
        result.append((image, distance))
    n = np.array(result, dtype=[('x', int), ('y', float)])
    # n = np.array(result)
    n.sort(order='y')

    nearest = []

    for i in range(k):
        nearest.append(training_images[n[i][0], 784])

    return n, nearest

r, p = NNClassify(test_images[89], 50)