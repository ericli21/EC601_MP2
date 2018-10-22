import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os 
import csv

print('\n')
#Pull dataset and separate in training, validation, and test sets
train_images = []
train_labels = []
val_images = []
val_labels = []
test_images = []
test_labels = []

all_imagefiles = []
all_labels = []

for labelname in sorted(glob.glob('./labels/*')):
	f = open(labelname, 'r')
	lines = f.readlines()
	tokens = lines[0].split('\t')
	if tokens[0] == 'stop':
		all_labels.append(0)
	else:
		all_labels.append(1)
for imagename in sorted(glob.glob('./images/*')):
	all_imagefiles.append(imagename)
print("Done")



def parsing(imagefile, label):
	pre_image1 = tf.read_file(imagefile)
	pre_image2 = tf.image.decode_jpeg(pre_image1, channels=3)
	image = tf.cast(pre_image2, tf.float32)
	return image, label

dataset = tf.data.Dataset.from_tensor_slices((all_imagefiles, all_labels))
dataset = dataset.map(parsing)
dataset = dataset.batch(8)

iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()
with tf.Session() as sess:
	value = sess.run(images)
	print(value)

print(dataset.shape)
'''

all_images = np.array(all_images)
all_images = all_images / 255.0

#Preprocessing data
class_names = ['stop', 'do not enter']

train_images = all_images[:50]
train_labels = all_labels[:50]

#Build model
(x, y, z) = train_images[0].shape

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(x,y,3)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)
#Train model
model.fit(train_images, train_labels, epochs=5)


#Use validation images for accuracy
val_loss, val_acc = model.evaluate(val_images, val_labels)

#Use test images for accuracy
test_results = model.evaluate(test_images, test_labels)

'''