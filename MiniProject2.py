import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import re

print('\n')
#Pull dataset and separate in training, validation, and test sets
train_filenames = []
train_images = []
train_labels = []
val_filenames = []
val_images = []
val_labels = []
test_filenames = []
test_images = []
test_labels = []

temp_images = []
temp_labels = []

for foldername in glob.glob('./export/*'):
	tempname = foldername.split("\\")
	set_type = tempname[1]
	for setname in glob.glob(foldername + '/*'):
		tempname_2 = setname.split("\\")
		data_type = tempname_2[2]
		for filename in glob.glob(setname + '/*'):
			if data_type == 'labels':
				f = open(filename, 'r')
				lines = f.readlines()
				label = ''
				tokens = lines[0].split('\t')
				label = label + tokens[0]
				temp_labels.append(label)
			else:
				image = Image.open(filename)
				temp_images.append(image)
	if set_type == 'test':
		test_labels = temp_labels
		test_images = temp_images
	if set_type == 'train':
		train_labels = temp_labels
		train_images = temp_images
	if set_type == 'validation':
		val_labels = temp_labels
		val_images = temp_images
	temp_images = []
	temp_labels = []

#Print image and label info
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
print(train_labels[0])
#Preprocessing data
for counter, image in enumerate(train_images):
	train_images[counter] = np.array(image) / 255.0
for counter, image in enumerate(val_images):
	val_images[counter] = np.array(image) / 255.0
for counter, image in enumerate(test_images):
	test_images[counter] = np.array(image) / 255.0
class_names = ['stop', 'do not enter']
#Build model

(x, y, z) = train_images[0].shape


model = keras.Sequential([
	keras.layers.Flatten(input_shape=(x,y)),
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

