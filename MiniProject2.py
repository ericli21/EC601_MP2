import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os 

#Define lists for training, validation, and testing
train_images = []
train_labels = []
val_images = []
val_labels = []
test_images = []
test_labels = []

#Define lists to store image file names and labels
all_images = []
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
	all_images.append(imagename)

#Define function to parse the image file names to images (resize) for dataset conversion
def parsing(imagefile, label):
	pre_image1 = tf.read_file(imagefile)
	pre_image2 = tf.image.decode_jpeg(pre_image1, channels=3)
	pre_image3 = tf.image.resize_images(pre_image2, [378, 504])
	image = tf.cast(pre_image3, tf.float32)
	return image, label

#Convert images and labels into 1 big dataset
dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
dataset = dataset.map(parsing)

#Create a training batch of 60/79 images, and a "other" batch of 19/79 images
dataset = dataset.batch(60)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
print(next_element[1])

#Define 2 models for comparison
class_names = ['stop', 'do not enter']
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(378,504,3)),
	keras.layers.Dense(130, activation=tf.nn.relu),
	keras.layers.Dense(13, activation=tf.nn.softmax),
	keras.layers.Dense(100, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax),
	keras.layers.Dense(50, activation=tf.nn.relu)
	])
model.compile(optimizer=tf.train.AdamOptimizer(),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
	)
model_2 = keras.Sequential([
	keras.layers.Flatten(input_shape=(378,504,3)),
	keras.layers.Dense(300, activation=tf.nn.softmax),
	keras.layers.Dense(30, activation=tf.nn.relu)
	])
model_2.compile(optimizer=tf.train.AdamOptimizer(),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
	)

#Create session to pull batches and run
with tf.Session() as sess:
	total = sess.run(next_element)
	#Separate training images and labels
	train_images = total[0]
	train_labels = total[1]
	#Preprocess the images
	train_images /= 255.0
	#Train model
	model.fit(train_images, train_labels, epochs=30)
	model_2.fit(train_images, train_labels, epochs=30)
	total_2 = sess.run(next_element)
	#Get the "other" images and labels
	other_images = total_2[0]
	other_labels = total_2[1]
	#Separate into a validation (18 images) and test set (1 image)
	val_images = other_images[:-1]
	val_labels = other_labels[:-1]
	test_images = other_images[-1:]
	test_labels = other_labels[-1:]
	#Evaluate using validation set
	val_loss, val_acc = model.evaluate(val_images, val_labels)
	val_loss_2, val_acc_2 = model_2.evaluate(val_images, val_labels)
	print('Val loss: ', val_loss, 'Val acc:', val_acc)
	print('Val loss 2: ', val_loss_2, 'Val acc 2:', val_acc_2)
	#Predict last test image
	prediction = model.predict(test_images)
	prediction_2 = model_2.predict(test_images)
	analyze = (np.argmax(prediction[0]) == test_labels[0])
	analyze_2 = (np.argmax(prediction_2[0]) == test_labels[0])
	print(analyze)
	print(analyze_2)





#Old code (wondering if anything can work using this way):
'''	
for labelname in sorted(glob.glob('./labels/*')):
	f = open(labelname, 'r')
	lines = f.readlines()
	tokens = lines[0].split('\t')
	if tokens[0] == 'stop':
		all_labels.append(0)
	else:
		all_labels.append(1)
for imagename in sorted(glob.glob('./images/*')):
	pre_image1 = cv2.imread(imagename)
	pre_image2 = cv2.resize(pre_image1, dsize=(378,504), interpolation=cv2.INTER_NEAREST)
	image = np.asarray(pre_image1, np.float32)
	all_images.append(image)

val_split = 0.2
split_index = int((1 - val_split) * len(all_images))
train_images = all_images[:split_index]
train_labels = all_labels[:split_index]
val_images = all_images[split_index:]
val_labels = all_labels[:split_index]

train_images = np.array(train_images)

print("\n")
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print("\n")

#all_images = np.array(all_images)
#all_images = all_images / 255.0

#Preprocessing data
class_names = ['stop', 'do not enter']



#Build model

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(378,504)),
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