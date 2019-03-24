# EC601 Mini Project 2: Tensorflow CNN model
This is Python script for the Mini Project 2. I decided to take pictures of "stop" signs and "do not enter" signs, and use Tensorflow to classify each image with these signs. Though it is mostly functional, there still could be improvement on the code and data. 

## Data
The HEIC files from the iPhone was converted into JPEG files, and fed into Neurala's Brainbuilder system. Each image was manually tagged with a class, and exported as txt files and images. Nuerala asked to separate the data into a training, validation, and test set, but it was easier to pool all the data and labels to one so that tensorflow can parse it faster. The image data and labels were then converted into a dataset, and batches were used to separate the data: 60 for training, 18 for validation, and 1 for test predictions.

## Model
Two models were made using Keras Dense layers. The first model featured multiple small RELU layers with soft max layers in between them. The other model featured a big RELU layer with one max layer following it. Based on loss and accuracy values from training and validation, the latter model was more reliable. Other methods considered were loading a ResNet model pre-trained with ImageNet data, as well as using Keras Conv2D layers with max pooling layers. Unfortunately the dataset ran into errors with this (dimensionality issues, incorrect dense layer input, bad stride size, etc.); see suppressed code for more.

## Improvement
Unfortunately, both models were wrong on the test prediction. This may be due to a few reasons. First, the numbers of epochs was set to 30 due to time and memory constraints. 30 is still too low and makes the model volatile for error. Second, the data is heavily "do not enter" signs, giving the model less understanding of a "stop sign". Finally, there may be a bug in the code (please let me know!). 

## Sources
1. https://www.tensorflow.org/guide/datasets
2. https://www.tensorflow.org/tutorials/keras/basic_classification
3. https://keras.io/applications/
4. https://arxiv.org/ftp/arxiv/papers/1801/1801.00631.pdf
