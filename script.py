

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

"""### **Data Loading and Cleaning of CIFAR-100**"""

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode="coarse")

train_labels

# Normalize all RGB values in our images dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshaping the arrays of labels to be single dimensional arrays
train_labels = train_labels.reshape(-1,)
test_labels = test_labels.reshape(-1,)

train_labels

# Array of all possible coarse(superclasses) and fine(subclasses) labels for our images in the CIFAR100 data
classes_coarse = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables', 'household_electrical_devices',
                  'household_furniture', 'insects', 'large_carnivores', 'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
                  'large_omnivores_and_herbivores', 'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
                  'trees', 'vehicles_1', 'vehicles_2']

# All superclasses for the images classified as living or nonliving things
classified_coarse = {"nonliving": [3,5,6,9,10,18,19],
                     "living": [0,1,2,4,7,8,11,12,13,14,15,16,17]}

new_classes = ["nonliving", "living"]

# Classification of superclasses from the CIFAR100 dataset into humans and non-humans
classified_dic = {"non-humans": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19],
                  "humans": [14]}

new_superclasses = ["non-humans", "humans"]

# Classigy all labels from their original coarse labels to a living or nonliving category
new_train_labels = []
new_test_labels = []

# We define a function to take in the original labels array and classify them into living or nonliving based on the classified_coarse dictionary
def re_classify(data, new):
  for Class in data:
    if Class in classified_coarse["living"]:
      new.append(1)
    else:
      new.append(0)

# We take the superclass train labels for all the images and classify them as living or nonliving for both train and test data
re_classify(train_labels, new_train_labels)
re_classify(test_labels, new_test_labels)

# Convert the new reclassified labels arrays to numpy arrays for easier calculations ahead
new_train_labels = np.array(new_train_labels)
new_test_labels = np.array(new_test_labels)

# Classifying all labels from their original coarse labels into either the human or non-human category
new_training_labels = []
new_testing_labels = []

# Writing a function to take in the original array of labels and classify them into human or non-human, based on the classified_dic dictionary
def new_classification(data, new):
  for Class in data:
    if Class in classified_dic["humans"]:
      new.append(1)
    else:
      new.append(0)

new_classification(train_labels, new_training_labels)
new_classification(test_labels, new_testing_labels)

# Converting the newly classified labels arrays to numpy arrays
new_training_labels = np.array(new_training_labels)
new_testing_labels = np.array(new_testing_labels)

# Function to plot images and label them with the correct corresponding labels in the 'y' parameter
def plot_images(images, binary_label, all_labels, image):
    plt.figure(figsize = (15,2))
    plt.imshow(images[image])
    bin_label = binary_label[image]
    actual_label = classes_coarse[all_labels[image]]
    if bin_label == 1:
      plt.xlabel("living - " + actual_label)
    else:
      plt.xlabel("non living - " + actual_label)

plot_images(test_images, new_test_labels, train_labels, 0)

print(new_train_labels)

# Writing a function to plot images and their corresponding labels
def to_plot(images, binary_label, all_labels, image):
    plt.figure(figsize = (15,2))
    plt.imshow(images[image])
    bin_label = binary_label[image]
    actual_label = classes_coarse[all_labels[image]]
    if bin_label == 1:
      plt.xlabel("human")
    else:
      plt.xlabel("non-human: " + actual_label)

"""### **Creating a Convolutional Neural Network for classification of Living v/s Non-living things**"""

# Create a Convolutional Neural Network to learn the labels of all different images in our dataset. 
# Here the Network has 2 convolutional and pooling layers, one hidden layer with 64 neurons and finally the
# output layer with 2 neurons that represent either class living or class nonliving
cnn = models.Sequential([
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = cnn.fit(train_images, new_train_labels, epochs=10)

plt.figure(figsize=[8,6])
plt.plot(history.history["loss"], linewidth=3)
plt.plot(history.history["accuracy"], linewidth=3)
plt.legend(["Training Loss", "Training Accuracy"], fontsize=18)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss/Accuracy %", fontsize=16)
plt.title("Loss vs Accuracy", fontsize=18)

# Evaluate the model on our array of new test labels which have only 0 and 1 values for living v/s nonliving
# instead of 0-19 values for all classes in the original data
cnn.evaluate(test_images, new_test_labels)

# Make predictions of labels on our testing dataset of 10000 images 
network_predictions = cnn.predict(test_images)

"""### **Testing**"""

print("We get an array of len:")
print(len(network_predictions))
print("")
print("Outputs of the network on our 10000 testing images dataset:")
print(network_predictions)
print("")
print("For every input to our network our network outputs a list of 2 values for example:")
print(network_predictions[0])

predicted_labels = []
for element in network_predictions:
  predicted_labels.append(np.argmax(element))
predicted_labels = np.array(predicted_labels)

# First 100 image's original labels
print(new_test_labels[:100])

# First 100 image's predicted labels
print(predicted_labels[:100])

# Plots of first 3 images with original labels
plot_images(test_images, new_test_labels, test_labels, 0)
plot_images(test_images, new_test_labels, test_labels, 1)
plot_images(test_images, new_test_labels, test_labels, 2)

# Plots of first 3 images with predicted labels
plot_images(test_images, predicted_labels, test_labels, 0)
plot_images(test_images, predicted_labels, test_labels, 1)
plot_images(test_images, predicted_labels, test_labels, 2)

print("predicted: " + new_classes[predicted_labels[0]] + ", actual: " + new_classes[new_test_labels[0]])
print("predicted: " + new_classes[predicted_labels[1]] + ", actual: " + new_classes[new_test_labels[1]])
print("predicted: " + new_classes[predicted_labels[2]] + ", actual: " + new_classes[new_test_labels[2]])
print("predicted: " + new_classes[predicted_labels[3]] + ", actual: " + new_classes[new_test_labels[3]])
print("predicted: " + new_classes[predicted_labels[4]] + ", actual: " + new_classes[new_test_labels[4]])
print("predicted: " + new_classes[predicted_labels[5]] + ", actual: " + new_classes[new_test_labels[5]])
print("predicted: " + new_classes[predicted_labels[6]] + ", actual: " + new_classes[new_test_labels[6]])
print("predicted: " + new_classes[predicted_labels[7]] + ", actual: " + new_classes[new_test_labels[7]])
print("predicted: " + new_classes[predicted_labels[8]] + ", actual: " + new_classes[new_test_labels[8]])
print("predicted: " + new_classes[predicted_labels[9]] + ", actual: " + new_classes[new_test_labels[9]])

"""### **Creating a Convolutional Neural Network to classify between Humans and Non-humans**"""

# Creating a Convolutional Neural Network that will learn the labels of all different images in our dataset. 
# Here the Network has 2 convolutional and pooling layers, one hidden layer with 128 neurons and finally the
# output layer with 2 neurons that represent either class living or class nonliving
cnn_humans = models.Sequential([
                                layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
                                layers.MaxPooling2D((2, 2)),
                                
                                layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
                                layers.MaxPooling2D((2, 2)),
                                
                                layers.Flatten(),
                                layers.Dense(64, activation='relu'),
                                layers.Dense(2, activation='softmax')
                                ])

cnn_humans.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

training_hist = cnn_humans.fit(train_images, new_training_labels, epochs=10)

plt.figure(figsize=[8,6])
plt.plot(training_hist.history["loss"], linewidth=3)
plt.plot(training_hist.history["accuracy"], linewidth=3)
plt.legend(["Training Loss", "Training Accuracy"], fontsize=18)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss & Accuracy %", fontsize=16)
plt.title("Training Loss & Accuracy Chart", fontsize=18)

# Evaluating the model on the array new_testing_labels which have only 0 and 1 values for humans or non-humans
# instead of 0-19 values for all classes in the original data
cnn_humans.evaluate(test_images, new_testing_labels)

# Make predictions of labels on our testing dataset of 10000 images 
model_prediction = cnn_humans.predict(test_images)

"""### **Testing the Human Classification CNN Model**"""

predictions = []
for element in model_prediction:
  predictions.append(np.argmax(element))
predictions = np.array(predictions)

# Plots of first 3 images with original labels
to_plot(test_images, new_testing_labels, test_labels, 6)
to_plot(test_images, new_testing_labels, test_labels, 88)
to_plot(test_images, new_testing_labels, test_labels, 65)

ii = np.where(new_testing_labels == 1)[0]
ii

# Plots of first 3 images with predicted labels
to_plot(test_images, predictions, test_labels, 6)
to_plot(test_images, predictions, test_labels, 88)
to_plot(test_images, predictions, test_labels, 65)

print("predicted: " + new_superclasses[predictions[0]] + ", actual: " + new_superclasses[new_testing_labels[0]])
print("predicted: " + new_superclasses[predictions[2]] + ", actual: " + new_superclasses[new_testing_labels[2]])
print("predicted: " + new_superclasses[predictions[3]] + ", actual: " + new_superclasses[new_testing_labels[3]])
print("predicted: " + new_superclasses[predictions[37]] + ", actual: " + new_superclasses[new_testing_labels[37]])
print("predicted: " + new_superclasses[predictions[65]] + ", actual: " + new_superclasses[new_testing_labels[65]])
print("predicted: " + new_superclasses[predictions[3]] + ", actual: " + new_superclasses[new_testing_labels[3]])
print("predicted: " + new_superclasses[predictions[5]] + ", actual: " + new_superclasses[new_testing_labels[5]])
print("predicted: " + new_superclasses[predictions[88]] + ", actual: " + new_superclasses[new_testing_labels[88]])
print("predicted: " + new_superclasses[predictions[6]] + ", actual: " + new_superclasses[new_testing_labels[6]])
print("predicted: " + new_superclasses[predictions[99]] + ", actual: " + new_superclasses[new_testing_labels[99]])