# importing basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data('/home/rohitkumar/Downloads/ocr_dataset_mnist/train-images-idx3-ubyte/data')

plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# reshape X_train and X_test
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# feature scaling
X_train = X_train / 255
X_test = X_test / 255


# categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = "all")
y_train = onehotencoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = onehotencoder.fit_transform(y_test.reshape(-1,1)).toarray()
num_classes = y_train.shape[1]
print(num_classes)

# importing all keras packages and libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initialize CNN
classifier = Sequential()

# step-1 convolution
classifier.add(Convolution2D(32, 5, 5, input_shape=(28, 28, 1), activation='relu'))

# step-2 pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# step-3 flattening
classifier.add(Flatten())

# step-4 full connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=10, activation='sigmoid'))

# compiling CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fitting CNN to Dataset
# classifier.fit_generator(
#         X_train, y_train,
#         steps_per_epoch=500,
#         epochs=10,
#         validation_data=(X_test, y_test),
#         validation_steps=20, verbose=2)
classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = classifier.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# save model
from keras.models import load_model
classifier.save('/media/rohitkumar/Rohit-Sonu/python3/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/ocr.h5')  # creates a HDF5 file 'my_model.h5'

classifier = load_model('/media/rohitkumar/Rohit-Sonu/python3/projects/digit-OCR/ocr.h5')
# Recognize images
from PIL import Image
img = Image.open('/media/rohitkumar/Rohit-Sonu/python3/projects/digit-OCR/testSet/3.png')
img = img.convert('1')
# print(img)

img = img.resize((28, 28), 0)
x = np.asarray(img)
print(x.shape)
plt.imshow(x, cmap=plt.get_cmap('gray'))
plt.show()

x = x[np.newaxis, ..., np.newaxis]
print(x.shape)

y_pred = classifier.predict(x)
# print(y_pred.index(max(y_pred)))
# y_pred = np.asarray(y_pred)
print(y_pred.argmax())

