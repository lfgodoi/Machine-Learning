'''

TITLE: 
   Sparse Autoencoder

DESCRIPTION:
   Feature extraction using different versions of sparse 
   autoencoder model built with TensorFlow and Keras.
      - Version 1: L1/L2 regularizers
      - Version 2: Dropout regularizers
   
VERSION: 
   Author: Leonardo Godói (eng.leonardogodoi@gmail.com)
   Creation date: 30-September-2019

REVISION HISTORY:
   V1.0 | 30-September-2019 | Leonardo Godói | Creation

'''

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

# Importing packages
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
import matplotlib.pyplot as plt
import numpy as np

# Loading train and test data from MNIST dataset
(X_train, _), (X_test, _) = mnist.load_data()

# Normalizing images from grayscale to 0 to 1 range
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Flattening images (28x28 -> 784)
X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

# Sparse autoencoder architecture

# Common input
input_img = Input(shape=(X_train.shape[1],))

# Version 1 - Regularized by L1 and L2
encoded_v1 = Dense(units=32, activation='relu', activity_regularizer=l1(0.01), kernel_regularizer=l2(0.01))(input_img)
decoded_v1 = Dense(units=784, activation='sigmoid')(encoded_v1)

# Version 2 - Regularized by dropout
encoded_v2 = Dense(units=32, activation='relu')(input_img)
dp_v2 = Dropout(0.5)(encoded_v2)
decoded_v2 = Dense(units=784, activation='sigmoid')(dp_v2)

# Building the models
encoder_v1 = Model(input_img, encoded_v1)
autoencoder_v1 = Model(input_img, decoded_v1)
encoder_v2 = Model(input_img, encoded_v2)
autoencoder_v2 = Model(input_img, decoded_v2)

# Checking models' architecture
autoencoder_v1.summary()
autoencoder_v2.summary()

# Compiling the models
autoencoder_v1.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder_v2.compile(optimizer='adam', loss='binary_crossentropy')

# Training the models
autoencoder_v1.fit(X_train, X_train,
                   epochs=10,
                   batch_size=256,
                   shuffle=True,
                   validation_data=(X_test, X_test))
autoencoder_v2.fit(X_train, X_train,
                   epochs=10,
                   batch_size=256,
                   shuffle=True,
                   validation_data=(X_test, X_test))

# Generating prediction from encoder (compressed) and autoencoder (reconstructed)
compressed_v1 = encoder_v1.predict(X_test)
reconstructed_v1 = autoencoder_v1.predict(X_test)
compressed_v2 = encoder_v2.predict(X_test)
reconstructed_v2 = autoencoder_v2.predict(X_test)

# Plotting original, compressed and reconstructed images for version 1
print('\nSparse autoencoder regularized by L1/L2\n')
plt.figure(figsize=(40, 4))
for i in range(10):
    # Original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Compressed
    ax = plt.subplot(3, 20, i + 20)
    plt.imshow(compressed_v1[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Reconstructed
    ax = plt.subplot(3, 20, 2 * 20 + i + 1)
    plt.imshow(reconstructed_v1[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Plotting original, compressed and reconstructed images for version 2
print('Sparse autoencoder regularized by dropout\n')
plt.figure(figsize=(40,4))
for i in range(10):
    # Original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Compressed
    ax = plt.subplot(3, 20, i + 20)
    plt.imshow(compressed_v2[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Reconstructed
    ax = plt.subplot(3, 20, 2 * 20 + i + 1)
    plt.imshow(reconstructed_v2[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------    