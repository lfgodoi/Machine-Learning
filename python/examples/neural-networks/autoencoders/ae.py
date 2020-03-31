'''

TITLE: 
   Autoencoder

DESCRIPTION:
   Feature extraction using a vanilla autoencoder model
   built with TensorFlow and Keras.
   
VERSION: 
   Author: Leonardo Godói (eng.leonardogodoi@gmail.com)
   Creation date: 29-September-2019

REVISION HISTORY:
   V1.0 | 29-September-2019 | Leonardo Godói | Creation

'''

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

# Importing packages
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
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

# Autoencoder model (784 - 32 -784)
input_img = Input(shape=(X_train.shape[1],))
encoded = Dense(units=32, activation='relu')(input_img)
decoded = Dense(units=784, activation='sigmoid')(encoded)

# Building the model
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# Checking model's architecture
autoencoder.summary()

# Compiling the model
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Training the model
autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

# Generating prediction from encoder (compressed) and autoencoder (reconstructed)
compressed = encoder.predict(X_test)
reconstructed = autoencoder.predict(X_test)

# Plotting original, compressed and reconstructed images
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
    plt.imshow(compressed[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Reconstructed
    ax = plt.subplot(3, 20, 2 * 20 + i + 1)
    plt.imshow(reconstructed[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

