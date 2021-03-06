'''

TITLE: 
   Convolutional Autoencoder

DESCRIPTION:
   Feature extraction using an autoencoder composed of
   convolutional layers for processing 2-D inputs.
   
VERSION: 
   Author: Leonardo Godói (eng.leonardogodoi@gmail.com)
   Creation date: 17-October-2019

REVISION HISTORY:
   V1.0 | 17-October-2019 | Leonardo Godói | Creation

'''

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

# Importing packages
from tensorflow.keras.models import Model                                                     
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D   
from tensorflow.keras.datasets import mnist                                                       
from tensorflow.keras.utils import to_categorical           
import matplotlib.pyplot as plt                                                                       
 
# Loading train and test data from MNIST dataset 
(X_train, y_train), (X_test, y_test) = mnist.load_data()                             

# Normalizing images from grayscale to 0 to 1 range
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Reshaping images for processing 
X_train = X_train.reshape(60000, 28, 28, 1)                                                
X_test = X_test.reshape(10000, 28, 28, 1)                                                     
 
# Formatando os conjuntos de saí­da 
y_train = to_categorical(y_train)                                                       
y_test = to_categorical(y_test)                                                            
 
# Convolutional autoencoder model     
input_img = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))                                                                
enc_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(input_img)    
enc_2 = MaxPooling2D((2, 2))(enc_1)     
enc_3 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(enc_2)  
enc_4 = MaxPooling2D((2, 2))(enc_3)
enc_5 = Conv2D(1, kernel_size=3, activation='relu', padding='same')(enc_4) 
dec_1 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(enc_5) 
dec_2 = UpSampling2D((2, 2))(dec_1)    
dec_3 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(dec_2)
dec_4 = UpSampling2D((2, 2))(dec_3)                                   
output_img = Conv2D(1, kernel_size=3, activation='relu', padding='same')(dec_4)  

# Building the model
autoencoder = Model(input_img, output_img)
encoder = Model(input_img, enc_5)                                         

# Compiling the model                                                   
autoencoder.compile(optimizer='adam',                                                     
                    loss='mse')                                                 

# Checking the model's architecture
autoencoder.summary()

# Training the model 
autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=10)

# Generating prediction from encoder (compressed) and autoencoder (reconstructed)
compressed = encoder.predict(X_test)
reconstructed = autoencoder.predict(X_test)

# Plotting original, compressed and reconstructed images
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
    plt.imshow(compressed[i].reshape(7, 7))
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