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
dec_1 = UpSampling2D((2, 2))(enc_4)    
dec_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(dec_1)
dec_3 = UpSampling2D((2, 2))(dec_2)                                   
output_img = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(dec_3)  

# Building the model
autoencoder = Model(input_img, output_img)
encoder = Model(input_img, enc_4)                                         

# Compiling the model                                                   
autoencoder.compile(optimizer='adam',                                                     
                    loss='mse')                                                 

# Checking the model's architecture
autoencoder.summary()

# Training the model 
autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=3)

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------