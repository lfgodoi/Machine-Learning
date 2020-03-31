'''

TITLE: 
   2-D Convolutional Neural Network

DESCRIPTION:
   Multiclass classification with a 2-D convolutional neural 
   network model.
   
VERSION: 
   Author: Leonardo Godói (eng.leonardogodoi@gmail.com)
   Creation date: 04-October-2019

REVISION HISTORY:
   V1.0 | 04-October-2019 | Leonardo Godói | Creation

'''

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

# Importing packages
from tensorflow.keras.models import Sequential                                                     
from tensorflow.keras.layers import Dense, Conv2D, Flatten  
from tensorflow.keras.datasets import mnist                                                       
from tensorflow.keras.utils import to_categorical                                                                                           
 
# Loading train and test data from MNIST dataset 
(X_train, y_train), (X_test, y_test) = mnist.load_data()                             

# Reshaping images for processing 
X_train = X_train.reshape(60000, 28, 28, 1)                                                
X_test = X_test.reshape(10000, 28, 28, 1)                                                     
 
# Formatando os conjuntos de saí­da 
y_train = to_categorical(y_train)                                                       
y_test = to_categorical(y_test)                                                            
 
# 2-D CNN architecture
cnn = Sequential()                                                                      
cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))         
cnn.add(Conv2D(32, kernel_size=3, activation='relu'))                                   
cnn.add(Flatten())                                                                  
cnn.add(Dense(y_train.shape[1], activation='softmax'))                                              

# Compiling the model                                                   
cnn.compile(optimizer='adam',                                                     
            loss='categorical_crossentropy',                                             
			metrics=['accuracy'])                                                 

# Training the model 
cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------