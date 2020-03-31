'''

TITLE: 
   Multilayer Perceptron for Multiclass Classification

DESCRIPTION:
   Classificating samples between multiple classes using
   a multilayer perceptron model.
   
VERSION: 
   Author: Leonardo Godói (eng.leonardogodoi@gmail.com)
   Creation date: 20-September-2019

REVISION HISTORY:
   V1.0 | 20-September-2019 | Leonardo Godói | Creation

'''

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

# Importing packages                                          
from tensorflow.keras.models import Sequential                          
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.datasets import make_blobs                     

# Gerating train and test data (10 classes)
X_train, y_train = make_blobs(n_samples=10000, centers=10, n_features=16, random_state=0)          
X_test, y_test = make_blobs(n_samples=5000, centers=10, n_features=16, random_state=0)

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Multilayer perceptron architecture
mlp = Sequential()                                                                      
mlp.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))                                 
mlp.add(Dropout(0.5))                                                                   
mlp.add(Dense(64, activation='relu'))                                                      
mlp.add(Dropout(0.5))                                                                     
mlp.add(Dense(y_train.shape[1], activation='softmax'))                                                  

# Setting the training algorithm
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)           

# Compiling the model                    
mlp.compile(loss='categorical_crossentropy',                                            
            optimizer=sgd,                                                                 
            metrics=['accuracy'])                                                          

# Checking the model's architecture
mlp.summary()

# Training the model
mlp.fit(X_train, y_train,                                                               
        epochs=20,                                                                       
        batch_size=128)

# Evaluating the model                                                               
_, accuracy = mlp.evaluate(X_test, y_test, batch_size=128)  
print('Accuracy: %.2f' % (accuracy*100) + '%')              

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------                       