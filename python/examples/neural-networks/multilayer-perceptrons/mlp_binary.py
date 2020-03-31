'''

TITLE: 
   Multilayer Perceptron for Binary Classification

DESCRIPTION:
   Classificating samples between two classes (0 or 1)
   using a multilayer perceptron model.
   
VERSION: 
   Author: Leonardo Godói (eng.leonardogodoi@gmail.com)
   Creation date: 19-September-2019

REVISION HISTORY:
   V1.0 | 19-September-2019 | Leonardo Godói | Creation

'''

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

# Importing packages                                         
from tensorflow.keras.models import Sequential                          
from tensorflow.keras.layers import Dense, Dropout 
from sklearn.datasets import make_blobs                        

# Gerating train and test data (2 classes)
X_train, y_train = make_blobs(n_samples=10000, centers=2, n_features=16, random_state=0)          
X_test, y_test = make_blobs(n_samples=5000, centers=2, n_features=16, random_state=0)

# Multilayer perceptron's architecture
mlp = Sequential()                                        
mlp.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))        
mlp.add(Dropout(0.5))                                     
mlp.add(Dense(4, activation='relu'))                     
mlp.add(Dropout(0.5))                                     
mlp.add(Dense(1, activation='sigmoid'))                 

# Compiling the model
mlp.compile(loss='binary_crossentropy',               
            optimizer='rmsprop',                           
            metrics=['accuracy'])                       

# Checking the model's architecture
mlp.summary()

# Training the model
mlp.fit(X_train, y_train,                                 
        epochs=20,                                 
        batch_size=32,
        validation_split=0.1)                                  

# Evaluating the model
_, accuracy = mlp.evaluate(X_test, y_test, batch_size=128)   
print('Accuracy: %.2f' % (accuracy*100) + '%') 

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------