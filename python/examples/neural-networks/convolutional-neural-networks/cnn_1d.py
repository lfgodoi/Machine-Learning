'''

TITLE: 
   1-D Convolutional Neural Network

DESCRIPTION:
   Multiclass classification with a 1-D convolutional neural 
   network model.
   
VERSION: 
   Author: Leonardo Godói (eng.leonardogodoi@gmail.com)
   Creation date: 07-October-2019

REVISION HISTORY:
   V1.0 | 07-October-2019 | Leonardo Godói | Creation

'''

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

# Importing packages
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import RMSprop

# Embedding parameters
max_features = 10000
max_len = 500
print('Loading data...')

# Loading IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')

# Adjusting data for 1-D processing
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# 1-D CNN model
cnn = Sequential()
cnn.add(Embedding(max_features, 128, input_length=max_len))
cnn.add(Conv1D(32, 7, activation='relu'))
cnn.add(MaxPooling1D(5))
cnn.add(Conv1D(32, 7, activation='relu'))
cnn.add(GlobalMaxPooling1D())
cnn.add(Dense(1))

# Checking model's architecture
cnn.summary()

# Compiling the model
cnn.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

# Training the model
history = cnn.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------








