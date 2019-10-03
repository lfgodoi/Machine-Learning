# Keras: Autoencoder (AE) para Classificação Binária
# Disponível em: https://medium.com/datadriveninvestor/simple-autoencoders-using-keras-6e67677f5679

# Importando os pacotes necessários
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Carregando os dados de treinamento e teste do dataset
(X_train, _), (X_test, _) = mnist.load_data()

# Normalizando os dados de 0 a 1
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Achatando as matrizes de 28x28 para vetores de 784
X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))
print(X_train.shape)
print(X_test.shape)

# Convertendo a imagem de entrada em tensores keras
input_img = Input(shape=(784,))

# Encoder (784 -> 32)
# Entrada: Imagem
encoded = Dense(units=32, activation='relu')(input_img)

# Decoder (32 -> 784)
# Entrada: Encoder
decoded = Dense(units=784, activation='sigmoid')(encoded)

# Criando o modelo do autoencoder
# Entrada: Imagem
# Saída: Decoder
autoencoder = Model(input_img, decoded)

# Extraindo a imagem comprimida
encoder = Model(input_img,encoded)

# Verificando a estrutura do modelo
autoencoder.summary()

# Compilando o modelo
autoencoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Treinando o modelo
autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

# Gerando predições da imagem reconstruída e da imagem comprimida
encoded_imgs = encoder.predict(X_test)
predicted = autoencoder.predict(X_test)

# Plotando as imagens original, codificada e reconstruída
plt.figure(figsize=(40,4))
for i in range(10):
    # Original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Comprimida
    ax = plt.subplot(3, 20, i + 20)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Reconstruída
    ax = plt.subplot(3, 20, 2 * 20 + i + 1)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

