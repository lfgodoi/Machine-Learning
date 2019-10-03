# Keras: Deep Autoencoder (Deep AE) para Classificação Binária
# Disponível em: https://medium.com/datadriveninvestor/deep-autoencoder-using-keras-b77cd3e8be95

# Importando os pacotes necessários
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Carregando o dataset MNIST
# X_train: subconjunto de treinamento
# X_test: subconjunto de teste
(X_train, _), (X_test, _) = mnist.load_data()

# Normalizando os dados de 0 a 1
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Achatando as matrizes para vetores
# (6000, 28, 28) -> (6000, 784)
X_train = X_train.reshape(len(X_train),np.prod(X_train.shape[1:])) 
X_test = X_test.reshape(len(X_test),np.prod(X_test.shape[1:]))
print(X_train.shape)
print(X_test.shape)

# Convertendo a imagem de entrada de dimnesão 784 para tensores keras
input_img = Input(shape=(784,))
print(input_img)

# Montando o autoencoder
encoded = Dense(units=128, activation='relu')(input_img)
encoded = Dense(units=64, activation='relu')(encoded)
encoded = Dense(units=32, activation='relu')(encoded)
decoded = Dense(units=64, activation='relu')(encoded)
decoded = Dense(units=128, activation='relu')(decoded)
decoded = Dense(units=784, activation='sigmoid')(decoded)

# Criando o autoencoder
# Entrada: imagem
# Saída: última camada do decoder
autoencoder = Model(input_img, decoded)

# Capturando a saída do encoder (imagem comprimida)
encoder = Model(input_img, encoded)

# Visualizando a estrutura do autoencoder
autoencoder.summary()

# Compilando o modelo (configurando para treinamento)
autoencoder.compile(optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])

# Treinando o modelo
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

# Gerando predição para imagem comprimida e imagem reconstruída
encoded_imgs = encoder.predict(X_test)
predicted = autoencoder.predict(X_test)

# Plotando as imagens
plt.figure(figsize=(40,4))
for i in range(10):
    # Imagem original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Imagem comprimida
    ax = plt.subplot(3, 20, 20 + i + 1)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Imagem reconstruída
    ax = plt.subplot(3, 20, 20 * 2 + i + 1)
    plt.imshow(predicted[i].reshape(28, 28))    
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()























