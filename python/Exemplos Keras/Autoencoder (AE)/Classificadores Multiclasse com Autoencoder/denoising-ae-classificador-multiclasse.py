# Keras: Denoising Autoencoder (DAE) para Classificação Binária
# Disponí­vel em: https://medium.com/datadriveninvestor/deep-autoencoder-using-keras-b77cd3e8be95

# Importando os pacotes necessários
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Carregando dataset MNIST (subconjuntos: treinamento e teste)
(X_train, _), (X_test, _) = mnist.load_data()

# Normalizando os dados de 0 a 1
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# Achatando as matrizes em vetores (6000, 28, 28 -> 6000, 784)
X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))

# Criando uma versão com ruído dos dados de treinamento
X_train_noisy = X_train + np.random.normal(loc=0.0,
                                           scale=0.5,
                                           size=X_train.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)

# Criando um versão com ruído dos dados de teste
X_test_noisy = X_test + np.random.normal(loc=0.0,
                                         scale=0.5,
                                         size=X_test.shape)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# Mostrando o formato das versões com ruído
print(X_train_noisy.shape)
print(X_test_noisy.shape)

# Convertendo a imagem de entrada de dimnesão 784 para tensores keras
input_img = Input(shape=(784,))

# Montando o autoencoder
encoded = Dense(units=128, activation='relu')(input_img)
encoded = Dense(units=64, activation='relu')(encoded)
encoded = Dense(units=32, activation='relu')(encoded)
decoded = Dense(units=64, activation='relu')(encoded)
decoded = Dense(units=128, activation='relu')(decoded)
decoded = Dense(units=784, activation='sigmoid')(decoded)

# Construindo o modelo
autoencoder = Model(input_img, decoded)

# Extraindo o encoder
encoder = Model(input_img, encoded)

# Compilando o autoencoder
autoencoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# Treinando o modelo
autoencoder.fit(X_train_noisy, X_train_noisy,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test_noisy, X_test_noisy))

# Extraindo a imagem do encoder
encoded_imgs = encoder.predict(X_test_noisy)

# Extraindo a imagem reconstruída
predicted = autoencoder.predict(X_test_noisy)

# Plotando as imagens
plt.figure(figsize=(40,4))
for i in range(10):
    # Imagem original
    ax = plt.subplot(4, 20, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Imagem com ruído
    ax = plt.subplot(4, 20, 20 + i + 1)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Imagem comprimida
    ax = plt.subplot(4, 20, 20 * 2 + i + 1)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Imagem reconstruída
    ax = plt.subplot(4, 20, 20 * 3 + i + 1)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()    









