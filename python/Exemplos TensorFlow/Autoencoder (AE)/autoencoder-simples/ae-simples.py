# TensorFlow: Deep Autoencoder (Deep AE) para Classificação Binária
# Disponível em: https://github.com/aymericdamien/TensorFlow-Examples/

# importando pacotes e módulos
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# Carregando o dataset MNIST
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parâmetros de treinamento da rede
learning_rate = 0.01
num_steps = 30000
batch_size = 256

# Variáveis auxiliares
display_step = 1000
examples_to_show = 10

# Parâmetros da rede neural (784 -> 256 -> 128 -> 256 -> 784)
num_hidden_1 = 256
num_hidden_2 = 128
num_input = 784

# Placeholder para valores não inicializados
X = tf.placeholder("float", [None, num_input])

# Pesos inicializados aleatoriamente
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}

# Biases inicializados aleatoriamente
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Encoder com duas camadas e funçãod e ativação Sigmoid
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Decoder com duas camadas e funçãod e ativação Sigmoid
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construindo o modelo (encoder + decoder)
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Matriz de predição (saída do Autoencoder)
y_pred = decoder_op
# Matriz de validação (a própria entrada)
y_true = X

# Função de perda: MSE
# reduce_mean -> média
# pow -> potenciação
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

# Otimizador: RMSProp
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Configurando a inicialização de variáveis
init = tf.global_variables_initializer()

# Iniciando a sessão
with tf.Session() as sess:

    # executando o inicializador
    sess.run(init)

    # Treinamento
    for i in range(1, num_steps+1):
        # Preparando os dados
        # Obtendo o próximo batch do MNIST
        # Apenas imagens são necessárias, não rótulos
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Executando a otimização (backpropagation) com a função de custo
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Apresentando relatório a cada determinada quantidade de passos
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Validação
    # Aplicando encoding e decoding no conjunto de validação
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # Dataset MNIST
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Mostrando as imagens originais
        for j in range(n):
            # Traçando os dígitos originais
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Mostrando as imagens reconstruídas
        for j in range(n):
            # Traçando os dígitos reconstruídos
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    # Gerando a imagems original
    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    # Gerando a imagem reconstruída
    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()