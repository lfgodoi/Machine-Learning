# TensorFlow: Multilayer Perceptron (MLP) para Classificação Multiclasse
# Disponível em: https://adventuresinmachinelearning.com/python-tensorflow-tutorial/

# Importando a biblioteca do TensorFlow e o dataset MNIST
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Carregando o dataset
# O argumento one-hot especifica que ao invés dos rótulos estarem associados a
# cada imagem como dígitos (ex.: 4), é um vetor com um nó igual a 1 e todos os
# outros igual a 0 (ex.: 0 0 0 1 0 0). Isso facilita a alimentação da saída da
# rede neural
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Variáveis representando os parâmetros da rede neural
learning_rate = 0.5
epochs = 10
batch_size = 100

# Declarando os placeholders (variáveis reservadas não inicializadas)
# Entrada x - 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# Saída y - 10 dígitos
y = tf.placeholder(tf.float32, [None, 10])

# Declarando as variáveis representando os pesos com inicialização aleatória
# Pesos e biases conectando a camada de entrada e a camada oculta
# 300 nós na camada oculta
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# Pesos e bias conectando a camada oculta e a camada de saída
# 300 nós na camada oculta
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# Calculando a saída da camada oculta
# z = x * W1 + b
hidden_out = tf.add(tf.matmul(x, W1), b1)
# Aplicando a função de ativação sobre o resultado
hidden_out = tf.nn.relu(hidden_out)

# Calculando a camada de saída pela função de ativação Softmax
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

# Criando uma versão escalonada da saída y, com valores entre 1e-10 e 0.999999
# Isso evita que ocorra a operação log(0), que resultaria em NaN
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

# Função de custo: Cross-Entropy
# reduce_mean -> média
# log -> logaritmo natural
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

# Otimizador: Gradient Descent
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Configurando a inicialização de variáveis
init_op = tf.global_variables_initializer()

# Definindo uma operação de avaliação de precisão
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Iniciando a sessão
with tf.Session() as sess:
   # Exeecutando o inicializador
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   
   # Treinamento
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))