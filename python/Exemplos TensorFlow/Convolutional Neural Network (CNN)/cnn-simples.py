# TensorFlow: 2-D Convolutional Neural Network (2-D CNN) para Classificação Multiclasse
# Disponível em: https://github.com/aymericdamien/TensorFlow-Examples/

# Importando pacotes e módulos
from __future__ import division, print_function, absolute_import
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Carregando o dataset MNIST
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Parâmetros de treinamento
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Parâmetros da rede neural
num_input = 784
num_classes = 10
dropout = 0.25

# Construindo a rede neural
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    
    # Definindo um escopo para reutilizar as variáveis
    with tf.variable_scope('ConvNet', reuse=reuse):
    
        # Entrada do estimador é um dicionário, suportando múltiplas entradas
        x = x_dict['images']

        # MNIST consiste de um vetor 1D com 784 features (28 x 28 pixels)
        # Redimensionando para o formato de imagem (Altura x largura x Canal)
        # Entrada do tensor é 4D: [Batch Size, Altura, largura, Canal]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Camada de convolução com 32 filtros um kernel de tamanho 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Camada de max pooling com 2 passos de 2 e kernel de tamanho 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Camada de convolução com 64 filtros e um kernel de tamanho 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Camada de max pooling com 2 passos de 2 e kernel de tamanho 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Achatando os dados em um vetor 1D para as camadas fully connected
        fc1 = tf.contrib.layers.flatten(conv2)

        # Camada fully connected
        fc1 = tf.layers.dense(fc1, 1024)
        # Aplicando Dropout (Se is_training é falso, não é aplicado)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Camada de saída - Predição de classe
        out = tf.layers.dense(fc1, n_classes)

    return out

# Definindo o modelo
def model_fn(features, labels, mode):
    # Construindo a rede neural
    # Duas versões diferentes são criadas, utilizando os mesmos pesos, porque 
    # o Dropout apresenta diferentes comportamentos no treinamento e na 
    # predição
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predições
    pred_classes = tf.argmax(logits_test, axis=1)

    # Se estiver no modo predição, retornar com antecedência
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Definindo função de perda e otimizador
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                             logits=logits_train, 
                             labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Avaliando a precisão do modelo
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # Estimadores TF requerem o retorno de um EstimatorSpec, que especifica
    # as diferentes operações para treinamento, avaliação, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Construindo o estimador
model = tf.estimator.Estimator(model_fn)

# Definindo a função de entrada para o treinamento
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Treinando o modelo
model.train(input_fn, steps=num_steps)

# Avaliando o modelo
# Definindo a função de entrada para a avaliação
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Usando o método de avaliação do estimador
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])