# Geração de espectrograma 3-D

# Importando os pacotes e módulos
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy import signal
import matplotlib.pyplot as plt

# Função para gerar representação 3-D de espectrograma
def genSpectrogram(filename, samples, sensor):
    dataset = pd.read_csv(filename, sep='\t', header=-1)
    dataset = dataset[:samples]
    dataset = dataset.iloc[:, :-1]
    for col in range(dataset.shape[1]):
        dataset.rename(columns={col:('Sensor ', str(col))}, inplace=True)
    fs = 10e3
    f, t, Sxx = signal.spectrogram(dataset.iloc[:,sensor], fs)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    tf, ft = np.meshgrid(t, f)
    ax.plot_surface(ft, tf, Sxx,cmap=plt.cm.coolwarm)
    # ax.set_xlabel('Frequência')
    # ax.set_ylabel('Tempo')
    # ax.set_zlabel('Amplitude')

# Gerando os espectrogramas
genSpectrogram('health_30_2.csv', 600000, 5)
genSpectrogram('ball_30_2.csv', 600000, 5)
genSpectrogram('comb_30_2.csv', 600000, 5)
genSpectrogram('inner_30_2.csv', 600000, 5)
genSpectrogram('outer_30_2.csv', 600000, 5)