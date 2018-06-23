
# coding: utf-8

# Import dos módulos necessários e declaração de constantes
# ---

# In[1]:


import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn import metrics
from sklearn.model_selection import train_test_split

# quantidade de canais que as imagens de entrada possuem
# escalar cor de cinza, entao 1
CHANNEL = 1
# valor da largura e da altura das imagens de entrada
WIDTH_HEIGHT = 20
# tamanho do batch
BATCH = 128


# Load do arquivo de dados
# ---

# In[2]:


load = np.loadtxt('exdata.csv', delimiter=',')

# cada coluna tem um padrao de digito
data = load[:-1].T

# a ultima linha eh a classificacao do digito
result = load[-1] 
# digito 0 corresponde ao valor 10
result[result == 10] = 0


# Pré-processamento dos dados e das classes
# ---

# In[3]:


# trasformar cada linha (digito) em uma matriz 20 x 20 x 1
data = data.reshape(data.shape[0], WIDTH_HEIGHT, WIDTH_HEIGHT, CHANNEL)

# converte array de 1 dimensao para uma matriz de dimensao 10
# ou seja, criar 10 classes, uma para cada digito possivel
result = keras.utils.to_categorical(result, 10)


# Separação dos dados em treinamento e teste
# ---

# In[4]:


in_train, in_test, out_train, out_test = train_test_split(data, 
                                                          result,
                                                          test_size=(25/100),
                                                          train_size=(75/100))


# Definição da arquitetura da rede neural
# ---

# In[5]:


# 'pilha' de camadas lineares
model = Sequential()

# primeira camada precisa saber o que espera de entrada
# Conv2D cria uma camada de 'convolution' (add cada elemento da imagem com o seu vizinho local)
# isso eh feito atraves do input_shape
# 32 eh o numero de filtros
# relu = rectified linear unit
model.add(Conv2D(32, 
                 kernel_size=(3, 3), 
                 activation='relu', 
                 input_shape=(WIDTH_HEIGHT, WIDTH_HEIGHT, CHANNEL)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# MaxPooling2D cria uma camada que faz um processo de discretizacao baseada em amostra
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout cria uma camada de regularizacao
# 0.25 eh a fracao da quantidade de entrada que entrara na camada
model.add(Dropout(0.25))

# Flatten cria uma camada que 'flatteniza'
model.add(Flatten())

# Dense cria uma camada que representa uma multiplicacao de matrizes
# 128 eh a dimensionalidade da saida
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# Compilação da rede neural
# ---

# In[6]:


# configuracao de que como sera o aprendizado de processo
# para qualquer problema de classificacao deve-se usar o accuracy
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# Input dos dados de treinamento
# ---

# In[7]:


# batch_size eh o numero de amostras por update do gradiente
# epochs = um epoch eh uma iteracao sobre os dados fornecidos
model.fit(in_train, out_train,
          batch_size=BATCH,
          epochs=10,
          verbose=1,
          validation_data=(in_test, out_test))


# Avaliação com os dados de teste
# ---

# In[8]:


# valores do modelo no modo de teste 
score = model.evaluate(in_test, out_test, verbose=0)
print('Teste loss:', score[0])
print('Teste acurácia:', score[1])


# Matriz de confusão
# ---

# In[9]:


# gera a predicao para o conjunto de teste
prediction = model.predict(in_test, batch_size=BATCH, verbose=0)

# ajuste dado para ter info correta
prediction_classes = np.argmax(prediction, axis=1)
out_test_classes = np.argmax(out_test, axis=1)

# gera a 'confusion matrix'
matrix = metrics.confusion_matrix(out_test_classes , prediction_classes)
print(matrix)


# In[10]:


# gera um relatorio com as principais metricas da classificacao
report = metrics.classification_report(out_test_classes, prediction_classes)
print(report)


# Precisão final
# ---

# In[11]:


# calcula a precisao da classificacao
value = metrics.accuracy_score(out_test_classes, prediction_classes)
print("Precisão no conjunto de teste: {:.2%}".format(value))

