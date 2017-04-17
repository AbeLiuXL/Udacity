# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import  numpy as np
import matplotlib.pyplot as plt
import NeuralNetwork
import sys
def MSE(y,Y):
    return np.mean((y-Y)**2)

np.random.seed(40)
x_train=np.random.random(600)*2*np.pi
y_train=np.sin(x_train)
plt.plot(x_train,y_train,'ro')
x_test = np.random.random(100)*2*np.pi
y_test = np.sin(x_test)
iterations = 2000
learning_rate = 0.03
input_nodes = 1
hidden_nodes = 5
output_nodes = 1
x_index =np.arange(x_train.shape[0])
network = NeuralNetwork.NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
losses = {'train':[],'test':[]}
for i in range(iterations):
    batch = np.random.choice(x_index,size=10,replace=False)
    x,y = x_train[batch],y_train[batch]
    network.train(x,y)
    train_loss = MSE(network.run(x_train[:,None]),y_train[:,None])
    test_loss = MSE(network.run(x_test[:,None]),y_test[:,None])
    sys.stdout.write("\rProgress:{:2.1f}".format(100*i/iterations)\
                     +"%...Train_loss:"+str(train_loss)[:5]\
                     +"...test_loss:"+str(test_loss)[:5])
    sys.stdout.flush()
    #x_index = np.delte(x_index,batch)
    
    losses['train'].append(train_loss)
    losses['test'].append(test_loss)
plt.figure(2)
plt.plot(losses['train'],label='train_loss')
plt.plot(losses['test'],label='test_loss')
plt.legend()
_ = plt.ylim

plt.figure(3)
plt.plot(x_test,network.run(x_test[:,None]),'o',label='y_run')
plt.plot(x_test,y_test,'o',label = 'y_test')
plt.legend()
_ = plt.ylim
