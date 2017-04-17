# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:42:55 2017

@author: Energy
"""
import  numpy as np
import matplotlib.pyplot as plt
class NeuralNetwork(object):
    def __init__(self,input_nodes, hidden_nodes, output_nodes,learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning = learning_rate
        self.weights_input_to_hidden = np.random.normal(0.0,self.input_nodes**-0.5,(self.input_nodes,self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0,self.hidden_nodes**-0.5,(self.hidden_nodes,self.output_nodes))
        self.activation_function = lambda x:1/(1+np.exp(-x))
    def train(self,train_data,train_label):
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for x,y in zip(train_data,train_label):
            hidden_input = x*self.weights_input_to_hidden
            hidden_output = self.activation_function(hidden_input)
            final_input = np.dot(hidden_output,self.weights_hidden_to_output)
            final_output = final_input
            error = y-final_output
            hidden_error = error*self.weights_hidden_to_output
            output_error_term = error
            hidden_error_term = hidden_error.T*hidden_output*(1-hidden_output)
            delta_weights_i_h += self.learning*hidden_error_term*x
            delta_weights_h_o += self.learning*output_error_term*hidden_output.T
        self.weights_hidden_to_output +=delta_weights_h_o
        self.weights_input_to_hidden +=delta_weights_i_h
    def run(self,x):
        hidden_inputs = np.dot(x,self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)# signals from hidden layer

            # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs
        
            