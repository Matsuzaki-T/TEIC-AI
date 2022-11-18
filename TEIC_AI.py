#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#TEIC_AI code


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def one_hot_encoding(x):
    unique = list(np.unique(x))
    for i in unique:
        x = np.where(x == i, unique.index(i), x)
    x = np.ravel(x).astype(int)
    one_hot = np.eye(len(np.unique(x)))[x]
    return unique, one_hot      
                                                                                                                                                                                                       
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) #overflow
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.res

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:input, t:output
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:input, t:output
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

def planning(Age,BW,BMI,Ccr,Alb,T0,T1,T2,T3,T4,T5,T6,T7):
    
    #expert data for normalization
    data_expert = pd.read_csv("./dataset/TEIC_data(expert_non-ICU)(TEIC-table3).csv")#expert data for normalization
    param_list=["Age","body weight","BMI","Creatinine clearance","Alb","TO"]
    parameter_model = np.array(data_expert[param_list]) #patameter1 in non-expert (numerical)
    loading_model = np.array(data_expert[["loading dose"]])#experts' loading
    maintenance_model = np.array(data_expert[["maintenance dose"]]) #experts' maintenance

    #transform loading to one-hot encoding
    ll, loading_one_hot = one_hot_encoding(loading_model)
    #transform maintenance to one-hot encoding
    lm, maintenance_one_hot = one_hot_encoding(maintenance_model)
    
    #patient data
    input1 = np.array([Age,BW,BMI,Ccr,Alb,T0])
    input1 = np.reshape(input1,[1,-1])
    input2 = np.array([T1,T2,T3,T4,T5,T6,T7])
    input2 = np.reshape(input2,[1,-1])
    
    #scaler
    sc = StandardScaler()
    sc.fit(parameter_model) #parameter in expert (numerical)
    input1_scaler = sc.transform(input1)
    input_scaler = np.concatenate([input1_scaler, input2], 1)
    #loading
    network_loading = TwoLayerNet(input_size=input_scaler.size, hidden_size=15, output_size=len(ll))
    #parameter incorporation
    for key in ('W1', 'b1', 'W2', 'b2'):
        network_loading.params[key] = np.load("./weight parameter/external validation (expertML)/loading.param({}_whole_non-ICU_lr_0.1).npy".format(key))
    pred_loading_one_hot_to_index = np.argmax(network_loading.predict(input_scaler),axis=1)[0]
    pred_loading_dose = ll[pred_loading_one_hot_to_index]
    
    #maintenance
    network_maintenance = TwoLayerNet(input_size=input_scaler.size, hidden_size=15, output_size=len(lm))
    
    #parameter incorporation
    for key in ('W1', 'b1', 'W2', 'b2'):
        network_maintenance.params[key] = np.load("./weight parameter/external validation (expertML)/maintenance.param({}_whole_non-ICU_lr_0.1).npy".format(key))
    
    pred_maintenance_one_hot_to_index = np.argmax(network_maintenance.predict(input_scaler),axis=1)[0]
    pred_maintenance_dose = lm[pred_maintenance_one_hot_to_index]
    print("AI-recommended dosing regimen is")
    print("loading dose: "+str(pred_loading_dose))
    print("maintenance dose: "+str(pred_maintenance_dose))
    return pred_loading_dose, pred_maintenance_dose

