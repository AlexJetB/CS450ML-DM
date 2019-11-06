"""
author: Alex Baker @alexjetb
program: Neural network nodes.
Description: Part 02 of 03 for the implementation of a neural network
             as set in Weeks 06-08 of CS450
"""
import numpy as np
#import pandas as pd
import math

class Node_Network:
    nodeArray = [] # empty list
    num_layers = 0
    num_inputs = 0
    num_nodes = 0
    num_outputs = 0
    maxHeight = 0
    learning_rate = 0.0
    layerPattern = np.empty(1)
    # A jagged matrix is created with a specified number of layers, and desired height
    # (length/columns) and a specified number of inputs (height/rows)
    def __init__(self,num_inputs=2,num_layers=1,learning_rate=0.1,max_height=2,
                 num_outputs=2):
#        print("working")
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.max_height = max_height
        self.num_outputs = num_outputs
        #Generate our random weights...
        weights = np.random.uniform(-1,0,(max_height,max_height,max_height))
        if max_height<1:
            print("Must have max_height=>1!")
            return
        else:
            self.layerPattern = np.random.randint(1, max_height, num_layers)
#        print(weights)
        # Construct empty jagged array
        self.nodeArray.append([dict()]*(num_inputs+1))
        with (np.nditer(self.layerPattern, flags=['c_index', 'multi_index'])) as it:
            for layer in it:
                blank_hidden = [dict()]*(layer.item()+1) # Account for bias node
                self.nodeArray.append(blank_hidden)
        self.nodeArray.append([dict()]*num_outputs)
        # Build each layer and initialize each node, including biases
        i_ind = 0
        totalLayers = len(self.nodeArray)-1
        for layer in self.nodeArray:
            j_ind=0
#            width=len(layer)-1
#            print(width)
            for node in layer:
                if i_ind==totalLayers:
                    outputs=0
                    weight = []
                if i_ind==totalLayers-1:
                    outputs =[0]*num_outputs
                    weight = weights[i_ind][j_ind][:num_outputs]
                if i_ind<totalLayers-1:
                    outputs=[0]*self.layerPattern[i_ind]
                    weight = list(weights[i_ind][j_ind][:self.layerPattern[i_ind]])

                if j_ind==0 and i_ind!=totalLayers:
                    self.nodeArray[i_ind][j_ind] = {'bias':True,
                                                    'input':-1,
                                                    'weights':weight,
                                                    'outputs':outputs,
                                                    'error':None}
                else:
                    self.nodeArray[i_ind][j_ind] = {'bias':False,
                                                    'input':0,
                                                    'weights':weight,
                                                    'outputs':outputs,
                                                    'error':0}
                j_ind+=1
            i_ind+=1

    # "Fitting" function for neural network
    def train(self,num_times=1,dataTrain=np.array([]),targetTrain=np.array([])):
        # Data and targets must be numpy arrays
        while num_times>0:
            i_ind=0
            for instance in dataTrain:
                targets = targetTrain[i_ind]
                result,output = self.predictOne(instance,targets)
                if result==False:
                    self.back_propagate(targets)
                    i_ind+=1
                    print("Training!")
                num_times-=1

    def back_propagate(self,targets):
        # Reverse list to backprop
        width=len(self.nodeArray)-1
        i_ind=width
        for layer in reversed(self.nodeArray):
            j_ind=0
            for node in layer:
                if i_ind==width: #Output error calculation
                    output=node['outputs']
                    target=targets[j_ind]
                    error = output*(1-output)*-(output-target)
                    node['error'] = error
                if i_ind<width:
                    output=node['outputs']
                    n_input=node['input']
                    weights=node['weights']
                    sum_of_wts=0
                    k_ind=0
                    for weight in weights:
                        i_err=self.nodeArray[i_ind+1][k_ind]['error']
                        sum_of_wts+=weight*i_err
                        k_ind+=1
                    node['error']=n_input*(1-n_input)*sum_of_wts
                j_ind+=1
            i_ind-=1
        print("BACK PROP!")

    # Predict a row
    # Weights are initialized separate from list, no need to indicate training
    def predictOne(self, dataTest=np.array([]), targetTest=np.array([])):
        if dataTest.size!=self.num_inputs:
            print('dataTest num of inputs must equal num_inputs!')
            return
        if dataTest.size==0:
            print('No data was given! Must give data to predict.')
            return
        if targetTest.size==0:
            print('No targets were given! Accuracy will not be predicted')

        # Initialize inputsnet
        j_ind=0
        for in_node in self.nodeArray[0]:
            if in_node['bias']!=True:
                in_node['input']=dataTest[j_ind]
                j_ind+=1

        column=0
        totalLayers = len(self.nodeArray)-1
        for layer in self.nodeArray:
            row=0
            for node in layer:
                node_wt = node['weights']
                if column==0 or node['bias']==True:
                    node['outputs']=[x * node['input'] for x in node_wt]
                if column>0 and node['bias']==False:
                    n_input=0
                    for pre_lay_node in self.nodeArray[column-1]:
                        n_input += pre_lay_node['outputs'][row-1]
                    n_input = 1 / (1 + math.exp(-n_input))
                    node['input']=n_input
                    node['outputs']=[x * node['input'] for x in node_wt]
                if column==totalLayers: #single output for output layer
                    node['outputs'] = node['input']

                row+=1
            column+=1

        outputs = []
        for node_out in self.nodeArray[totalLayers]:
            outputs.append(node_out['outputs'])

        # Find highest activation value
        target = max(outputs)
#        print(target)
        outputs = [1 if x==target else 0 for x in outputs]

        if targetTest.size>0:
            if list(targetTest)!=outputs:
                return False,outputs
            else:
                return True,outputs
        else:
            return outputs

n_net = Node_Network(3,3,0.1,10,4)

#result,outputs = n_net.predictOne(np.array([1,2,3]),np.array([0,0,0,1]))

n_net.train(8,np.array([[1,2,3],[1,2,3]]),np.array([[0,0,0,1],[0,1,0,0]]))
