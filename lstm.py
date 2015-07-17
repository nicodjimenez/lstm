import random

import numpy as np
import math

np.random.seed(0)
mem_cell_ct = 100
x_dim = 50
concat_len = x_dim + mem_cell_ct

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

class LstmParam:
    def __init__(self):
        self.wg =      np.random.random((mem_cell_ct, concat_len)) - 0.5 
        self.wi =      np.random.random((mem_cell_ct, concat_len)) - 0.5
        self.wf =      np.random.random((mem_cell_ct, concat_len)) - 0.5
        self.wo =      np.random.random((mem_cell_ct, concat_len)) - 0.5
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 

    def apply_diff(self, lr = 1):
        self.wg += lr * self.wg_diff
        self.wi += lr * self.wi_diff
        self.wf += lr * self.wf_diff
        self.wo += lr * self.wo_diff
        # reset diffs
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 

class LstmState:
    def __init__(self):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s  = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)

        # simply concat of x(t) and h(t-1)
        self.top_diff = np.zeros(concat_len)
        self.bottom_diff = np.zeros(concat_len)
    
class LSTM:
    def __init__(self, lstm_param, prev = None):
        # initialize new activations
        self.state = LstmState()
        # store reference to parameters
        self.param = lstm_param
        self.prev = prev 
        self.next = None
        self.x = None
        self.xc = None
        self.bottom_diff = None

    def forward(self, x):
        # add element to linked list
        self.next = LSTM(lstm_param)
        self.next.prev = self

        # init code: set prev state to zero
        if self.prev:
            xc = np.hstack((x,  self.prev.state.h))
            prev_s = self.prev.state.s
        else:
            xc = np.hstack((x,  np.zeros_like(self.state.h)))
            prev_s = np.zeros_like(self.state.f)

        self.state.g = np.tanh(np.dot(self.param.wg, xc))
        self.state.i = sigmoid(np.dot(self.param.wi, xc))
        self.state.f = sigmoid(np.dot(self.param.wf, xc))
        self.state.o = sigmoid(np.dot(self.param.wo, xc))
        self.state.s = np.multiply(self.state.g, self.state.i) + np.multiply(prev_s, self.state.f)
        self.state.h = np.multiply(self.state.s, self.state.o)
        self.x = x
        self.xc = xc
    
    def backward(self, top_diff):
        # top_diff is the diff w.r.t to h 
        ds = np.multiply(self.state.o, top_diff)
        do = np.multiply(self.state.s, top_diff)
        di = np.multiply(self.state.g, ds)
        dg = np.multiply(self.state.i, ds)
        if self.prev:
            df = np.multiply(self.state.i, ds)
        else:
            df = np.zeros_like(self.state.f)

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = (1. - self.state.i) * self.state.i * di 
        df_input = (1. - self.state.f) * self.state.f * df 
        do_input = (1. - self.state.o) * self.state.o * do 
        dg_input = (1. - self.state.g ** 2) * dg

        # diffs w.r.t. params
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)

        # diffs w.r.t. inputs
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)
        self.bottom_diff = dxc

class EuclideanLoss:
    def __init__(self):
        pass

    @classmethod
    def forward(self, pred, label):
        return np.linalg.norm(pred - label)

    @classmethod
    def backward(self, pred, label):
        return 2. * np.multiply((pred - label), pred)
        
random.seed(0)
target_value = np.random.random(mem_cell_ct)
#print EuclideanLoss.forward(target_value, target_value)
#print EuclideanLoss.backward(target_value, target_value)

lstm_param = LstmParam() 
lstm = LSTM(lstm_param)
input_val = np.random.random(x_dim)

for _ in range(10):
    lstm.forward(input_val)
    loss = EuclideanLoss.forward(lstm.state.h, target_value)
    print "Loss: ", loss
    loss_diff = EuclideanLoss.backward(lstm.state.h, target_value)
    lstm.backward(loss_diff)
    lstm.param.apply_diff(0.0001)

