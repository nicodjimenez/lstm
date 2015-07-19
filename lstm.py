import random

import numpy as np
import math

np.random.seed(0)

# parameters for input data dimension and lstm cell count 
mem_cell_ct = 20
x_dim = 100
concat_len = x_dim + mem_cell_ct

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

class LstmParam:
    def __init__(self):
        # weight matrices
        self.wg =      (np.random.random((mem_cell_ct, concat_len)) - 0.5) * 0.1
        self.wi =      (np.random.random((mem_cell_ct, concat_len)) - 0.5) * 0.1
        self.wf =      (np.random.random((mem_cell_ct, concat_len)) - 0.5) * 0.1
        self.wo =      (np.random.random((mem_cell_ct, concat_len)) - 0.5) * 0.1
        # bias terms
        self.bg =      (np.random.random(mem_cell_ct) - 0.5) * 0.1
        self.bi =      (np.random.random(mem_cell_ct) - 0.5) * 0.1
        self.bf =      (np.random.random(mem_cell_ct) - 0.5) * 0.1
        self.bo =      (np.random.random(mem_cell_ct) - 0.5) * 0.1
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct) 

    def apply_diff(self, lr = 1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct) 

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
        #self.bottom_diff = None

    def forward(self, x):
        # add element to linked list
        self.next = LSTM(self.param)
        self.next.prev = self

        # init code: set prev state to zero
        if self.prev:
            xc = np.hstack((x,  self.prev.state.h))
            prev_s = self.prev.state.s
        else:
            xc = np.hstack((x,  np.zeros_like(self.state.h)))
            prev_s = np.zeros_like(self.state.f)

        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
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
            df = self.state.i * ds
        else:
            df = np.zeros_like(self.state.f)

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = (1. - self.state.i) * self.state.i * di 
        df_input = (1. - self.state.f) * self.state.f * df 
        do_input = (1. - self.state.o) * self.state.o * do 
        dg_input = (1. - self.state.g ** 2) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input       
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input       

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)
        self.state.bottom_diff = dxc

class EuclideanLoss:
    def __init__(self):
        pass

    @classmethod
    def forward(self, pred, label):
        return (np.linalg.norm(pred - label) ** 2) / len(pred)

    @classmethod
    def backward(self, pred, label):
        return 2. * (pred - label) / len(pred)

    @classmethod
    def check_gradients(self):
        pred = np.random.random(5)
        label = np.random.random(5)
        loss_0 = self.forward(pred, label)
        diff = self.backward(pred, label)
        #rand_noise = np.random.random(5) * 0.0001
        delta = 0.0000001
        pred[0] += delta
        #pred += rand_noise
        loss_1 = self.forward(pred, label)
        loss_deriv = (loss_1 - loss_0) / delta
        #pred_deriv = delta * diff[0]
        print "loss deriv:" , loss_deriv
        print "diff[0]", diff[0]
        print "per error: ", (loss_deriv - diff[0]) / loss_deriv

def check_gradients():
    lstm_param = LstmParam() 
    lstm = LSTM(lstm_param)
    input_val = np.random.random(x_dim)
    target_value = np.zeros(mem_cell_ct)

    # init activations
    lstm.forward(input_val)
    loss_0 = EuclideanLoss.forward(lstm.state.h, target_value)
    loss_diff = EuclideanLoss.backward(lstm.state.h, target_value)
    lstm.backward(loss_diff)
    bottom_diff = lstm.state.bottom_diff

    # modify input 
    delta = 0.00001
    input_val[0] += delta

    lstm.forward(input_val)
    loss_1 = EuclideanLoss.forward(lstm.state.h, target_value)
    loss_grad = (loss_1 - loss_0) / delta

    print "bottom diff:", bottom_diff[0]
    print "loss grad:", loss_grad
    print (loss_grad - bottom_diff[0]) / loss_grad

def test():
    #target_value = np.zeros(mem_cell_ct)
    target_value = np.random.random(mem_cell_ct)
    #print EuclideanLoss.forward(target_value, target_value)
    #print EuclideanLoss.backward(target_value, target_value)

    lstm_param = LstmParam() 
    lstm = LSTM(lstm_param)
    input_val = np.random.random(x_dim)
    #input_val = np.ones(x_dim)

    for _ in range(40):
        lstm.forward(input_val)
        loss = EuclideanLoss.forward(lstm.state.h, target_value)
        print "Loss: ", loss
        loss_diff = EuclideanLoss.backward(lstm.state.h, target_value)
        lstm.backward(loss_diff)
        lstm.param.apply_diff(1)

test()
#check_gradients()
#EuclideanLoss.check_gradients()
"""
TODO: make network learn sequence
TODO: try to train on sequence data
"""


