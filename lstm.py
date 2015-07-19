import random

import numpy as np
import math

np.random.seed(0)

# parameters for input data dimension and lstm cell count 
mem_cell_ct = 20
x_dim = 2
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
        # reset diffs to zero
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
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        self.bottom_diff_x = np.zeros_like(x_dim)
    
class LSTM:
    def __init__(self, lstm_param, lstm_state = None):
        # initialize new activations
        # note: one can make lstm_state share memory across layers
        if lstm_state == None:
            self.state = LstmState()
        else:
            lstm_state = lstm_state

        # store reference to parameters
        self.param = lstm_param
        self.x = None
        self.xc = None

    def bottom_data_is(self, x, s_prev = None, h_prev = None):
        # if this is the first lstm unit in the network
        if s_prev == None: s_prev = np.zeros_like(self.state.s)
        if h_prev == None: h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  h_prev))

        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o
        self.x = x
        self.xc = xc
    
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.state.o * top_diff_h + top_diff_s
        #ds = self.state.o * top_diff_h 
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

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

        # save bottom diffs
        self.state.bottom_diff_s = ds
        self.state.bottom_diff_x = dxc[:x_dim]
        self.state.bottom_diff_h = dxc[x_dim:]

class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_unit_list = []
        # input sequence
        self.x_list = []
        self.y_list = []

    def y_list_is(self, y_list, loss_layer):
        """
        Updates diffs by setting target sequence 
        with corresponding loss layer. 
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_layer.loss(self.lstm_unit_list[idx].state.h, y_list[idx])
        #print "top loss: ", loss
        diff_h = loss_layer.bottom_diff(self.lstm_unit_list[idx].state.h, y_list[idx])
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(mem_cell_ct)
        self.lstm_unit_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

        ## ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ## we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_unit_list[idx].state.h, y_list[idx])
        #    #print "new loss: ", loss
            diff_h = loss_layer.bottom_diff(self.lstm_unit_list[idx].state.h, y_list[idx])
            diff_h += self.lstm_unit_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_unit_list[idx + 1].state.bottom_diff_s
            self.lstm_unit_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1 

        #self.y_list = y_list
        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_unit_list):
            # need to add new lstm unit
            self.lstm_unit_list.append(LSTM(self.lstm_param))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_unit_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_unit_list[idx - 1].state.s
            h_prev = self.lstm_unit_list[idx - 1].state.h
            self.lstm_unit_list[idx].bottom_data_is(x, s_prev, h_prev)

        print self.lstm_unit_list[idx].state.h[0]
    #def 

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

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


