from lstm import *

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

    for _ in range(10):
        lstm.forward(input_val)
        loss = EuclideanLoss.forward(lstm.state.h, target_value)
        print "Loss: ", loss
        loss_diff = EuclideanLoss.backward(lstm.state.h, target_value)
        lstm.backward(loss_diff)
        lstm.param.apply_diff(1)

def learn_sequence():
    target_value = np.random.random(mem_cell_ct)
    #print EuclideanLoss.forward(target_value, target_value)
    #print EuclideanLoss.backward(target_value, target_value)
    lstm_param = LstmParam() 
    lstm = LSTM(lstm_param)
    input_val = np.random.random(x_dim)
    #input_val = np.ones(x_dim)

    for _ in range(10):
        lstm.forward(input_val)
        loss = EuclideanLoss.forward(lstm.state.h, target_value)
        print "Loss: ", loss
        loss_diff = EuclideanLoss.backward(lstm.state.h, target_value)
        lstm.backward(loss_diff)
        lstm.param.apply_diff(1)

#test()
#check_gradients()
#EuclideanLoss.check_gradients()
"""
TODO: make network learn sequence
TODO: try to train on sequence data
"""

lstm_param = LstmParam() 
lstm_net = LstmNetwork(lstm_param)

#y_list = range(4)
y_list = [1]
input_val = np.random.random(x_dim)

for _ in range(500):
    print "new iteration"
    for _ in range(len(y_list)):
        lstm_net.x_list_add(input_val)

    loss = lstm_net.y_list_is(y_list, ToyLossLayer)
    print "loss: ", loss
    lstm_param.apply_diff(1)
    lstm_net.x_list_clear()
#print "loss: ", loss





