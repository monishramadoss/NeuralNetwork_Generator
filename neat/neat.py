import torch
import torch.nn as nn



class RecurrentNet():
    def __init__(self, n_inputs, n_hidden, n_outputs,input_to_hidden, hidden_to_hidden, output_to_hidden,
                 input_to_output, hidden_to_output, output_to_output,
                 hidden_responses, output_responses,
                 hidden_biases, output_biases,
                 batch_size=1,
                 use_current_activs=False,
                 activation=nn.Sigmoid,
                 n_internal_steps=1,
                 dtype=torch.float64,
                 device='cpu'):

        self.use_current_activs = use_current_activs
        self.activation = activation
        self.n_internal_steps = n_internal_steps
        self.dtype = dtype

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs


