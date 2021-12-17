from module import *
import torch
import math
import numpy
import matplotlib.pyplot as plt

##############################################################################
################################ MAIN NETWORK ################################
##############################################################################

### This network works as suggested in the assignement statement. It takes as parameters :
#		input_dim : the dimension of the input (in our case, 2)
#		output_dim : the dimension of the output (in our case, 1)
#		nb_hidden : the size of the hidden layers
class Network(Module):

	# Initializing the network layers, and the sequence of layers
	def __init__(self, input_dim, output_dim, nb_hidden):
		super().__init__()

		# 1 input unit, 3 hidden layers of 25 units, and 1 output unit 
		self.inputUnit = Linear(input_dim, nb_hidden)
		self.hiddenLayer1 = Linear(nb_hidden, nb_hidden)
		self.hiddenLayer2 = Linear(nb_hidden, nb_hidden)
		self.hiddenLayer3 = Linear(nb_hidden, nb_hidden)
		self.outputUnit = Linear(nb_hidden, output_dim)

		# For the sequence of layers, we use ReLU as activation function
		# Using the Sequential() module defined in module.py
		self.sequence = Sequential([
			self.inputUnit,
			ReLU(),
			self.hiddenLayer1,
			ReLU(),
			self.hiddenLayer2,
			ReLU(),
			self.hiddenLayer3,
			ReLU(),
			self.outputUnit,
			ReLU()
		])

	# Calling the forward pass on the defined sequence
	def forward(self, x):
		return self.sequence.forward(x)

	# Calling the backward pass on the defined sequence
	def backward(self, gradwrtoutput):
		return self.sequence.backward(gradwrtoutput)

	def zero_grad(self):
		self.sequence.zero_grad()

	# Rendering the parameters of the sequence (param aggregation)
	def param(self):
		return self.sequence.param()

##############################################################################
######################### MODEL TRAINING AND TESTING #########################
##############################################################################

# Print the number of errors, the loss and the error rate for a given epoch
def print_epoch_results(epoch, loss, train_errors, test_errors, train_input_size, test_input_size):
	print('Epoch {:2d} | loss = {:7.3f} | {:5.1f} train errors | {:5.1f} test errors | train error rate {:4.1f}% | test error rate {:4.1f}%'.format(
			epoch,
			loss,
			train_errors,
			test_errors,
			train_errors / train_input_size * 100,
			test_errors / test_input_size * 100
	))

##############################################################################

# Runs the model's forward and backward passes, computes the loss and the error rates
# Uses mini-batches of size mini_batch_size
def train_model(model, train_input, train_target, test_input, test_target, mini_batch_size, nb_epochs, eta):

	for e in range(nb_epochs):
		e_acc_loss = 0 # Accumulated loss for this epoch

		for b in range(0, train_input.size(0), mini_batch_size):
			b_output = model.forward(train_input.narrow(0, b, mini_batch_size))
			# The loss is implemented as module, therefore we initialize it with the target and call the forward pass with the output to compute the MSE
			b_loss = LossMSE(train_target.narrow(0, b, mini_batch_size))
			e_acc_loss += b_loss.forward(b_output).item()

			# Resetting the gradient for the model with parameters
			model.zero_grad()
			# Calling the backward pass with the computed loss
			model.backward(b_loss.backward())

			for p in model.param():
				
				p[0][0] -= eta * p[0][1]
				p[1][0] -= eta * p[1][1]

		train_errors = compute_nb_errors(model, train_input, train_target)
		test_errors = compute_nb_errors(model, test_input, test_target)
		print_epoch_results(e + 1, e_acc_loss, train_errors, test_errors, train_input.size(0), test_input.size(0))

	# Error repartition visulization
	"""
	mis = []
	output = model.forward(test_input)
	for i in range(0, output.size(0)):
		if output[i] < 0.5 and 1 == test_target[i]:
			mis.append(i)

	return mis
	"""

##############################################################################

# Computes the number of errors of a model over a given input with respect to a target
def compute_nb_errors(model, input, target):

	# Computing the output with the model forward pass
	output = model.forward(input)

	# Assigns each point to the closest category
	# if > 0.5, let's consider the point is inside the sphere and flagging it as 1, otherwise flagging it as 0
	# This value will represent the NN classification
	inside = (output > 0.5).long()

	# A point is misclassified if the target is different from the NN classification
	# Will return 1 if misclassified, 0 otherwise
	errors = (target - inside).abs()

	# Couting the errors
	return errors.sum()

##############################################################################
############################## DATA GENERATION ###############################
##############################################################################

# Generates a set of nb points uniformly in [0, 1] x [0, 1], labelling them if they are in a disk of center 0.5 and radius sqrt(1/2pi)
# As in série 5 (but changing the disk center)
def generate_dataset(nb):
	# Uniform distribution in [0, 1] x [0, 1]
	input = torch.empty(nb, 2).uniform_(0, 1)

	# Labelling the points whether they are in the disk or not
	target = input.sub(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
	
	# Reshape the tensors to add a dimension
	input = input.reshape(nb, 1, 2)
	target = target.reshape(nb, 1, 1)

	return input, target


############################################################################## 

# Parameters
mini_batch_size = 20
nb_hidden = 25 # Number of hidden units
nb_epochs = 30
nb = 10000
eta = 1e-3 # Learning rate

train_input, train_target = generate_dataset(nb)
test_input, test_target = generate_dataset(nb)

model = Network(2, 1, nb_hidden)

train_model(model, train_input, train_target, test_input, test_target, mini_batch_size, nb_epochs, eta) 
