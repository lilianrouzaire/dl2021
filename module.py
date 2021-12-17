import torch
import math
from math import sqrt
from torch import empty


torch.set_grad_enabled(False)

##############################################################################
########################### ABSTRACT PARENT CLASS ############################
##############################################################################

class Module(object):
	
	def forward(self, *input):
		raise NotImplementedError("No forward is defined")
		
	def backward(self, *gradwrtoutput):
		raise NotImplementedError("No backward is defined")
		
	def param(self):
		return  []

	def zero_grad(self):
		pass

##############################################################################
################################## MODULES ###################################
##############################################################################

# Fully connected layer
class Linear(Module):
	
	def __init__(self, input_dim, output_dim, gain = sqrt(2)):
		# Initializing weights with "Xavier initialization" (slides 5.5 pages 13-16)
		self.weight = torch.empty(output_dim, input_dim).normal_(0, gain * sqrt(2 / (input_dim + output_dim)))
		# Initializing weights with a uniform distribution
		#self.weight = torch.empty(output_dim, input_dim).uniform_(-1 / math.sqrt(input_dim), 1 / math.sqrt(input_dim))

		# Initializing bias to zero
		self.bias = torch.empty(output_dim).zero_()

		# Initializing the gradient wrt to the parameters to zero
		self.gradwrtbias = torch.empty(output_dim).zero_()
		self.gradwrtweight = torch.empty(output_dim, input_dim).zero_()

		# Storing the input/output dimensions is necessary for the backward pass (see below)
		self.input_dim = input_dim
		self.output_dim = output_dim

	def forward(self, input):
		self.input = input # Stores the input in the class for the backprop
		return input @ self.weight.T + self.bias # Matrix multiplication of the transposed weight with the input and then add the bias

	def backward(self, gradwrtoutput):
		# Accumulating the gradient wrt the params (slides 3.6)
		self.gradwrtweight += (gradwrtoutput.transpose(1, 2) @ self.input).mean(0)
		# Need to reshape the gradient wrt the ouput to the dimension of the output so that we can add it to the bias tensor
		self.gradwrtbias += gradwrtoutput.mean(0).reshape(self.output_dim)
		return gradwrtoutput @ self.weight

	def zero_grad(self):
		# Setting the gradient wrt to the params at 0
		self.gradwrtweight = torch.empty(self.output_dim, self.input_dim).zero_()
		self.gradwrtbias = torch.empty(self.output_dim).zero_()

	def param(self):
		return [
		[self.weight, self.gradwrtweight],
		[self.bias, self.gradwrtbias]
		]

class ReLU(Module):

	def forward(self, x):
		self. x = x
		# ReLU returns max(0, x)
		# Cannot use max() function from math on tensors, need a element-wise function from pytorch
		return torch.maximum(torch.empty(x.size()).zero_(), x)

	def backward(self, gradwrtoutput):
		# The derivative of ReLU is 1 for x > 0 and 0 for x <= 0
		d = (self.x > 0).long()
		return d * gradwrtoutput

	def param(self):
		return []

class Tanh(Module):

	def forward(self, x):
		self.x = x
		return x.tanh()

	def backward(self, gradwrtoutput):
		# The derivative of tanh is 1/cosh^2
		d = 1/(self.x.cosh()).pow(2)
		return d * gradwrtoutput

	def param(self):
		return []


class Sequential(Module):

	def __init__(self, modules):
		# Storing the modules as a class parameter
		self.modules = modules

	def forward(self, x):
		# Activating the forward pass over the input for each module
		for module in self.modules:
			x = module.forward(x)

		return x

	def backward(self, gradwrtoutput):
		# Taking the modules in reverse order, and activating the backward pass for each one of them
		for module in reversed(self.modules):
			gradwrtoutput = module.backward(gradwrtoutput)

		return gradwrtoutput

	def zero_grad(self):
		for module in self.modules:
			if module.param(): module.zero_grad()

	def param(self):
		# Putting together the parameters of all modules
		params = []
		for module in self.modules:
			if module.param(): params.append(module.param())
		return params


# As LossMSE must be a module, we also make a class inheriting fron the abstract class Module
# The initialization stores the target, and the forward pass computes the MSE with the output
# The backward is implemeted as defined in practical 3
class LossMSE(Module):

	def __init__(self, target):
		self.target = target

	def forward(self, output):
		self.output = output
		# Compute the least squares (square of the norm of the difference vector)
		return (output - self.target).pow(2).sum()

	def backward(self):
		# As in Série 3
		return 2.0 * (self.output - self.target)