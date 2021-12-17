import torch

from torch import optim
from torch import Tensor
from torch import nn
import torch.nn.functional as F


from dlc_practical_prologue import generate_pair_sets

# Convention :
#   - tensors and variables used for the current epoch only are prefixed with e_
#   - tensors and variables used for the current mini-batch only are prefixed with b_

##############################################################################
############################## HELPER FUNCTIONS ##############################
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

#############################################################

# Compute the number of errors of a model given the input and the target
def compute_nb_errors(model, input, target, mini_batch_size):

	nb_errors = 0

	for b in range(0, input.size(0), mini_batch_size):
		b_output = model(input.narrow(0, b, mini_batch_size))
		b_output = (b_output > 0).int().float()
		b_target = target.narrow(0, b, mini_batch_size)

		for k in range(0, mini_batch_size):
			if b_output[k] != b_target[k]:
				nb_errors += 1

	return nb_errors

#############################################################

# This custom loss function aims at providing an indicator of the correctness of the prediction regarding the predicted and true classes for each image
#	- prediction : the prediction of the model (whether the first image is lesser or equal to the second)
#	- predicted_class_img1 : the result of the MNIST Classification block module for the first image (a digit)
#	- predicted_class_img2 : the result of the MNIST Classification block module for the second image (a digit)
#	- true_class_img1 : the true digit value of the first image
#	- true_class_img2 : the true digit value of the second image

#### NOT COMPLETED - NOT WORKING
def custom_loss(prediction, predicted_class_img1, predicted_class_img2, true_class_img1, true_class_img2):
	return 0

#############################################################

# Train the model and assess its performances with the test dataset
def train_model(model, train_input, train_target, train_classes, test_input, test_target, test_classes, nb_epochs, mini_batch_size, optimizer):

	for e in range(nb_epochs):

		e_acc_loss = 0

		model.train() # Tell the framework we are in training phase

		# Partition the tensors in small batches of size mini_batch_size
		for b in range(0, train_input.size(0), mini_batch_size):
			# For each batch, resize the tensors according to mini_batch_size
			b_input = train_input.narrow(0, b, mini_batch_size)
			b_target = train_target.narrow(0, b, mini_batch_size)
			b_classes = train_classes.narrow(0, b, mini_batch_size)

			# Classes is a N x 2 tensor containing the classes of both digits
			# Then the classes of the first image are at position [:, 0] (first column) and the classes for the second image are at position [:, 1] (second column) 
			b_true_class_img1 = b_classes[:, 0]
			b_true_class_img2 = b_classes[:, 1]

			# Run the model
			# If the model tries and retrieve each image class, the return value will be a tuple containing each predicted image class
			# Otherwise it will be the predicted target of the problem
			b_output = model(b_input)

			# Compute the loss
			criterion = nn.BCEWithLogitsLoss()
			e_loss = criterion(b_output, b_target.type_as(b_output)) # Apply Cross Entropy Loss to the output with respect to the target
			e_acc_loss += e_loss.item() # e_loss is a tensor 1x1, we extract the value with item()

			e_loss.requires_grad_(True)

			model.zero_grad() # Reset the gradient before the backpropagation process
			e_loss.backward() # Backprop
			optimizer.step() # Optimizes the gradient descent

		model.eval() # Tell the framework we are in evaluation phase

		with torch.no_grad(): # Disable gradient for error rate computation
			e_train_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
			e_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)

		print_epoch_results(e + 1, e_acc_loss, e_train_errors, e_test_errors, train_input.size(0), test_input.size(0))

# NOT COMPLETED - NOT WORKING
def train_model_custom_loss(model, train_input, train_target, train_classes, test_input, test_target, test_classes, nb_epochs, mini_batch_size, optimizer):

		e_acc_loss = 0

		model.train() # Tell the framework we are in training phase

		# Partition the tensors in small batches of size mini_batch_size
		for b in range(0, train_input.size(0), mini_batch_size):
			# For each batch, resize the tensors according to mini_batch_size
			b_input = train_input.narrow(0, b, mini_batch_size)
			b_target = train_target.narrow(0, b, mini_batch_size)
			b_classes = train_classes.narrow(0, b, mini_batch_size)

			# Classes is a N x 2 tensor containing the classes of both digits
			# Then the classes of the first image are at position [:, 0] (first column) and the classes for the second image are at position [:, 1] (second column) 
			b_true_class_img1 = b_classes[:, 0]
			b_true_class_img2 = b_classes[:, 1]

			# Run the model
			# If the model tries and retrieve each image class, the return value will be a tuple containing each predicted image class
			# Otherwise it will be the predicted target of the problem
			b_output = model(b_input)

			b_prediction = b_output[0]
			b_predicted_class_img1 = b_output[1]
			b_predicted_class_img2 = b_output[2]
			e_acc_loss += custom_loss(b_prediction, b_predicted_class_img1, b_predicted_class_img2, b_true_class_img1, b_true_class_img2)
		
			model.zero_grad() # Reset the gradient before the backpropagation process
			#e_loss.backward() # Backprop
			optimizer.step() # Optimizes the gradient descent

		model.eval() # Tell the framework we are in evaluation phase

		#with torch.no_grad(): # Disable gradient for error rate computation
		#	e_train_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
		#	e_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)

		#print_epoch_results(e + 1, e_acc_loss, e_train_errors, e_test_errors, train_input.size(0), test_input.size(0))


##############################################################################
############################ NETWORKS DEFINITIONS ############################
##############################################################################

# From https://debuggercafe.com/deep-learning-with-pytorch-image-classification-using-neural-networks/
class MNISTClassificationBlock(nn.Module):
	def __init__(self, p = 0.0, nb_hidden = 50):
		super().__init__()
		
		self.conv1 = nn.Conv2d(1, 30, 3)
		self.conv2 = nn.Conv2d(30, 30, 3)
		self.conv3 = nn.Conv2d(30, 50, 2)
		self.fc1 = nn.Linear(50, nb_hidden) 
		self.fc2 = nn.Linear(nb_hidden, 10)
		self.dropout = nn.Dropout(p)

	def forward(self, x):

		x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
		x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
		x = F.relu(self.conv3(x))
		x = F.relu(self.fc1(x.view(-1, 50)))
		x = self.dropout(x)
		x = self.fc2(x)

		return x

class ResidualBlock(nn.Module):

	def __init__(self, nb_channels = 3, kernel_size = 3):
		super().__init__()

		self.conv1 = nn.Conv2d(1, 5, kernel_size = kernel_size, padding = (kernel_size - 1) // 2)
		self.bn1 = nn.BatchNorm2d(5)
		self.conv2 = nn.Conv2d(5, 10, kernel_size = kernel_size, padding = (kernel_size - 1) // 2)
		self.bn2 = nn.BatchNorm2d(10)

	def forward(self, x):
		y = self.conv1(x)
		y = self.bn1(y)
		y = F.relu(y)
		y += x
		y = self.conv2(y)
		y = self.bn2(y)

		return y

class SimpleNetwork(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv = nn.Conv2d(2, 15, kernel_size=14) # Convolution
		self.lin = nn.Linear(15, 1) # Linear
		

	def forward(self, x):
		x = self.conv(x)
		x = F.relu(x)
		x = self.lin(x.view(-1,15))
		x = x.view(-1)
		
		return x

class WSNetwork(nn.Module):

	def __init__(self, nb_hidden = 30, p = 0.0):
		super().__init__()
		
		self.conv1 = nn.Conv2d(2, 15, 3)
		self.conv2 = nn.Conv2d(15, 15, 3)
		self.conv3 = nn.Conv2d(15, 64, 2)
		self.fc1 = nn.Linear(64, nb_hidden)
		self.fc2 = nn.Linear(nb_hidden, 1)
		self.dropout = nn.Dropout(p)
						
	def forward(self, x):
				
		x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
		x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
		x = F.relu(self.conv3(x))
		x = F.relu(self.fc1(x.view(-1, 64)))
		x = self.dropout(x)
		x = self.fc2(x)
		x = x.view(-1)
		
		return x

class ALNetwork(nn.Module):

	def __init__(self, nb_hidden = 30, p = 0.0):
		super().__init__()
		
		self.mnist_classification_block = MNISTClassificationBlock()
		self.residual_blocks = ResidualBlock(3)
						
	def forward(self, x):
				
		img1 = x.split(1,1)[0]
		img2 = x.split(1,1)[1]

		# Get the image class for each image separately
		_, img1_class = self.mnist_classification_block(img1).max(1)
		_, img2_class = self.mnist_classification_block(img2).max(1)

		# Network prediction : whether the first digit is lesser or equal to the second
		x = (img2_class - img1_class).view(-1)

		x = (x > 0).int().float()

		return x, img1_class, img2_class

##############################################################################
############################## NETWORKS RUNNING ##############################
##############################################################################

# Defining parameters
nb_resblock = 1
nb_hidden = 50
dropout_prob = 0.3 # Probability of dropout
eta = 2e-3 # Learning rate
mini_batch_size = 100
nb_epochs = 25
nb_rounds = 1

# Use the dlc_practical_prologue helper functions
nb_samples = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(nb_samples)

########################################
###### Running the simple network ######
########################################

for r in range(nb_rounds):

	model = WSNetwork(p = dropout_prob)

	print(model)
	nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(str(nb_params) + ' available parameters')

	optimizer = torch.optim.Adam(model.parameters(), lr = eta)
	
	train_model(model, train_input, train_target, train_classes, test_input, test_target, test_classes, nb_epochs, mini_batch_size, optimizer)

