import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self,input_size):
		super(Generator,self).__init__()
		self.gen = nn.Sequential(
			nn.Linear(input_size,256),
			nn.ReLU(True),
			nn.Linear(256,256),
			nn.ReLU(True),
			nn.Linear(256,784*3),
			nn.Tanh()
			)
	def forward(self,x):
		x = self.gen(x)
		return x
