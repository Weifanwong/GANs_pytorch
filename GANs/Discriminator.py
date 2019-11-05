import torch
import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.dis = nn.Sequential(
			nn.Linear(784*3,256),
			nn.LeakyReLU(0.02),
			nn.Linear(256,256),
			nn.LeakyReLU(0.02),
			nn.Linear(256,1),
			nn.Sigmoid()
			)
	def forward(self,x):
		x = self.dis(x)
		return x
