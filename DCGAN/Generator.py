import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self,input_size):
		super(Generator,self).__init__()
		self.Linear = nn.Sequential(nn.Linear(input_size,256*4*4))
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(
				in_channels=256, 
				out_channels=128,
				kernel_size=4, 
				stride=2, 
				padding=1, 
				),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.ConvTranspose2d(
				in_channels=128, 
				out_channels=64,
				kernel_size=4, 
				stride=2, 
				padding=1, 
				),
			nn.BatchNorm2d(64),		
			nn.ReLU(True),
			nn.ConvTranspose2d(
				in_channels=64, 
				out_channels=1,
				kernel_size=4, 
				stride=2, 
				padding=1, 
				),
			nn.Tanh()
			)
	def forward(self,x):
		x = self.Linear(x)
		x = x.view([x.size(0),256,4,4])
		x = self.deconv(x)
		return x
