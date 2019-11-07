import torch
import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(
				in_channels=1, 
				out_channels=64,
				kernel_size=4, 
				stride=2, 
				padding=1
				),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(
				in_channels=64, 
				out_channels=128,
				kernel_size=4, 
				stride=2, 
				padding=1, 
				),	
			nn.BatchNorm2d(128),	
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(
				in_channels=128, 
				out_channels=256,
				kernel_size=4, 
				stride=2, 
				padding=1, 
				),
			nn.BatchNorm2d(256),							
			nn.LeakyReLU(0.2,inplace=True),	
			)
		self.Linear = nn.Sequential(nn.Linear(256*4*4,1),nn.Sigmoid())
	def forward(self,x):
		x = self.conv(x)
		x = x.view(x.size(0),-1)
		x = self.Linear(x)
		return x
