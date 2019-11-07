from Discriminator import Discriminator
from Generator import Generator
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
import torch.autograd as autograd


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 32, 32)
    return out

#hyperparameters
batch_size = 50
num_epoch = 100
input_dimension = 128
weight_cliping_limit = 0.01
#Image processing
img_transform = transforms.Compose([
	transforms.Resize(32),
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.5,),std=(0.5,))
	])
#mnist datasets
mnist = datasets.MNIST(root='./data/mnist/',train=True,transform=img_transform,download=True)
#split batch
dataloader = torch.utils.data.DataLoader(dataset=mnist,batch_size=batch_size,shuffle=True)

#
check_point_dir = 'F:\\pytorch\\DCGAN\\checkpoint\\'
try:
	checkpoint = torch.load(check_point_dir+'discriminator.pth')
	Dis = Discriminator()
	Dis.load_state_dict(checkpoint['model_state_dict'])

	checkpoint = torch.load(check_point_dir+'generator.pth')	
	Gen = Generator(input_dimension)
	Gen.load_state_dict(checkpoint['model_state_dict'])
	start_epoch = checkpoint['epoch']
except:
	Dis = Discriminator()
	Gen = Generator(input_dimension)
	if torch.cuda.is_available():
		Dis = Dis
		Gen = Gen
	start_epoch = 0
criterion = nn.BCELoss()	
D_optimizer = torch.optim.Adam(Dis.parameters(),lr=0.0003)
G_optimizer = torch.optim.Adam(Gen.parameters(),lr=0.0003)

for epoch in range(start_epoch,num_epoch):
	for i, (img,_) in enumerate(dataloader):
		img = img.view(batch_size,-1)
		real_img = to_img(img.cpu().data)
		real_label = Variable(torch.ones(batch_size))
		fake_label = Variable(torch.zeros(batch_size))

		#Discriminator training
		input_noise = Variable(torch.randn(batch_size,input_dimension))
		fake_img = Gen(input_noise)
		real_img_logits = Dis(real_img)
		fake_img_logits = Dis(fake_img)

		#D loss for GANs
		D_loss_real = criterion(real_img_logits,real_label)
		D_loss_fake = criterion(fake_img_logits,fake_label)
		D_loss = D_loss_real + D_loss_fake
		D_optimizer.zero_grad() #
		D_loss.backward()
		D_optimizer.step()

		#clip for D parameters
		for p in Dis.parameters():
			p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

		#Generator training
		input_noise = Variable(torch.randn(batch_size,input_dimension))	
		fake_img = Gen(input_noise)	
		fake_img_logits = Dis(fake_img)

		#G loss for GANs
		G_loss_fake = criterion(fake_img_logits,real_label)
		G_loss = G_loss_fake
		G_optimizer.zero_grad()
		G_loss.backward()
		G_optimizer.step()
		if (i + 1) % 1 == 0:
			print('Epoch: {}, D_real: {:.4f}, D_fake: {:.4f}'.format(epoch,real_img_logits.data.mean(),fake_img_logits.data.mean()))

	fake_img = to_img(fake_img.cpu().data)
	save_image(fake_img,'./img/fake_images-{}.png'.format(epoch + 1))

	torch.save({
		'epoch':epoch,
		'model_state_dict':Gen.state_dict(),
		'optimizer_state_dict':G_optimizer.state_dict(),
		},
		'./checkpoint/generator.pth')
	torch.save({
		'epoch':epoch,
		'model_state_dict':Dis.state_dict(),
		'optimizer_state_dict':D_optimizer.state_dict(),
		},
		'./checkpoint/discriminator.pth')