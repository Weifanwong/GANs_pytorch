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


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 3, 28, 28)
    return out

#hyperparameters
batch_size = 50
num_epoch = 100
input_dimension = 128
#Image processing
img_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x.repeat(3,1,1)),	
	transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
	])
#mnist datasets
mnist = datasets.MNIST(root='./data/mnist/',train=True,transform=img_transform,download=True)
#split batch
dataloader = torch.utils.data.DataLoader(dataset=mnist,batch_size=batch_size,shuffle=True)

#
Dis = Discriminator()
Gen = Generator(input_dimension)
if torch.cuda.is_available():
	Dis = Dis
	Gen = Gen

criterion = nn.BCELoss()
D_optimizer = torch.optim.Adam(Dis.parameters(),lr=0.0003)
G_optimizer = torch.optim.Adam(Gen.parameters(),lr=0.0003)


for epoch in range(num_epoch):
	for i, (img,_) in enumerate(dataloader):
		img = img.view(batch_size,-1)
		real_img = to_img(img.cpu().data)
		save_image(real_img,'./img/real_images-{}.png'.format(epoch + 1))
		real_img = Variable(img)
		real_label = Variable(torch.ones(batch_size))
		fake_label = Variable(torch.zeros(batch_size))


		#Discriminator training
		input_noise = Variable(torch.randn(batch_size,input_dimension))
		fake_img = Gen(input_noise)		
		real_img_logits = Dis(real_img)
		fake_img_logits = Dis(fake_img)
		D_loss_real = criterion(real_img_logits,real_label)
		D_loss_fake = criterion(fake_img_logits,fake_label)
		D_loss = D_loss_real + D_loss_fake
		print(D_loss.data)

		D_optimizer.zero_grad() #
		D_loss.backward()
		D_optimizer.step()

		#Generator training
		input_noise = Variable(torch.randn(batch_size,input_dimension))
		fake_img = Gen(input_noise)	
		fake_img_logits = Dis(fake_img)
		G_loss_fake = criterion(fake_img_logits,real_label)
		G_loss = G_loss_fake
		G_optimizer.zero_grad()
		G_loss.backward()
		G_optimizer.step()
		if (i + 1) % 50 == 0:
			print('Epoch: {}, D_real: {:.4f}, D_fake: {:.4f}'.format(epoch,real_img_logits.data.mean(),fake_img_logits.data.mean()))

	fake_img = to_img(fake_img.cpu().data)
	save_image(fake_img,'./img/fake_images-{}.png'.format(epoch + 1))

torch.save(Gen.state_dict(),'./generator.pth')
torch.save(Dis.state_dict(),'./discriminator.pth')