import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
from EWC import EWC


def sample_batch_index(total, batch_size):
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx

batch_size = 32

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = dset.MNIST('./data/', transform=img_transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True,
                        num_workers=0)

images = []
labels = []
for i, data in enumerate(dataloader):
    image, label = data
    images.append(image)
    labels.append(label)

images = torch.cat(images)
print(images.shape)

batchSize = 32  # We set the size of the batch.
imageSize = 28  # We set the size of the generated images (64x64).
num_worker = 10
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(512),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 2, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

netG = G().to(device)
netG.apply(weights_init)

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            # input 1*28*28
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # 64*14*14
            nn.Conv2d(64, 128, 2, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            # 128*8*8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            # 256*4*4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        img = output.view(-1)
        return img

netD = D().to(device)
netD.apply(weights_init)

optimizerD = optim.RMSprop(netD.parameters(), lr=0.0002, alpha = 0.9)
optimizerG = optim.RMSprop(netG.parameters(), lr=0.0002, alpha = 0.9)

name = "./result/test_EWC"
if not os.path.exists(name):
    os.makedirs(name)

if not os.path.exists(name + '/models'):
    os.makedirs(name +'/models')

if not os.path.exists(name + '/gen_images'):
    os.makedirs(name + '/gen_images')

losses = []
num_epochs = 50001

netD.load_state_dict(torch.load('./pretrain_weight/netD_20000.pt'))
netG.load_state_dict(torch.load('./pretrain_weight/netG_20000.pt'))
ewc = EWC(netG, netD, device, dataloader)
for epoch in range(num_epochs):
    for _ in range(3):
        batch_idx = sample_batch_index(images.shape[0], batchSize)
        data = images[batch_idx]
        real = Variable(data).to(device)
        noise = Variable(torch.randn(real.size()[0], 100, 1, 1)).to(device)
        fake = netG(noise)
        pred_real = netD(real)
        pred_fake = netD(fake)
        netD.zero_grad()
        loss_real = torch.mean(pred_real)
        loss_fake = torch.mean(pred_fake)
        ewc_loss = ewc.penalty(netD, gen=False)

        errD = -loss_real + loss_fake + ewc_loss
        errD.backward(retain_graph = True)
        optimizerD.step()

        for parm in netD.parameters():
            parm.data.clamp_(-0.01, 0.01)

    loss_fake_sum = loss_fake
    errG = - loss_fake_sum + ewc.penalty(netG)
    netG.zero_grad()
    errG.backward()
    optimizerG.step()

    if use_gpu:
        errD = errD.cpu()
        errG = errG.cpu()

    losses.append((errD.item(),errG.item()))
    if  epoch % 10== 0:
        print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f' % (epoch, num_epochs, errD.item(),errG.item()))
    if (epoch % 1000 == 0):
        noise = Variable(torch.randn(real.size()[0], 100, 1, 1)).to(device)
        fake = netG(noise)
        vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (name + "/gen_images", epoch),
                          normalize=True)
    if epoch % 1000 == 0:
        torch.save(netG.state_dict(), name + "/models/netG_%04d.pt" % (epoch))
        torch.save(netD.state_dict(), name + "/models/netD_%04d.pt" % (epoch))