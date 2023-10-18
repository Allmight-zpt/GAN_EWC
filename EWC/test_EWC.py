import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.EWC import EWC
from utils.GAN import D, G
from utils.tools import getMnist, getFashionMnist, sample_batch_index, createWorkDir, preProcess
from torch.utils.tensorboard import SummaryWriter

'''
1. 使用在FashionMnist数据集上预训练得到的模型参数进行GAN模型的初始化
2. 将EWC算法应用于GAN，让其在Mnist数据集上进行训练
3. 让GAN在学习生成Mnist数据集的同时防止FashionMnist数据集的遗忘
'''

# 超参数设置
batchSize = 64
num_epochs = 50001
losses = []
workDirName = "../result/test_EWC"
logDir = '../logs/test_EWC'
DataRoot = '../data'
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
dataloader = getMnist(DataRoot, batchSize)
FashionMnist_dataloader = getFashionMnist(DataRoot, batchSize)

# 数据预处理
images = preProcess(dataloader)

# 初始化生成器和判别器
netG = G().to(device)
netD = D().to(device)

# 加载预训练模型
netD.load_state_dict(torch.load('../pretrain_weight/GANs/FashionMnist_netD_20000.pt'))
netG.load_state_dict(torch.load('../pretrain_weight/GANs/FashionMnist_netG_20000.pt'))

# 初始化优化器
optimizerD = optim.RMSprop(netD.parameters(), lr=0.0002, alpha=0.9)
optimizerG = optim.RMSprop(netG.parameters(), lr=0.0002, alpha=0.9)

# 构造工作目录
createWorkDir(workDirName)

# 创建EWC实例
ewc = EWC(netG, netD, device, FashionMnist_dataloader)

# 创建 SummaryWriter
writer = SummaryWriter(log_dir=logDir)

# 开始训练
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
        ewc_loss_D = ewc.penalty(netD, gen=False)
        writer.add_scalar(tag='ewc_loss_D', scalar_value=ewc_loss_D)
        errD = -loss_real + loss_fake + ewc_loss_D * 10
        errD.backward(retain_graph=True)
        optimizerD.step()

        for parm in netD.parameters():
            parm.data.clamp_(-0.01, 0.01)

    loss_fake_sum = loss_fake
    ewc_loss_G = ewc.penalty(netG)
    writer.add_scalar(tag='ewc_loss_G', scalar_value=ewc_loss_G)
    writer.add_scalar(tag='loss_fake_sum', scalar_value=-loss_fake_sum)
    errG = - loss_fake_sum + ewc_loss_G * 60000
    netG.zero_grad()
    errG.backward()
    optimizerG.step()

    if use_gpu:
        errD = errD.cpu()
        errG = errG.cpu()

    losses.append((errD.item(),errG.item()))
    if epoch % 10== 0:
        print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f' % (epoch, num_epochs, errD.item(),errG.item()))
    if (epoch % 1000 == 0):
        noise = Variable(torch.randn(real.size()[0], 100, 1, 1)).to(device)
        fake = netG(noise)
        vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (workDirName + "/gen_images", epoch), normalize=True)
    if epoch % 1000 == 0:
        torch.save(netG.state_dict(), workDirName + "/models/netG_%04d.pt" % (epoch))
        torch.save(netD.state_dict(), workDirName + "/models/netD_%04d.pt" % (epoch))