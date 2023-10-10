import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.EWC import EWC
from utils.GAN import D, G
from utils.tools import getMnist, getFashionMnist, sample_batch_index, createWorkDir, preProcess

'''
1. 使用OOD判别器进行判断，OOD判别器置信度底的图象记为data_new，即为生成新数据的尝试，置信度高的图象记为data_old
2. 使用判别器对 data_new 进行判断，作为pred_fake
3. 使用判别器对 data_old 进行判断，作为pred_real的一部分
'''

# 超参数设置
batchSize = 64
imageSize = 28
num_worker = 10
num_epochs = 50001
OOD_threshold = 0.8
losses = []
workDirName = "./result/EWC_Double_Disc"
MnistDataRoot = './data'
FashionMnistDataRoot = './data'
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
dataloader = getMnist(MnistDataRoot, batchSize)
FashionMnist_dataloader = getFashionMnist(FashionMnistDataRoot, batchSize)

# 数据预处理
images = preProcess(dataloader)

# 初始化生成器和判别器
netG = G().to(device)
netD = D().to(device)
netD_OOD = D().to(device)

# 加载预训练模型
netD.load_state_dict(torch.load('./pretrain_weight/netD_20000.pt'))
netG.load_state_dict(torch.load('./pretrain_weight/netG_20000.pt'))
netD_OOD.load_state_dict(torch.load('./pretrain_weight/netD_20000.pt'))

# 初始化优化器
optimizerD = optim.RMSprop(netD.parameters(), lr=0.0002, alpha=0.9)
optimizerG = optim.RMSprop(netG.parameters(), lr=0.0002, alpha=0.9)

# 构造工作目录
createWorkDir(workDirName)

# 创建EWC实例
ewc = EWC(netG, netD, device, FashionMnist_dataloader)

# 开始训练
for epoch in range(num_epochs):
    for _ in range(3):
        batch_idx = sample_batch_index(images.shape[0], batchSize)
        data = images[batch_idx]
        real = Variable(data).to(device)
        noise = Variable(torch.randn(real.size()[0], 100, 1, 1)).to(device)
        fake = netG(noise)
        pred_real = netD(real)
        pred_fake_OOD = netD_OOD(fake)
        netD.zero_grad()
        loss_real = torch.mean(pred_real)
        loss_fake = torch.mean(pred_fake)
        # 待完善
        ewc_loss = ewc.penalty(netD, gen=False)

        errD = -loss_real + loss_fake + ewc_loss * 0
        errD.backward(retain_graph = True)
        optimizerD.step()

        for parm in netD.parameters():
            parm.data.clamp_(-0.01, 0.01)

    loss_fake_sum = loss_fake
    errG = - loss_fake_sum + ewc.penalty(netG) * (10000/(epoch + 1))
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