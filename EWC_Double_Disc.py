import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.EWC import EWC
from utils.GAN import D, G
from utils.tools import getMnist, getFashionMnist, sample_batch_index, createWorkDir, preProcess, batchToOne
from torch.utils.tensorboard import SummaryWriter

'''
1. 使用OOD-ID分类器进行判断，分类器置信度底的图象记为data_new，即为生成ID数据的尝试，置信度高的图象记为data_old 即OOD数据
2. 使用判别器对 data_new 进行判断，作为pred_fake
3. 使用判别器对 data_old 进行判断，作为pred_real的一部分

(分类器换成孪生网络？？)
'''

# 超参数设置
batchSize = 64
imageSize = 28
num_worker = 10
num_epochs = 50001
OOD_threshold = 0.8
losses = []
workDirName = "./result/EWC_Double_Disc"
logDir = './logs/EWC_Double_Disc'
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
net_Classifier = D().to(device)

# 加载预训练模型
netD.load_state_dict(torch.load('pretrain_weight/FashionMnist_netD_20000.pt'))
netG.load_state_dict(torch.load('pretrain_weight/FashionMnist_netG_20000.pt'))
net_Classifier.load_state_dict(torch.load('./result/FashionMnistClassifier/models/net_0009.pt'))

# 初始化优化器
optimizerD = optim.RMSprop(netD.parameters(), lr=0.0002, alpha=0.9)
optimizerG = optim.RMSprop(netG.parameters(), lr=0.0002, alpha=0.9)

# 构造工作目录
createWorkDir(workDirName)

# 创建EWC实例
ewc = EWC(netG, netD, device, FashionMnist_dataloader)

# 创建SummaryWriter对象
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
        pred_fake_Classifier = net_Classifier(fake)
        netD.zero_grad()
        '''
        使用分类器，获取分类置信度，当置信度达到一定条件才使用分类结果
        '''
        # loss_real = 0
        # loss_fake = 0
        # pred_fake_min = pred_fake_Classifier.min()
        # pred_fake_max = pred_fake_Classifier.max()
        # 进行归一化
        # pred_fake_Classifier = (pred_fake_Classifier - pred_fake_min) / (pred_fake_max - pred_fake_min)
        ''' 画图 '''
        # writer.add_scalar(tag='pred_fake_max', scalar_value=pred_fake_max)
        # writer.add_scalar(tag='pred_fake_min', scalar_value=pred_fake_min)
        # writer.add_scalar(tag='var', scalar_value=pred_fake_Classifier.var())
        # writer.add_scalar(tag='mean', scalar_value=pred_fake_Classifier.mean())
        # fake_clone = fake.clone()
        # fake_clone = fake_clone.cpu()
        # allImage = batchToOne(fake_clone, 8, 8)
        # writer.add_image(tag='gen_fake', img_tensor=allImage)
        ''' 画图 '''
        # if False:
        #     big = torch.topk(pred_fake_Classifier, k=int(10))
        #     small = torch.topk(pred_fake_Classifier, k=int(54), largest=False)
        #     pred_fake_big = pred_fake[big[1]]
        #     pred_fake_small = pred_fake[small[1]]
        #     loss_real = torch.mean(pred_fake_big)
        #     loss_fake = torch.mean(pred_fake_small)
        # else:
        loss_fake = torch.mean(pred_fake)
        loss_real = torch.mean(pred_real)

        ewc_loss = ewc.penalty(netD, gen=False)
        errD = -loss_real + loss_fake + ewc_loss * 100
        errD.backward(retain_graph=True)
        optimizerD.step()

        for parm in netD.parameters():
            parm.data.clamp_(-0.01, 0.01)

    loss_fake_sum = loss_fake
    errG = - loss_fake_sum + ewc.penalty(netG) * 10000
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