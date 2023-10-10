import torch.utils.data
from torch.autograd import Variable
from utils.GAN import D, G
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from utils.tools import createWorkDir

'''
更换G查看判别器的敏感性寻找阈值（未完成）
(G 生成的图片和训练过程保存的不一致？？)
1. EWC 有效 整理效果
2. 多个epoch训练的D反而不能很好的分类？
'''

# 超参数设置
batchSize = 64
OOD_threshold = 0.8
workDirName = "./result/test_Classifier"
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化生成器和判别器
netG = G().to(device)
netD = D().to(device)

# 加载预训练模型
netD.load_state_dict(torch.load('./result/FashionMnistClassifier/models/net_0009.pt'))
netG.load_state_dict(torch.load('./result/test_EWC/models/netG_3000.pt'))

# 构造工作目录
createWorkDir(workDirName)

# 开始测试
noise = Variable(torch.randn(batchSize, 100, 1, 1)).to(device)
fake = netG(noise)
pred_fake = netD(fake)
pred_fake = pred_fake.cpu().detach().numpy()
print(pred_fake)
plt.plot(pred_fake)
plt.ylim(0, 1)
plt.show()
vutils.save_image(fake.data, '%s/fake.png' % (workDirName + "/gen_images"), normalize=True)
