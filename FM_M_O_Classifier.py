import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import torchvision.utils as vutils
from utils.Classifier import ThreeClassClassifier
from utils.tools import getFashionMnist, getMnist, createWorkDir
from torch.utils.tensorboard import SummaryWriter

# 定义超参数
batchSize = 64
epoch = 10
DataRoot = './data'
logDir = './logs/FM_M_O_Classifier'
workDirName = "./result/FM_M_O_Classifier"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构造工作目录
createWorkDir(workDirName)

# 获取FashionMnist数据集、Mnist数据集、其他数据集
_, fashion_mnist_trainSet = getFashionMnist(DataRoot, batchSize, True)
_, mnist_trainSet = getMnist(DataRoot, batchSize, True)
random_trainSet = [(torch.rand(1, 28, 28), 10) for _ in range(len(fashion_mnist_trainSet))]


# 创建标签
fashion_mnist_labels = torch.ones(len(fashion_mnist_trainSet))  # Fashion MNIST的标签为0
mnist_labels = torch.zeros(len(mnist_trainSet))  # MNIST的标签为1
random_labels = torch.full((len(mnist_labels),), 2)

# 合并数据集和标签
combined_dataset = ConcatDataset([fashion_mnist_trainSet, mnist_trainSet, random_trainSet])
combined_labels = torch.cat((fashion_mnist_labels, mnist_labels, random_labels), 0)

# 创建数据加载器
combined_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.stack([sample[0] for sample in combined_dataset]), combined_labels),
    batch_size=64, shuffle=True, num_workers=0
)

net = ThreeClassClassifier().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 实例一个writer
writer = SummaryWriter(log_dir=logDir)

# 训练模型
for epoch in range(epoch):  # 可以根据需要调整训练的轮次
    running_loss = 0.0
    for i, data in enumerate(combined_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    torch.save(net.state_dict(), workDirName + "/models/net_%04d.pt" % (epoch))
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(combined_loader)}")
    writer.add_scalar(tag='loss', scalar_value=(running_loss / len(combined_loader)))

# 测试模型
net.eval()
acc = 0
for i, data in enumerate(combined_loader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    vutils.save_image(inputs, '%s/fake1.png' % (workDirName + "/gen_images"), normalize=True)
    outputs = net(inputs)
    # print(labels)
    # print(outputs)
    # print(outputs.argmax(dim=1))
    arg = outputs.argmax(dim=1)
    acc += sum(labels == arg) / len(labels)
print(acc / len(combined_loader))
print("Finished Training")
