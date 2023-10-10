import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import torchvision.utils as vutils
from utils.GAN import D
from utils.tools import getFashionMnist, getMnist

# 定义超参数
batchSize = 64
epoch = 10
DataRoot = './data'
workDirName = "./result/FashionMnistClassifier"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取FashionMnist数据集
_, fashion_mnist_trainSet = getFashionMnist(DataRoot, batchSize, True)
_, mnist_trainSet = getMnist(DataRoot, batchSize, True)

# 创建标签
fashion_mnist_labels = torch.ones(len(fashion_mnist_trainSet))  # Fashion MNIST的标签为0
mnist_labels = torch.zeros(len(mnist_trainSet))  # MNIST的标签为1

# 合并数据集和标签
combined_dataset = ConcatDataset([fashion_mnist_trainSet, mnist_trainSet])
combined_labels = torch.cat((fashion_mnist_labels, mnist_labels), 0)

# 创建数据加载器
combined_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.stack([sample[0] for sample in combined_dataset]), combined_labels),
    batch_size=64, shuffle=True, num_workers=0
)

net = D().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 训练模型
for epoch in range(epoch):  # 可以根据需要调整训练的轮次
    running_loss = 0.0
    for i, data in enumerate(combined_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.float().view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    torch.save(net.state_dict(), workDirName + "/models/net_%04d.pt" % (epoch))
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(combined_loader)}")

# 测试模型
net.eval()
for i, data in enumerate(combined_loader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    vutils.save_image(inputs, '%s/fake.png' % (workDirName + "/gen_images"), normalize=True)
    outputs = net(inputs)
    print(labels)
    print(outputs)
    break

print("Finished Training")
