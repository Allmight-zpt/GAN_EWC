import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.Siamese import SiameseNetwork, ContrastiveLoss
from utils.tools import createWorkDir, getFashionMnist, getMnist
import numpy as np
'''
阈值选择参考，测试结果如下：
Accuracy: 48.30% Threshold: 0.00
Accuracy: 83.78% Threshold: 0.10
Accuracy: 89.93% Threshold: 0.20
Accuracy: 93.34% Threshold: 0.30
Accuracy: 95.57% Threshold: 0.40
Accuracy: 97.06% Threshold: 0.50
Accuracy: 98.01% Threshold: 0.60
Accuracy: 98.53% Threshold: 0.70
Accuracy: 98.82% Threshold: 0.80
Accuracy: 98.97% Threshold: 0.90
Accuracy: 99.03% Threshold: 1.00
Accuracy: 99.03% Threshold: 1.10
Accuracy: 98.99% Threshold: 1.20
Accuracy: 98.89% Threshold: 1.30
Accuracy: 98.75% Threshold: 1.40
Accuracy: 98.54% Threshold: 1.50
Accuracy: 98.21% Threshold: 1.60
Accuracy: 97.93% Threshold: 1.70
Accuracy: 97.61% Threshold: 1.80
Accuracy: 96.96% Threshold: 1.90
Accuracy: 95.31% Threshold: 2.00
Accuracy: 92.61% Threshold: 2.10
Accuracy: 91.54% Threshold: 2.20
Accuracy: 90.59% Threshold: 2.30
Accuracy: 90.08% Threshold: 2.40
Accuracy: 89.34% Threshold: 2.50
Accuracy: 89.10% Threshold: 2.60
Accuracy: 88.67% Threshold: 2.70
Accuracy: 88.46% Threshold: 2.80
Accuracy: 88.23% Threshold: 2.90
'''
# 定义超参数
batchSize = 64
epoch = 10
classNumber = 20
DataRoot = './data'
logDir = './logs/FM_M_O_Siamese'
workDirName = "./result/FM_M_O_Siamese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构造工作目录
createWorkDir(workDirName)

# 获取FashionMnist数据集、Mnist数据集
_, fashion_mnist_trainSet = getFashionMnist(DataRoot, batchSize, True)
_, mnist_trainSet = getMnist(DataRoot, batchSize, True)

# 创建数据加载器
similar_pairs = []
dissimilar_pairs = []

# 对mnist数据集进行整理
# 1. 先分类
class_dict = {i: [] for i in range(classNumber)}
for i in range(len(mnist_trainSet)):
    img, label = mnist_trainSet[i]
    class_dict[label].append(i)
for i in range(len(fashion_mnist_trainSet)):
    img, label = fashion_mnist_trainSet[i]
    class_dict[label + int(classNumber / 2)].append(i)

# 2. 再配对
for i in range(classNumber):
    if i < 10:
        dataSet_i = mnist_trainSet
    else:
        dataSet_i = fashion_mnist_trainSet
    for j in range(classNumber):
        if j < 10:
            dataSet_j = mnist_trainSet
        else:
            dataSet_j = fashion_mnist_trainSet
        if i == j:
            idx = np.random.permutation(len(class_dict[i]))
            for k, idx_value in enumerate(idx):
                similar_pairs.append(
                    (dataSet_i[class_dict[i][k]][0], dataSet_i[class_dict[i][idx_value]][0], torch.tensor(0)))
        else:
            for k in range(round(min(len(class_dict[i]), len(class_dict[j])) / classNumber)):
                dissimilar_pairs.append(
                    (dataSet_i[class_dict[i][k]][0], dataSet_j[class_dict[j][k]][0], torch.tensor(1)))

# 合并相似和不相似的图像对
all_pairs = similar_pairs + dissimilar_pairs
mnist_pairs_dataset = torch.utils.data.TensorDataset(torch.stack([sample[0] for sample in all_pairs]),
                                                     torch.stack([sample[1] for sample in all_pairs]),
                                                     torch.stack([sample[2] for sample in all_pairs]))

# 构造dataloader
mnist_pairs_loader = DataLoader(mnist_pairs_dataset, batch_size=batchSize, shuffle=True)

# 创建孪生网络模型
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 实例一个writer
writer = SummaryWriter(log_dir=logDir)

# 开始训练
for e in range(epoch):
    running_loss = 0
    for i, (input1, input2, label) in enumerate(mnist_pairs_loader, 0):
        input1, input2, label = input1.to(device), input2.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    torch.save(model.state_dict(), workDirName + "/models/net_%04d.pt" % e)
    print(f"Epoch [{e + 1}/{epoch}] Loss: {running_loss / len(mnist_pairs_loader):.4f}")
    writer.add_scalar(tag='loss', scalar_value=(running_loss / len(mnist_pairs_loader)))
print("Finished Training")

# 开始测试
model.eval()
model.load_state_dict(torch.load('./result/FM_M_O_Siamese/models/net_0009.pt'))
maxThreshold = 3
thresholdGap = 0.1
correct = [0 for i in range(int(maxThreshold / thresholdGap))]
total = [0 for i in range(int(maxThreshold / thresholdGap))]

for i, (input1, input2, label) in enumerate(mnist_pairs_loader, 0):
    input1, input2, label = input1.to(device), input2.to(device), label.to(device)
    output1, output2 = model(input1, input2)
    similarity_scores = torch.norm(output1 - output2, dim=1)  # 计算欧氏距离
    predictions = 0
    for th in range(int(maxThreshold / thresholdGap)):
        threshold = th * thresholdGap
        predictions = (similarity_scores > threshold).float()  # 根据阈值进行分类
        correct[th] += (predictions == label).sum().item()
        total[th] += label.size(0)

    # same_idx = torch.nonzero(label == 0).view(-1)
    # diff_idx = torch.nonzero(label == 1).view(-1)
    # same_score = torch.mean(similarity_scores[same_idx])
    # diff_score = torch.mean(similarity_scores[diff_idx])
    # writer.add_scalar(tag='same_score', scalar_value=same_score)
    # writer.add_scalar(tag='diff_score', scalar_value=diff_score)

for th in range(int(maxThreshold / thresholdGap)):
    threshold = th * thresholdGap
    accuracy = 100 * correct[th] / total[th]
    print(f"Accuracy: {accuracy:.2f}% Threshold: {threshold:.2f}")
print("Finished Testing")

