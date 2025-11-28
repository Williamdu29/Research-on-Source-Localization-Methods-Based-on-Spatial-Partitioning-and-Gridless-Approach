import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LSN(nn.Module):
    def __init__(self, num_sources):
        super(LSN, self).__init__()
        
        # 输入层：接受 (2, 7, 7) 的输入，表示协方差矩阵的实部和虚部
        self.conv1 = nn.Conv2d(2, 64, kernel_size=2, stride=1, padding=0)   # 卷积层1，7x7 -> 6x6
        self.bn1 = nn.BatchNorm2d(64)  # 批处理归一化层1

        # 第一个深度可分离卷积层，6x6 -> 5x5
        self.conv2_dw = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0, groups=64)  # 逐通道卷积
        self.conv2_pw = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # 逐点卷积
        self.bn2 = nn.BatchNorm2d(64)  # 批处理归一化层2

        # 第二个深度可分离卷积层，5x5 -> 4x4
        self.conv3_dw = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0, groups=64)  # 逐通道卷积
        self.conv3_pw = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # 逐点卷积
        self.bn3 = nn.BatchNorm2d(64)  # 批处理归一化层3

        # 展平操作
        self.flatten = nn.Flatten()
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)  # 输入为 64*4*4，输出为 1024
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        
        self.dropout = nn.Dropout(p=0.5)  # Dropout 层，防止过拟合

        # 输出层，表示信源数量的概率
        self.fc_out = nn.Linear(64, num_sources) # 输出为信源数量的类别数


    def forward(self, x):
        # 卷积层 + 激活函数 + 批处理归一化
        x = F.relu(self.bn1(self.conv1(x)))  # ReLU 激活
        x = F.relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))
        x = F.relu(self.bn3(self.conv3_pw(self.conv3_dw(x))))
        
        # 展平操作
        x = self.flatten(x)
        
        # 全连接层
        x = F.relu(self.fc1(x)) # 全连接后经过 ReLU 激活使用非线性变换
        x = self.dropout(x)     # Dropout 层
        x = F.relu(self.fc2(x))
        x = self.dropout(x)     # Dropout 层
        x = F.relu(self.fc3(x))
        x = self.dropout(x)     # Dropout 层
        x = F.relu(self.fc4(x))
        x = self.dropout(x)     # Dropout 层
        
        # 输出层 + Softmax
        out = self.fc_out(x)
        return F.softmax(out, dim=-1)  # Softmax 输出信源数量的概率

# 假设信源最大数量为 6
model = LSN(num_sources=6)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 示例训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    
    # 假设我们有一个批次的数据：数据 x 和标签 y
    # x 是 (batch_size, 2, 7, 7) 形状的张量，表示协方差矩阵的实部和虚部
    # y 是 (batch_size) 形状的张量，表示信源数量的标签
    x = torch.randn(32, 2, 7, 7)  # 假设 batch_size=32，输入数据大小为 7x7
    y = torch.randint(0, 6, (32,))  # 随机生成 0 到 5 之间的标签
    
    # 前向传播
    outputs = model(x)
    
    # 计算损失
    loss = criterion(outputs, y)
    
    # 后向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 假设测试数据，使用训练好的模型预测
model.eval()  # 设置为评估模式
test_x = torch.randn(10, 2, 7, 7)  # 10 个测试样本
predictions = model(test_x)
print(predictions)  # 输出信源数量的预测概率
