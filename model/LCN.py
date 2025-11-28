import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LCN(nn.Module):
    def __init__(self, N):
        super(LCN, self).__init__()
        self.N = N  # 信源最大数量
        out_dim = 3*N - 5
        
        # 卷积层：深度可分离卷积（DSC）
        self.conv1 = nn.Conv2d(2, 128, kernel_size=2, stride=1, padding=0) # 普通卷积， 7x7 -> 6x6
        self.bn1 = nn.BatchNorm2d(128)  # 批处理归一化
        
        # 第一个深度可分离卷积层
        self.conv2_dw = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0, groups=128)  # 逐通道卷积
        self.conv2_pw = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)  # 逐点卷积
        # 6x6 -> 5x5
        self.bn2 = nn.BatchNorm2d(128)  # 批处理归一化
        
        # 第二个深度可分离卷积层
        self.conv3_dw = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0, groups=128)  # 逐通道卷积
        self.conv3_pw = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)  # 逐点卷积
        # 5x5 -> 4x4
        self.bn3 = nn.BatchNorm2d(128)  # 批处理归一化

        # 全局平均池化（GAP）
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层

        # 输出层
        self.fc_out = nn.Linear(128, out_dim)  # 输出层
    
    def forward(self, x):
        # 卷积层 + 批处理归一化 + 激活函数（LeakyReLU）
        x = F.leaky_relu(self.bn1(self.conv1(x)))  #卷积 + 批处理归一化 + LeakyReLU
        x = F.leaky_relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))  # 深度可分离卷积 + 批处理归一化 + LeakyReLU
        x = F.leaky_relu(self.bn3(self.conv3_pw(self.conv3_dw(x))))  # 深度可分离卷积 + 批处理归一化 + LeakyReLU
        
        # 全局平均池化
        x = self.global_pool(x)  # 对特征图进行全局平均池化
        x = torch.flatten(x, 1)  # 展平操作，准备输入到全连接层
        
        # 输出层
        out = self.fc_out(x)  # 输出信号源的概率分布
        x = F.leaky_relu(x)   # activation at output
    
        return x
    

# 假设信源最大数量为 6
model = LCN(N=6)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 示例训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    
    # 假设我们有一个批次的数据：数据 x 和标签 y
    # x 是 (batch_size, 2, 7, 7) 形状的张量，表示协方差矩阵的实部和虚部
    # y 是 (batch_size) 形状的张量，表示信号源的关键特征标签
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

# 假设测试数据，使用训练好的模型进行预测
model.eval()  # 设置为评估模式
test_x = torch.randn(10, 2, 7, 7)  # 10 个测试样本
predictions = model(test_x)
print(predictions)  # 输出信号源的预测概率
