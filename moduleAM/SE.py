from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ==========================================
# 1. 定义注意力机制模块 (SE & CBAM)
# ==========================================

# --- SE (Squeeze-and-Excitation) 模块 ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # 通道加权


# --- CBAM (Convolutional Block Attention Module) 模块 ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAMLayer, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ==========================================
# 2. 定义带有注意力机制的 ResNet Block
# ==========================================

# --- 基础 Block ---
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# --- SE-ResNet Block ---
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # 加入SE注意力

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# --- CBAM-ResNet Block ---
class CBAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(CBAMBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAMLayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)  # 加入CBAM注意力

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ==========================================
# 3. 组装完整的 ResNet34 网络
# ==========================================
class ResNet_Custom(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet_Custom, self).__init__()
        self.inplanes = 64

        # 修改输入层：适配单通道(1) 24x24 图像
        # 原ResNet是 7x7, stride=2, 这里改为 5x5, stride=1 以保留小尺寸特征
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 24 -> 12

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 12 -> 6
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 6 -> 3
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=2)  # 3 -> 2 (实际上由于kernel=3, stride=2, padding=1, 3->2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 实例化函数
def se_resnet34(**kwargs):
    return ResNet_Custom(SEBasicBlock, [3, 4, 6, 3], **kwargs)


def cbam_resnet34(**kwargs):
    return ResNet_Custom(CBAMBasicBlock, [3, 4, 6, 3], **kwargs)


# ==========================================
# 4. 数据处理与训练流程 (保持原样，仅替换模型)
# ==========================================

# 数据集路径 (请根据实际情况修改)
train_file = '../finalFE/ILM_SNP_Train.csv'
valid_file = '../finalFE/ILM_SNP_Valid.csv'
test_file = '../finalFE/ILM_SNP_Test.csv'

# 加载数据
train_data = pd.read_csv(train_file)
val_data = pd.read_csv(valid_file)
test_data = pd.read_csv(test_file)

# 提取特征和标签
features = train_data.columns.drop(['Class'])
X_train = train_data[features].values
X_val = val_data[features].values
X_test = test_data[features].values

# 标签映射
train_data['Class'] = train_data['Class'].map({'T': 1, 'F': 0})
val_data['Class'] = val_data['Class'].map({'T': 1, 'F': 0})
test_data['Class'] = test_data['Class'].map({'T': 1, 'F': 0})

y_train = train_data['Class'].astype(int).values
y_val = val_data['Class'].astype(int).values
y_test = test_data['Class'].astype(int).values

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# 将每一行数据转换为 24x24 的图像
def to_image(tensor, size=(24, 24)):
    if tensor.shape[1] > size[0] * size[1]:
        tensor = tensor[:, :size[0] * size[1]]
    elif tensor.shape[1] < size[0] * size[1]:
        pad_size = size[0] * size[1] - tensor.shape[1]
        tensor = torch.cat([tensor, torch.zeros(tensor.shape[0], pad_size)], dim=1)
    return tensor.reshape(-1, 1, *size)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

X_train = to_image(X_train)
X_val = to_image(X_val)
X_test = to_image(X_test)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# 可视化 (可选)
def show_image(tensor):
    plt.imshow(tensor.squeeze().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()


# show_image(X_train[0])

# 自定义 Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建数据加载器
batch_size = 32
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 5. 模型选择：在这里切换 SE 或 CBAM
# ==========================================

# 选择 1: SE-ResNet34
model = se_resnet34(num_classes=2)

# 选择 2: CBAM-ResNet34 (取消下面这行的注释即可)
# model = cbam_resnet34(num_classes=2)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# 训练模型 (保持原样)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        print(f'Val   Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    print(f'Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# 开始训练
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)

# 测试模型
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

all_labels = []
all_preds = []
all_preds_proba = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probs = outputs.softmax(dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_preds_proba.extend(probs.cpu().numpy())

# 计算指标
test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
test_auc = roc_auc_score(all_labels, all_preds_proba)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

print('-' * 20)
print(f'Test Acc: {test_acc:.4f} AUC: {test_auc:.4f}')
print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')
print(f'MCC: {mcc:.4f}')