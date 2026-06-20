import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt

# 数据集路径
train_file = '../finalFE/ILM_SNP_Train.csv'
valid_file = '../finalFE/ILM_SNP_Valid.csv'
test_file  = '../finalFE/ILM_SNP_Test.csv'

# 加载数据
train_data = pd.read_csv(train_file)
val_data = pd.read_csv(valid_file)
test_data = pd.read_csv(test_file)

# 提取特征和标签
features = train_data.columns.drop(['Class'])  # 假设 Class 列是标签
X_train = train_data[features].values
X_val = val_data[features].values
X_test = test_data[features].values
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

# 将每一行数据转换为 224x224 的图像，因为 ResNet-34 期望输入尺寸为 224x224
def to_image(tensor, size=(24, 24)):
    # 如果特征数量大于 size[0] * size[1]，则裁剪
    if tensor.shape[1] > size[0] * size[1]:
        tensor = tensor[:, :size[0] * size[1]]
    # 如果特征数量小于 size[0] * size[1]，则填充
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

# 可视化函数
def show_image(tensor):
    tensor = tensor.squeeze(0)  # 去除批次维度
    plt.imshow(tensor.numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

# 显示训练数据中的第一张图像
show_image(X_train[0])

# 自定义 Dataset 类
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

# 定义模型
model = models.resnet34(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)  # 修改输入通道数为1
num_features = model.fc.in_features  # 获取原全连接层输入维度
model.fc = nn.Linear(num_features, 2)  # 替换为二分类输出层

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
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

        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 保存最好的模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最好的模型权重
    model.load_state_dict(best_model_wts)
    return model

# 训练模型
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)

# 测试模型
model.eval()
running_corrects = 0
all_labels = []
all_preds = []

for inputs, labels in test_loader:
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels.data)
    all_labels.extend(labels.cpu().numpy())
    all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy()[:, 1])

test_acc = running_corrects.double() / len(test_loader.dataset)
test_auc = roc_auc_score(all_labels, all_preds)
precision = precision_score(all_labels, np.round(all_preds))
recall = recall_score(all_labels, np.round(all_preds))
f1 = f1_score(all_labels, np.round(all_preds))

# 计算MCC
mcc = matthews_corrcoef(all_labels, np.round(all_preds))

print(f'Test Acc: {test_acc:.4f} AUC: {test_auc:.4f}')
print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
