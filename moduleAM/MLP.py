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

# 数据集路径（保持相同）
train_file = '../finalMF/MF2_ILM_INDEL_Train.csv'
valid_file = '../finalMF/MF2_ILM_INDEL_Valid.csv'
test_file = '../finalMF/MF2_ILM_INDEL_Test.csv'

# 加载数据（保持相同）
train_data = pd.read_csv(train_file)
val_data = pd.read_csv(valid_file)
test_data = pd.read_csv(test_file)

# 提取特征和标签（保持相同）
features = train_data.columns.drop(['Class'])
X_train = train_data[features].values
X_val = val_data[features].values
X_test = test_data[features].values

# 标签编码（保持相同）
train_data['Class'] = train_data['Class'].map({'T': 1, 'F': 0})
val_data['Class'] = val_data['Class'].map({'T': 1, 'F': 0})
test_data['Class'] = test_data['Class'].map({'T': 1, 'F': 0})

y_train = train_data['Class'].astype(int).values
y_val = val_data['Class'].astype(int).values
y_test = test_data['Class'].astype(int).values

# 标准化数据（保持相同）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 转换为Tensor（删除图像转换）
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# 自定义Dataset类（保持相同）
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建数据加载器（保持相同）
batch_size = 32
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)


# 初始化模型
input_size = X_train.shape[1]
model = MLP(input_size)

# 定义损失函数和优化器（保持相同）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数（保持相同）
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

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# 训练模型（保持相同）
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)

# 测试模型（保持相同）
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
mcc = matthews_corrcoef(all_labels, np.round(all_preds))

print(f'Test Acc: {test_acc:.4f} AUC: {test_auc:.4f}')
print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}')
print(f'MCC: {mcc:.4f}')