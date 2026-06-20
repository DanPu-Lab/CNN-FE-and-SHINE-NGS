from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, precision_recall_curve, auc
)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import ResNet34_Weights

# 数据集路径
train_file = '../finalMF/MF2_ION_INDEL_Train.csv'
valid_file = '../finalMF/MF2_ION_INDEL_Valid.csv'
test_file = '../finalMF/MF2_ION_INDEL_Test.csv'

# 加载数据
train_data = pd.read_csv(train_file)
val_data = pd.read_csv(valid_file)
test_data = pd.read_csv(test_file)

# 数据预处理
features = train_data.columns.drop(['Class'])
for df in [train_data, val_data, test_data]:
    df['Class'] = df['Class'].map({'T': 1, 'F': 0})

X_train = train_data[features].values
X_val = val_data[features].values
X_test = test_data[features].values
y_train = train_data['Class'].astype(int).values
y_val = val_data['Class'].astype(int).values
y_test = test_data['Class'].astype(int).values

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# 转换为图像格式函数
def to_image(tensor, size=(24, 24)):
    if tensor.shape[1] > size[0] * size[1]:
        tensor = tensor[:, :size[0] * size[1]]
    elif tensor.shape[1] < size[0] * size[1]:
        pad_size = size[0] * size[1] - tensor.shape[1]
        tensor = torch.cat([tensor, torch.zeros(tensor.shape[0], pad_size)], dim=1)
    return tensor.reshape(-1, 1, *size)


# 转换为张量并调整形状
X_train = to_image(torch.tensor(X_train, dtype=torch.float32))
X_val = to_image(torch.tensor(X_val, dtype=torch.float32))
X_test = to_image(torch.tensor(X_test, dtype=torch.float32))

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建DataLoader
batch_size = 32
train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CustomDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(CustomDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# 定义模型
model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
model.fc = nn.Linear(model.fc.in_features, 2)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_val_acc = val_correct / len(val_loader.dataset)
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    return model


# 训练配置
model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001)
)

# 测试阶段
model.eval()
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# 检查标签分布
print("\n测试集标签分布:")
print(f"类别0: {(all_labels == 0).sum()} 样本")
print(f"类别1: {(all_labels == 1).sum()} 样本")

# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()


# 安全除法函数
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else float('nan')


# 计算AUPRC
try:
    precision_pr, recall_pr, _ = precision_recall_curve(all_labels, all_probs)
    auprc = auc(recall_pr, precision_pr)
except ValueError as e:
    print(f"\n警告：无法计算AUPRC - {e}")
    auprc = float('nan')

# 完整指标计算
metrics = {
    "Accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
    "AUC": roc_auc_score(all_labels, all_probs),
    "AUPRC": auprc,
    "Precision": safe_divide(tp, tp + fp),
    "Recall (Sensitivity)": safe_divide(tp, tp + fn),
    "Specificity": safe_divide(tn, tn + fp),
    "F1-Score": safe_divide(2 * tp, 2 * tp + fp + fn),
    "F1-Minor": f1_score(all_labels, all_preds,
                         pos_label=0 if (all_labels == 0).sum() < (all_labels == 1).sum() else 1),
    "BACC": (safe_divide(tp, tp + fn) + safe_divide(tn, tn + fp)) / 2,
    "NPV": safe_divide(tn, tn + fn),
    "OFO (FOR)": safe_divide(fn, fn + tn),  # 新增 OFO 计算
    "MCC": matthews_corrcoef(all_labels, all_preds),
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn
}

# 格式化输出
max_key_length = max(len(k) for k in metrics.keys())
print("\n模型性能指标:")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k:{max_key_length}} : {v:.4f}")
    else:
        print(f"{k:{max_key_length}} : {v}")

print("\n混淆矩阵:")
print(f"{'':<15}{'Predicted Negative':<20}{'Predicted Positive':<20}")
print(f"{'Actual Negative':<15}{tn:<20}{fp:<20}")
print(f"{'Actual Positive':<15}{fn:<20}{tp:<20}")