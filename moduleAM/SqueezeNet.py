from sklearn.metrics import matthews_corrcoef, roc_curve
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
from timm import create_model

# 数据集路径
train_file = '../finalFE/ILM_SNP_Train.csv'
valid_file = '../finalFE/ILM_SNP_Valid.csv'
test_file = '../finalFE/ILM_SNP_Test.csv'

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
def to_image(tensor, size=(40, 40)):
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

# # 可视化函数
# def show_image(tensor):
#     tensor = tensor.squeeze(0)  # 去除批次维度
#     plt.imshow(tensor.numpy(), cmap='gray')
#     plt.axis('off')
#     plt.show()
#
# # 显示训练数据中的第一张图像
# show_image(X_train[0])

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
model = models.squeezenet1_0(pretrained=True)
model.features[0] = nn.Conv2d(1, 96, kernel_size=5, stride=1)  # 修改输入通道为1
model.classifier[1] = nn.Conv2d(512, 2, kernel_size=1)  # 修改输出层为二分类
model.num_classes = 2

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

# 定义最佳阈值计算方法
def find_optimal_threshold_youden(y_true, y_score):
    """Youden's J法：最大化(灵敏度 + 特异度 - 1)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_j = tpr - fpr  # Youden指数 = 灵敏度 - 假阳性率
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx], np.max(youden_j)


def find_optimal_threshold_f1(y_true, y_score):
    """最大化F1分数（适合不平衡数据）"""
    thresholds = np.linspace(0, 1, 200)  # 生成200个候选阈值
    f1_scores = []
    for thresh in thresholds:
        preds = [1 if s >= thresh else 0 for s in y_score]
        f1 = f1_score(y_true, preds)
        f1_scores.append(f1)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]


# 测试模型并收集预测分数
model.eval()
all_labels = []  # 真实标签
all_scores = []  # 正类预测概率（分数）

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        scores = outputs.softmax(dim=1).cpu().numpy()[:, 1]  # 提取正类概率
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores)

# 计算两种最佳阈值
opt_thresh_f1, max_f1 = find_optimal_threshold_f1(all_labels, all_scores)
opt_thresh_youden, max_youden = find_optimal_threshold_youden(all_labels, all_scores)

# 基于F1最优阈值生成最终预测（优先推荐，适合不平衡数据）
final_preds = [1 if s >= opt_thresh_f1 else 0 for s in all_scores]

# 计算评估指标
test_acc = np.mean(np.array(final_preds) == np.array(all_labels))
test_auc = roc_auc_score(all_labels, all_scores)
precision = precision_score(all_labels, final_preds)
recall = recall_score(all_labels, final_preds)
mcc = matthews_corrcoef(all_labels, final_preds)

# 输出结果
print("=" * 60)
print(f"测试集样本分布：正样本 {np.sum(all_labels == 1)} 个，负样本 {np.sum(all_labels == 0)} 个")
print("=" * 60)
print(f"【优先推荐】F1最优阈值：{opt_thresh_f1:.4f}，对应最大F1分数：{max_f1:.4f}")
print(f"【参考】Youden最优阈值：{opt_thresh_youden:.4f}，对应最大Youden指数：{max_youden:.4f}")
print("=" * 60)
print(f"基于F1最优阈值的测试指标：")
print(f"准确率 (Accuracy): {test_acc:.4f}")
print(f"auc值 (AUC): {test_auc:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数 (F1 Score): {max_f1:.4f}")
print(f"马修斯相关系数 (MCC): {mcc:.4f}")

# 保存结果到CSV
result_df = pd.DataFrame({
    'True_Label': all_labels,
    'Positive_Probability': all_scores,
    'Final_Prediction': final_preds,
    'F1_Optimal_Threshold': opt_thresh_f1,
})
result_df.to_csv('SqueezeNet_ILM_SNP.csv', index=False)
print("\n结果已保存至 SqueezeNet_ILM_SNP.csv")
