# from sklearn.metrics import matthews_corrcoef
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# from torch.utils.data import Dataset, DataLoader
# import torchvision.models as models
# import matplotlib.pyplot as plt
# from timm import create_model
#
# # 数据集路径
# train_file = 'MF2_ILM_INDEL_Train.csv'
# valid_file = 'MF2_ILM_INDEL_Valid.csv'
# test_file = 'MF2_ILM_INDEL_Test.csv'
#
# # 加载数据
# train_data = pd.read_csv(train_file)
# val_data = pd.read_csv(valid_file)
# test_data = pd.read_csv(test_file)
#
# # 提取特征和标签
# features = train_data.columns.drop(['Class'])  # 假设 Class 列是标签
# X_train = train_data[features].values
# X_val = val_data[features].values
# X_test = test_data[features].values
# train_data['Class'] = train_data['Class'].map({'T': 1, 'F': 0})
# val_data['Class'] = val_data['Class'].map({'T': 1, 'F': 0})
# test_data['Class'] = test_data['Class'].map({'T': 1, 'F': 0})
#
# y_train = train_data['Class'].astype(int).values
# y_val = val_data['Class'].astype(int).values
# y_test = test_data['Class'].astype(int).values
#
# # 标准化数据
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)
#
# # 将每一行数据转换为 224x224 的图像，因为 ResNet-34 期望输入尺寸为 224x224
# def to_image(tensor, size=(40, 40)):
#     # 如果特征数量大于 size[0] * size[1]，则裁剪
#     if tensor.shape[1] > size[0] * size[1]:
#         tensor = tensor[:, :size[0] * size[1]]
#     # 如果特征数量小于 size[0] * size[1]，则填充
#     elif tensor.shape[1] < size[0] * size[1]:
#         pad_size = size[0] * size[1] - tensor.shape[1]
#         tensor = torch.cat([tensor, torch.zeros(tensor.shape[0], pad_size)], dim=1)
#     return tensor.reshape(-1, 1, *size)
#
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# X_train = to_image(X_train)
# X_val = to_image(X_val)
# X_test = to_image(X_test)
#
# y_train = torch.tensor(y_train, dtype=torch.long)
# y_val = torch.tensor(y_val, dtype=torch.long)
# y_test = torch.tensor(y_test, dtype=torch.long)
#
# # 可视化函数
# def show_image(tensor):
#     tensor = tensor.squeeze(0)  # 去除批次维度
#     plt.imshow(tensor.numpy(), cmap='gray')
#     plt.axis('off')
#     plt.show()
#
# # 显示训练数据中的第一张图像
# show_image(X_train[0])
#
# # 自定义 Dataset 类
# class CustomDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
# # 创建数据加载器
# batch_size = 32
# train_dataset = CustomDataset(X_train, y_train)
# val_dataset = CustomDataset(X_val, y_val)
# test_dataset = CustomDataset(X_test, y_test)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # 定义模型
# model = models.densenet169(pretrained=True)
#
# # 修改输入层
# model.features.conv0 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)  # 修改输入通道为1
#
# # 修改输出层为二分类
# num_features = model.classifier.in_features
# model.classifier = nn.Linear(num_features, 2)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
#     best_model_wts = model.state_dict()
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch + 1}/{num_epochs}')
#         print('-' * 10)
#
#         # 训练阶段
#         model.train()
#         running_loss = 0.0
#         running_corrects = 0
#
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
#
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = running_corrects.double() / len(train_loader.dataset)
#
#         print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#
#         # 验证阶段
#         model.eval()
#         running_loss = 0.0
#         running_corrects = 0
#
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)
#
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
#
#         epoch_loss = running_loss / len(val_loader.dataset)
#         epoch_acc = running_corrects.double() / len(val_loader.dataset)
#
#         print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#
#         # 保存最好的模型
#         if epoch_acc > best_acc:
#             best_acc = epoch_acc
#             best_model_wts = model.state_dict()
#
#     print(f'Best val Acc: {best_acc:.4f}')
#
#     # 加载最好的模型权重
#     model.load_state_dict(best_model_wts)
#     return model
#
# # 训练模型
# model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30)
#
# # 测试模型
# model.eval()
# running_corrects = 0
# all_labels = []
# all_preds = []
#
# for inputs, labels in test_loader:
#     outputs = model(inputs)
#     _, preds = torch.max(outputs, 1)
#     running_corrects += torch.sum(preds == labels.data)
#     all_labels.extend(labels.cpu().numpy())
#     all_preds.extend(outputs.softmax(dim=1).cpu().detach().numpy()[:, 1])
#
# test_acc = running_corrects.double() / len(test_loader.dataset)
# test_auc = roc_auc_score(all_labels, all_preds)
# precision = precision_score(all_labels, np.round(all_preds))
# recall = recall_score(all_labels, np.round(all_preds))
# f1 = f1_score(all_labels, np.round(all_preds))
#
# # 计算MCC
# mcc = matthews_corrcoef(all_labels, np.round(all_preds))
#
# print(f'Test Acc: {test_acc:.4f} AUC: {test_auc:.4f}')
# print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}')
# print(f'MCC: {mcc:.4f}')

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
from torchvision.models import DenseNet121_Weights  # 导入DenseNet权重类
import matplotlib.pyplot as plt

# -------------------------- 自动检测GPU并设置设备 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 训练设备：{device}")
if torch.cuda.is_available():
    print(f"✅ GPU可用：{torch.cuda.get_device_name(0)}")
    print(f"📊 GPU显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  无可用GPU，将使用CPU训练（速度较慢，建议使用GPU）")

# 数据集路径
train_file = '../finalFE/ILM_SNP_Train.csv'
valid_file = '../finalFE/ILM_SNP_Valid.csv'
test_file = '../finalFE/ILM_SNP_Test.csv'

# 加载数据
train_data = pd.read_csv(train_file)
val_data = pd.read_csv(valid_file)
test_data = pd.read_csv(test_file)

# 提取特征和标签
features = train_data.columns.drop(['Class'])  # 假设Class是标签列
X_train = train_data[features].values
X_val = val_data[features].values
X_test = test_data[features].values

# 标签映射（T->1, F->0）
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


# 将特征转换为图像格式（适配DenseNet输入：224x224单通道，DenseNet经典输入尺寸）
def to_image(tensor, size=(224, 224)):
    """将特征向量转换为指定尺寸的单通道图像（DenseNet适配224x224）"""
    feature_len = size[0] * size[1]  # 图像总像素数
    if tensor.shape[1] > feature_len:
        tensor = tensor[:, :feature_len]  # 裁剪多余特征
    elif tensor.shape[1] < feature_len:
        pad_size = feature_len - tensor.shape[1]
        tensor = torch.cat([tensor, torch.zeros(tensor.shape[0], pad_size)], dim=1)  # 填充0
    return tensor.reshape(-1, 1, *size)  # 输出形状：(n_samples, 1, H, W)


# 转换为张量并重塑为图像（DenseNet适配224x224输入）
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_train = to_image(X_train, size=(224, 224))  # 关键：DenseNet默认输入224x224
X_val = to_image(X_val, size=(224, 224))
X_test = to_image(X_test, size=(224, 224))

# 标签转换为张量
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# 图像可视化函数（可选）
def show_image(tensor):
    """可视化单张图像（去除通道维度）"""
    tensor = tensor.squeeze(0)  # (1, 224, 224) -> (224, 224)
    plt.imshow(tensor.numpy(), cmap='gray')
    plt.axis('off')
    plt.show()


# 显示训练集中第一张图像（可选）
# show_image(X_train[0])

# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # 图像数据 (1, 224, 224)
        self.y = y  # 标签

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建数据加载器（DenseNet121计算量适中，batch_size=32可运行，显存不足可改为16）
batch_size = 32
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------- 定义DenseNet-121模型（适配单通道输入和二分类）--------------------------
# 加载预训练DenseNet-121（兼容新旧版本torchvision）
try:
    # 新版本：使用权重类（推荐）
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    print("✅ 成功加载DenseNet-121官方预训练权重（新版本）")
except:
    # 旧版本：使用pretrained参数
    model = models.densenet121(pretrained=True)
    print("✅ 成功加载DenseNet-121预训练权重（旧版本）")

# 1. 适配单通道输入：修改第一个卷积层（DenseNet默认in_channels=3）
# DenseNet的第一层卷积在 model.features[0]，默认参数：out_channels=64, kernel_size=7, stride=2, padding=3
model.features[0] = nn.Conv2d(
    in_channels=1,  # 单通道输入（灰度图）
    out_channels=64,  # 保持原输出通道数（DenseNet121第一层输出64）
    kernel_size=7,  # 保持原卷积核大小（7x7）
    stride=2,  # 保持原步长
    padding=3,  # 保持原填充（保证输入224x224时，输出112x112）
    bias=False
)

# 2. 适配二分类：修改最后一个全连接层（DenseNet默认输出1000类）
num_features = model.classifier.in_features  # 获取分类头输入维度（1024，DenseNet121固定）
model.classifier = nn.Linear(num_features, 2)  # 替换为二分类输出

# 将模型移至GPU/CPU
model = model.to(device)

# 损失函数和优化器（DenseNet参数较多，学习率适当降低）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)  # 学习率5e-4，平衡收敛速度和过拟合

# 可选：添加学习率调度器（基于验证集损失调整，提升稳定性）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)


# 训练函数（保持设备适配和调度器逻辑）
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
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
            # 将数据移至设备
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
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                # 将数据移至设备
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        print(f'Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}\n')

        # 学习率调度
        scheduler.step(val_epoch_loss)

        # 保存最佳模型（基于验证集准确率）
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)  # 加载最佳权重
    return model


# 训练模型（DenseNet121收敛速度中等，30个epoch足够，显存不足可减至20）
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, device=device)


# 定义最佳阈值计算方法（保持不变）
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
        # 将数据移至设备
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # 结果移回CPU用于计算
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

# 保存结果到CSV（文件名改为DenseNet对应的名称）
result_df = pd.DataFrame({
    'True_Label': all_labels,
    'Positive_Probability': all_scores,
    'Final_Prediction': final_preds,
    'F1_Optimal_Threshold': opt_thresh_f1,
})
result_df.to_csv('DenseNet121_ILM_SNP.csv', index=False)
print("\n结果已保存至 DenseNet121_ILM_SNP.csv")
