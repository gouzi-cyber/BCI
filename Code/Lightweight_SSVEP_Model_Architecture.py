import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


# 轻量化模型（使用标准卷积）
class LightweightSSVEP_CNN(nn.Module):
    def __init__(self, num_channels=8, num_classes=12, num_features=None):
        super(LightweightSSVEP_CNN, self).__init__()

        # 1. 主干特征提取网络（使用标准卷积，通道数保持4和8）
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 7), stride=1, padding=(0, 3)),  # 输入1通道，输出4通道
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(4, 8, kernel_size=(num_channels, 1), stride=1),  # 输入4通道，输出8通道
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout2d(0.3),
        )

        # 2. 空间注意力模块（保持8通道）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 3. 通道注意力模块（保持8通道，确保瓶neck层至少为1）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8, max(1, 8 // 16), kernel_size=1),  # 确保输出通道数至少为1（8 // 16 = 0.5 → 1）
            nn.ReLU(),
            nn.Conv2d(max(1, 8 // 16), 8, kernel_size=1),
            nn.Sigmoid()
        )

        # 4. SE-Block（保持8通道，确保瓶neck层至少为1）
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8, max(1, 8 // 16), kernel_size=1),  # 确保输出通道数至少为1（8 // 16 = 0.5 → 1）
            nn.ReLU(),
            nn.Conv2d(max(1, 8 // 16), 8, kernel_size=1),
            nn.Sigmoid()
        )

        # 5. 获取展平后的特征维度
        self.flatten_features = self._get_flatten_features(num_channels, num_features)

        # 6. 分类头（保持不变，基于新特征维度）
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_features, 256),  # 减少中间层，直接到256
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # 直接输出
        )

        # 7. L2正则化参数
        self.l2_lambda = 0.01

        # 8. 初始化权重
        self._initialize_weights()

    def _get_flatten_features(self, num_channels, num_features):
        x = torch.randn(1, 1, num_channels, num_features)
        x = self.backbone(x)
        x = x * self.spatial_attention(x)
        x = x * self.channel_attention(x)
        x = x * self.se_block(x)
        x = x.view(x.size(0), -1)
        return x.shape[1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.view(x.size(0), 1, 8, -1)

        # 1. 主干网络特征提取
        x = self.backbone(x)

        # 2. 注意力机制（带残差连接）
        identity = x
        spatial_attn = self.spatial_attention(x)
        x = x * spatial_attn
        channel_attn = self.channel_attention(x)
        x = x * channel_attn
        se_attn = self.se_block(x)
        x = x * se_attn
        x = x + identity

        # 3. 分类
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 增强模型（保持不变）
class EnhancedSSVEP_CNN(nn.Module):
    def __init__(self, num_channels=8, num_classes=12, num_features=None):
        super(EnhancedSSVEP_CNN, self).__init__()

        # 主干特征提取网络
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, kernel_size=(num_channels, 1), stride=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Dropout2d(0.3)
        )

        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256 // 16, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # SE-Block
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256 // 16, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # 获取展平后的特征维度
        self.flatten_features = self._get_flatten_features(num_channels, num_features)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_features, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _get_flatten_features(self, num_channels, num_features):
        x = torch.randn(1, 1, num_channels, num_features)
        x = self.backbone(x)
        x = x * self.spatial_attention(x)
        x = x * self.channel_attention(x)
        x = x * self.se_block(x)
        x = x.view(x.size(0), -1)
        return x.shape[1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.view(x.size(0), 1, 8, -1)

        x = self.backbone(x)
        identity = x
        spatial_attn = self.spatial_attention(x)
        x = x * spatial_attn
        channel_attn = self.channel_attention(x)
        x = x * channel_attn
        se_attn = self.se_block(x)
        x = x * se_attn
        x = x + identity
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        # 学生模型的交叉熵损失
        student_loss = self.criterion(student_logits, targets)
        # 知识蒸馏损失（KL散度）
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=1)
        kl_loss = nn.KLDivLoss(reduction='batchmean')(soft_predictions, soft_targets) * (self.temperature ** 2)
        # 总损失
        total_loss = self.alpha * student_loss + (1 - self.alpha) * kl_loss
        return total_loss


class SmoothEarlyStopping:
    def __init__(self, patience=15, window_size=3, min_delta=1e-4):
        self.patience = patience
        self.window_size = window_size
        self.min_delta = min_delta
        self.best_avg_score = None
        self.counter = 0
        self.best_weights = None
        self.scores = []

    def __call__(self, score, model):
        self.scores.append(score)
        if len(self.scores) >= self.window_size:
            avg_score = sum(self.scores[-self.window_size:]) / self.window_size
            if self.best_avg_score is None or avg_score > self.best_avg_score + self.min_delta:
                self.best_avg_score = avg_score
                self.counter = 0
                self.best_weights = model.state_dict().copy()
                return False
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False


def train_with_distillation(train_loader, val_loader, student_model, teacher_model, num_features, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device).eval()  # 教师模型固定，不更新参数

    # 加载预训练权重，使用新路径
    checkpoint = torch.load(
        'C:\\Users\\PC\\Desktop\\learning exercise\\BCI\\Code\\FPGA\\Pruning_1\\best_lightweight_model.pth',
        weights_only=True)
    state_dict = checkpoint['model_state_dict']  # 从检查点中提取模型权重

    # 打印状态字典的键，检查结构差异
    print("Loaded state_dict keys:", list(state_dict.keys()))
    print("Model state_dict keys:", list(student_model.state_dict().keys()))

    # 尝试加载权重，忽略不匹配的键
    try:
        student_model.load_state_dict(state_dict, strict=False)  # strict=False 容忍缺失和额外键
    except RuntimeError as e:
        print(f"警告：加载权重时出现错误: {e}")
        # 手动映射权重（如果需要）
        model_state = student_model.state_dict()
        for name, param in state_dict.items():
            if name in model_state:
                model_state[name].copy_(param)
        student_model.load_state_dict(model_state)

    # 加载教师模型权重，使用新路径
    teacher_checkpoint = torch.load(
        'C:\\Users\\PC\\Desktop\\learning exercise\\BCI\\Code\\FPGA\\Pruning_1\\best_model.pth', weights_only=True)
    teacher_state_dict = teacher_checkpoint['model_state_dict']
    teacher_model.load_state_dict(teacher_state_dict, strict=True)

    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=0.0005,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )

    criterion = DistillationLoss()
    early_stopping = SmoothEarlyStopping(patience=15, window_size=3)
    accumulation_steps = 2

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    for epoch in range(num_epochs):
        student_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)  # 教师模型预测
            student_outputs = student_model(inputs)  # 学生模型预测
            loss = criterion(student_outputs, teacher_outputs, labels) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * accumulation_steps
            _, predicted = student_outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch + 1} | Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.0 * train_correct / train_total:.2f}%')

        # 验证阶段
        student_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                student_outputs = student_model(inputs)
                teacher_outputs = teacher_model(inputs)  # 使用教师模型生成软标签
                loss = criterion(student_outputs, teacher_outputs, labels)
                val_loss += loss.item()
                _, predicted = student_outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f} | Training Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%')

        if early_stopping(val_acc, student_model):
            print('\nEarly stopping triggered')
            student_model.load_state_dict(early_stopping.best_weights)
            break

        if val_acc > max(history['val_acc'][:-1], default=0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, 'best_distilled_lightweight_model.pth')

    return student_model, history


def test_lightweight_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = 0
    total = 0
    predictions = []
    probabilities = []
    confusion_matrix = torch.zeros(12, 12)  # 假设12个类别

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 确保输入维度正确
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新混淆矩阵
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    accuracy = 100. * correct / total
    class_accuracies = confusion_matrix.diag() / confusion_matrix.sum(1)

    return (
        np.array(predictions),
        np.array(probabilities),
        accuracy,
        confusion_matrix,
        class_accuracies
    )