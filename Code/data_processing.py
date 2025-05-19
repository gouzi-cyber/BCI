import scipy.io as sio
import numpy as np
from scipy import signal
from scipy.signal.windows import hann
import torch
from scipy.fftpack import fft, fftfreq
import pywt
import torch.utils.data
from scipy import stats

def augment_data(data, labels):
    """增强的数据增强函数"""
    augmented_data = []
    augmented_labels = []

    for i in range(len(data)):
        original_sample = data[i]
        label = labels[i]

        # 1. 添加原始数据
        augmented_data.append(original_sample)
        augmented_labels.append(label)

        # 2. 添加多种噪声级别的高斯噪声
        noise_levels = [0.02, 0.05, 0.08]
        for noise_level in noise_levels:
            noisy_sample = original_sample + np.random.normal(0, noise_level, original_sample.shape)
            augmented_data.append(noisy_sample)
            augmented_labels.append(label)

        # 3. 随机时间偏移
        shifts = np.random.randint(-20, 20, 2)
        for shift in shifts:
            shifted_sample = np.roll(original_sample, shift, axis=-1)
            augmented_data.append(shifted_sample)
            augmented_labels.append(label)

        # 4. 随机振幅缩放
        scales = np.random.uniform(0.8, 1.2, 2)
        for scale in scales:
            scaled_sample = original_sample * scale
            augmented_data.append(scaled_sample)
            augmented_labels.append(label)

        # 5. 添加混合增强
        mixed_sample = original_sample * np.random.uniform(0.9, 1.1)
        mixed_sample = np.roll(mixed_sample, np.random.randint(-10, 10), axis=-1)
        mixed_sample += np.random.normal(0, 0.03, mixed_sample.shape)
        augmented_data.append(mixed_sample)
        augmented_labels.append(label)

    return np.array(augmented_data), np.array(augmented_labels)

def load_data(file_path):
    """加载.mat文件数据"""
    try:
        mat_data = sio.loadmat(file_path)
        data = mat_data['data']  # (8, 710, 2, 10, 12)
        # 选择干电极数据
        dry_data = data[:, :, 1, :, :]  # (8, 710, 10, 12)
        print(f'Data electrode data shape:{dry_data.shape}')

        # 转置数据使得试验维度在前
        data = np.transpose(dry_data, (2, 3, 0, 1))  # (2, 10, 12, 8, 710)
        print(f'Transpose data shape:{data.shape}')
        # 计算样本数
        samples = 10 * 12  # 区块 × 目标 = 120
        channels = 8
        time_points = 710

        # 重塑为 (120, 8, 710)
        reshaped_data = data.reshape(samples, channels, time_points)
        print(f'Final reshaped data shape:{reshaped_data.shape}')

        return reshaped_data, samples
    except Exception as e:
        print(f"加载数据出错: {str(e)}")
        raise

def create_labels(samples):
    """创建标签"""
    labels = []
    for _ in range(10):  # 区块数
        labels.extend([i % 12 for i in range(12)])  # 12个目标频率
    labels = np.array(labels)
    print(f'标签范围检查：min={labels.min()},max={labels.max()}')
    return labels

def apply_multistep_filter(data, fs=250):
    """多步滤波处理"""
    # 设计带通滤波器 (8-16Hz，针对SSVEP频率范围)
    nyquist = fs / 2
    low = 8 / nyquist
    high = 16 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')

    # 设计陷波滤波器（50Hz工频干扰）
    notch_b, notch_a = signal.iirnotch(50 / nyquist, 30)

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # 应用带通滤波
            temp_data = signal.filtfilt(b, a, data[i, j])
            # 应用陷波滤波
            filtered_data[i, j] = signal.filtfilt(notch_b, notch_a, temp_data)

    return filtered_data

def extract_enhanced_features(filtered_data, target_frequencies, fs=250):
    """增强的特征提取函数
    Args:
        filtered_data: 形状为 (samples, channels, time_points) 的滤波后数据
        target_frequencies: 目标频率列表
        fs: 采样率，默认250Hz
    Returns:
        combined_features: 形状为 (samples, channels, features) 的特征数组
    """
    samples, channels, time_points = filtered_data.shape
    print(f"Input data shape: {filtered_data.shape}")

    # 1. 频域特征提取
    freq_features = np.zeros((samples, channels, len(target_frequencies) * 3))  # 主频率、谐波和半次谐波
    phase_features = np.zeros((samples, channels, len(target_frequencies)))

    window = hann(time_points)
    freqs = fftfreq(time_points, 1 / fs)

    # 为每个目标频率找到主频率和谐波频率的索引
    target_freq_indices = []
    for freq in target_frequencies:
        main_freq_idx = np.argmin(np.abs(freqs - freq))
        harmonic_freq_idx = np.argmin(np.abs(freqs - freq * 2))  # 二次谐波
        subharmonic_freq_idx = np.argmin(np.abs(freqs - freq / 2))  # 半次谐波
        target_freq_indices.append((main_freq_idx, harmonic_freq_idx, subharmonic_freq_idx))

    # 2. 时域特征
    time_features = np.zeros((samples, channels, 8))  # 8个时域特征

    # 3. 小波特征
    wavelet_level = 5
    wavelet_features_size = sum([len(pywt.wavedec(filtered_data[0, 0], 'db4', level=wavelet_level)[i])
                                 for i in range(wavelet_level + 1)])
    wavelet_features = np.zeros((samples, channels, wavelet_features_size))

    # 特征提取主循环
    for i in range(samples):
        for j in range(channels):
            current_data = filtered_data[i, j]
            windowed_data = current_data * window

            # 1. 频域特征
            fft_result = fft(windowed_data)
            for k, (main_idx, harm_idx, subharm_idx) in enumerate(target_freq_indices):
                freq_features[i, j, k * 3] = np.abs(fft_result[main_idx])
                freq_features[i, j, k * 3 + 1] = np.abs(fft_result[harm_idx])
                freq_features[i, j, k * 3 + 2] = np.abs(fft_result[subharm_idx])
                phase_features[i, j, k] = np.angle(fft_result[main_idx])

            # 2. 时域特征
            time_features[i, j, 0] = np.mean(current_data)  # 均值
            time_features[i, j, 1] = np.std(current_data)  # 标准差
            time_features[i, j, 2] = np.sqrt(np.mean(current_data ** 2))  # RMS
            time_features[i, j, 3] = np.max(np.abs(current_data))  # 峰值
            time_features[i, j, 4] = np.percentile(current_data, 75) - np.percentile(current_data, 25)  # IQR
            time_features[i, j, 5] = stats.skew(current_data)  # 偏度
            time_features[i, j, 6] = stats.kurtosis(current_data)  # 峰度
            time_features[i, j, 7] = np.sum(np.abs(np.diff(current_data)))  # 信号复杂度

            # 3. 小波特征
            coeffs = pywt.wavedec(current_data, 'db4', level=wavelet_level)
            feature_idx = 0
            for coef in coeffs:
                coef_len = len(coef)
                wavelet_features[i, j, feature_idx:feature_idx + coef_len] = coef
                feature_idx += coef_len

    # 特征标准化
    def normalize_features(features):
        # 对每个受试者的数据分别进行标准化
        for subject_start in range(0, samples, 120):  # 每个受试者120个样本
            subject_end = subject_start + 120
            # 对特征维度进行标准化
            mean = np.mean(features[subject_start:subject_end], axis=0, keepdims=True)
            std = np.std(features[subject_start:subject_end], axis=0, keepdims=True)
            std[std == 0] = 1  # 防止除零
            features[subject_start:subject_end] = (features[subject_start:subject_end] - mean) / std
        return features

    # 对所有特征进行标准化
    freq_features = normalize_features(freq_features)
    phase_features = normalize_features(phase_features)
    time_features = normalize_features(time_features)
    wavelet_features = normalize_features(wavelet_features)

    # 组合所有特征，保持通道结构
    total_features = freq_features.shape[2] + phase_features.shape[2] + time_features.shape[2] + wavelet_features.shape[2]
    combined_features = np.zeros((samples, channels, total_features))

    # 在特征维度上连接
    current_idx = 0
    for features in [freq_features, phase_features, time_features, wavelet_features]:
        feature_size = features.shape[2]
        combined_features[:, :, current_idx:current_idx + feature_size] = features
        current_idx += feature_size

    print(f"Output feature shape: {combined_features.shape}")
    print(f"Feature breakdown:")
    print(f"- Frequency features: {freq_features.shape[2]}")
    print(f"- Phase features: {phase_features.shape[2]}")
    print(f"- Time domain features: {time_features.shape[2]}")
    print(f"- Wavelet features: {wavelet_features.shape[2]}")
    print(f"Total features per channel: {total_features}")

    return combined_features

def prepare_data_loaders(features, labels, batch_size=128, train_split=0.7, val_split=0.15):
    """优化的数据加载器函数，分为训练集、验证集和测试集"""
    # 调整特征维度为 (batch_size, channels, height, width)
    if len(features.shape) == 3:
        features = features.reshape(features.shape[0], 1, features.shape[1], features.shape[2])

    # 将数据转换为float32以减少内存使用
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # 创建数据集
    dataset = torch.utils.data.TensorDataset(features, labels)

    # 计算划分点
    total_size = len(features)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    # 确保划分比例合理，避免浮点数计算误差导致的负数或零
    if test_size <= 0:
        test_size = total_size - train_size - val_size
        if test_size <= 0:
            raise ValueError("划分比例不合理，测试集大小为0或负数，请调整 train_split 和 val_split")

    # 使用random_split划分数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    # 创建优化的数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 改为0，不使用多进程
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 改为0，不使用多进程
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 改为0，不使用多进程
        pin_memory=True
    )

    # 修改CUDA数据加载器
    class CudaDataLoader:
        def __init__(self, loader):
            self.loader = loader
            self.device = torch.device("cuda")
            # 保存原始数据集的长度
            self.dataset = loader.dataset
            self._length = len(loader)

        def __iter__(self):
            for batch in self.loader:
                inputs, targets = batch
                if len(inputs.shape) == 4:
                    inputs = inputs.float()
                yield inputs.to(self.device), targets.to(self.device)

        def __len__(self):
            return self._length

    # 包装成CUDA数据加载器
    train_loader = CudaDataLoader(train_loader)
    val_loader = CudaDataLoader(val_loader)
    test_loader = CudaDataLoader(test_loader)

    return train_loader, val_loader, test_loader

def analyze_features(features, labels, target_frequencies):
    """分析提取的特征"""
    samples, channels, num_features = features.shape
    num_freq = len(target_frequencies)

    print("\n特征分析:")
    print(f"特征形状: {features.shape}")
    print(f"频率特征数: {num_freq}")
    print(f"相位特征数: {num_freq}")

    # 分析频率特征（前半部分）和相位特征（后半部分）
    freq_features = features[:, :, :num_freq]
    phase_features = features[:, :, num_freq:]

    # 计算每个频率的平均幅值
    mean_amplitudes = np.mean(freq_features, axis=(0, 1))

    # 计算每个类别的平均特征
    unique_labels = np.unique(labels)
    class_features = {}
    for label in unique_labels:
        class_mask = labels == label
        class_features[label] = np.mean(freq_features[class_mask], axis=(0, 1))

    return {
        'mean_amplitudes': mean_amplitudes,
        'class_features': class_features
    }