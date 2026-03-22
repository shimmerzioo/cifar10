import torch
import numpy as np
import sys
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent.parent
airbench_dir = root_dir / "cifar10-airbench"
sys.path.insert(0, str(airbench_dir))

from airbench import train94, train96, infer, CifarLoader

def split_data(loader, train_ratio=0.7, calib_ratio=0.15):
    """
    划分数据为训练集、校准集和测试集
    """
    n = len(loader.labels)
    indices = torch.randperm(n)
    
    train_size = int(n * train_ratio)
    calib_size = int(n * calib_ratio)
    
    train_indices = indices[:train_size]
    calib_indices = indices[train_size:train_size+calib_size]
    test_indices = indices[train_size+calib_size:]
    
    # 创建三个数据集的加载器
    train_loader = CifarLoader('cifar10', train=True, batch_size=1024, aug={"flip": True, "translate": 2})
    train_loader.images = loader.images[train_indices]
    train_loader.labels = loader.labels[train_indices]
    
    calib_loader = CifarLoader('cifar10', train=True, batch_size=1024, aug=None)
    calib_loader.images = loader.images[calib_indices]
    calib_loader.labels = loader.labels[calib_indices]
    
    test_loader = CifarLoader('cifar10', train=True, batch_size=1024, aug=None)
    test_loader.images = loader.images[test_indices]
    test_loader.labels = loader.labels[test_indices]
    
    return train_loader, calib_loader, test_loader


def train_calibrate_model(train_loader, calib_loader, target="94"):
    """
    训练基础模型并使用校准集进行校准
    """
    # 训练基础模型
    if target == "94":
        net, _ = train94(train_loader, epochs=16, label_smoothing=0)
    elif target == "96":
        net, _ = train96(train_loader, verbose=2)
    else:
        raise ValueError(f"Invalid target: {target}")
    
    # 在校准集上获取预测
    calib_logits = infer(net, calib_loader)
    calib_probs = calib_logits.softmax(dim=1)
    
    # 计算非一致性分数（使用置信度的负对数）
    calib_labels = calib_loader.labels
    calib_scores = -torch.log(calib_probs[torch.arange(len(calib_labels)), calib_labels])
    calib_scores = calib_scores.to(torch.float32)
    
    return net, calib_scores


def generate_prediction_sets(net, loader, calib_scores, alpha=0.1):
    """
    生成预测集合
    """
    # 确保 calib_scores 是 float 类型
    calib_scores = calib_scores.to(torch.float32)
    # 计算阈值（正确的共形预测阈值计算）
    n = len(calib_scores)
    threshold = torch.quantile(calib_scores, (n + 1) * (1 - alpha) / n)
    
    # 获取预测
    logits = infer(net, loader)
    probs = logits.softmax(dim=1)
    
    # 生成预测集合
    prediction_sets = []
    for i in range(len(loader.labels)):
        scores = -torch.log(probs[i])
        pred_set = torch.where(scores <= threshold)[0].tolist()
        # 确保预测集合不为空
        if not pred_set:
            # 如果为空，添加概率最高的类别
            pred_set = [torch.argmax(probs[i]).item()]
        prediction_sets.append(pred_set)
    
    return prediction_sets, threshold


def compute_uncertainty_scores(prediction_sets):
    """
    计算不确定性分数
    """
    # 预测集大小作为不确定性分数（集合越大，不确定性越高）
    uncertainty_scores = [len(pred_set) for pred_set in prediction_sets]
    
    # 归一化到 [0, 1]
    max_size = max(uncertainty_scores)
    if max_size > 0:
        uncertainty_scores = [score / max_size for score in uncertainty_scores]
    
    return torch.tensor(uncertainty_scores)