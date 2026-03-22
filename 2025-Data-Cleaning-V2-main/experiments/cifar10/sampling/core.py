import torch
import numpy as np


def hard_threshold_sampling(uncertainty_scores, threshold=0.5):
    """
    硬阈值采样：不确定性分数低于阈值的样本被选中
    """
    mask = uncertainty_scores < threshold
    # 将布尔掩码转换为概率向量
    probabilities = torch.zeros_like(uncertainty_scores, dtype=torch.float32)
    probabilities[mask] = 1.0
    # 归一化概率
    if probabilities.sum() > 0:
        probabilities = probabilities / probabilities.sum()
    return probabilities


def inverse_probability_sampling(uncertainty_scores, temperature=1.0):
    """
    反比例概率采样：不确定性越低，采样概率越高
    """
    # 确保不确定性分数在 [0, 1] 范围内
    uncertainty_scores = torch.clamp(uncertainty_scores, 0, 1)
    
    # 计算采样概率：不确定性越低，概率越高
    probabilities = 1.0 - uncertainty_scores
    probabilities = probabilities ** temperature
    
    # 归一化概率
    if probabilities.sum() > 0:
        probabilities = probabilities / probabilities.sum()
    
    return probabilities


def exponential_sampling(uncertainty_scores, temperature=1.0):
    """
    指数衰减采样：不确定性越低，采样概率呈指数增长
    """
    # 确保不确定性分数在 [0, 1] 范围内
    uncertainty_scores = torch.clamp(uncertainty_scores, 0, 1)
    
    # 计算采样概率：不确定性越低，概率越高
    probabilities = torch.exp(-temperature * uncertainty_scores)
    
    # 归一化概率
    if probabilities.sum() > 0:
        probabilities = probabilities / probabilities.sum()
    
    return probabilities


class dynamic_sampling_scheduler:
    """
    动态采样调度器：随训练过程调整采样策略
    """
    def __init__(self, initial_threshold=0.3, final_threshold=0.8, total_epochs=16):
        """
        初始化动态采样调度器
        :param initial_threshold: 初始阈值（训练初期，更严格）
        :param final_threshold: 最终阈值（训练后期，更宽松）
        :param total_epochs: 总训练轮数
        """
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.total_epochs = total_epochs
    
    def get_threshold(self, current_epoch):
        """
        根据当前轮数获取阈值
        :param current_epoch: 当前训练轮数
        :return: 当前轮数的阈值
        """
        # 线性增加阈值
        progress = current_epoch / self.total_epochs
        threshold = self.initial_threshold + (self.final_threshold - self.initial_threshold) * progress
        return threshold
    
    def get_sampling_probabilities(self, uncertainty_scores, current_epoch):
        """
        根据当前轮数获取采样概率
        :param uncertainty_scores: 不确定性分数
        :param current_epoch: 当前训练轮数
        :return: 采样概率
        """
        threshold = self.get_threshold(current_epoch)
        
        # 结合硬阈值和反比例概率采样
        # 1. 首先应用硬阈值
        mask = uncertainty_scores < threshold
        
        # 2. 对通过硬阈值的样本应用反比例概率采样
        filtered_scores = uncertainty_scores[mask]
        if len(filtered_scores) == 0:
            return torch.zeros_like(uncertainty_scores, dtype=torch.float32)
        
        probabilities = inverse_probability_sampling(filtered_scores)
        
        # 构建完整的概率向量
        full_probabilities = torch.zeros_like(uncertainty_scores, dtype=torch.float32)
        full_probabilities[mask] = probabilities
        
        # 归一化概率
        if full_probabilities.sum() > 0:
            full_probabilities = full_probabilities / full_probabilities.sum()
        
        return full_probabilities