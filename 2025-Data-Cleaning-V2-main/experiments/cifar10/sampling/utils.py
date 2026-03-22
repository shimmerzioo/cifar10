import torch
from .core import hard_threshold_sampling, inverse_probability_sampling, exponential_sampling

def calculate_sampling_probabilities(uncertainty_scores, sampling_method, **kwargs):
    """
    计算采样概率
    :param uncertainty_scores: 不确定性分数
    :param sampling_method: 采样方法名称
    :param kwargs: 采样方法的参数
    :return: 采样概率
    """
    if sampling_method == 'hard_threshold':
        return hard_threshold_sampling(uncertainty_scores, **kwargs)
    elif sampling_method == 'inverse_probability':
        return inverse_probability_sampling(uncertainty_scores, **kwargs)
    elif sampling_method == 'exponential':
        return exponential_sampling(uncertainty_scores, **kwargs)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

def apply_sampling(loader, uncertainty_scores, sampling_probabilities, sample_size=None):
    """
    应用采样策略，返回采样后的加载器
    :param loader: 原始数据加载器
    :param uncertainty_scores: 不确定性分数
    :param sampling_probabilities: 采样概率（概率向量）
    :param sample_size: 采样大小（如果为None，则使用原始大小）
    :return: 采样后的数据加载器
    """
    # 最小样本数，确保有足够的数据进行训练
    min_samples = 10000  # 例如，至少保留10000个样本
    
    # 确保采样概率是浮点型张量
    if not isinstance(sampling_probabilities, torch.Tensor):
        sampling_probabilities = torch.tensor(sampling_probabilities, dtype=torch.float32)
    
    # 确保概率向量的长度与数据一致
    if len(sampling_probabilities) != len(loader.labels):
        raise ValueError(f"Sampling probabilities length ({len(sampling_probabilities)}) must match data length ({len(loader.labels)})")
    
    # 检查概率和是否大于0
    if sampling_probabilities.sum() <= 0:
        # 如果所有概率都是0，使用均匀采样
        sampling_probabilities = torch.ones_like(sampling_probabilities) / len(sampling_probabilities)
    else:
        # 确保概率和为1
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
    
    # 确定采样大小
    if sample_size is None:
        sample_size = len(loader.labels)
    
    # 确保采样大小不小于最小值
    sample_size = max(sample_size, min_samples)
    
    # 根据概率采样
    try:
        indices = torch.multinomial(sampling_probabilities, sample_size, replacement=True)
    except Exception as e:
        # 如果采样失败，使用均匀采样
        print(f"Sampling failed: {e}, using uniform sampling")
        indices = torch.randperm(len(loader.labels))[:sample_size]
    
    # 创建新的加载器
    sampled_loader = type(loader)(
        'cifar10', 
        train=True, 
        batch_size=loader.batch_size, 
        aug=loader.aug,
        altflip=getattr(loader, 'altflip', False)
    )
    sampled_loader.images = loader.images[indices]
    sampled_loader.labels = loader.labels[indices]
    
    return sampled_loader