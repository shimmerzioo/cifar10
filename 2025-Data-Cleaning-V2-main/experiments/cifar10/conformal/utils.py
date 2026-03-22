import torch
import pandas as pd
from pathlib import Path


def save_results(loader, prediction_sets, uncertainty_scores, output_dir):
    """
    保存评估结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存预测集合
    torch.save(prediction_sets, output_dir / "prediction_sets.pth")
    
    # 保存不确定性分数
    torch.save(uncertainty_scores, output_dir / "uncertainty_scores.pth")
    
    # 保存标签
    torch.save(loader.labels, output_dir / "labels.pth")
    
    # 创建结果DataFrame
    results = []
    for i, (label, pred_set, score) in enumerate(zip(loader.labels, prediction_sets, uncertainty_scores)):
        results.append({
            "index": i,
            "label": label.item(),
            "prediction_set": pred_set,
            "set_size": len(pred_set),
            "uncertainty_score": score.item(),
            "correct": label.item() in pred_set
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)
    
    return df


def compute_metrics(df):
    """
    计算评估指标
    """
    # 覆盖率：真实标签在预测集合中的比例
    coverage = df["correct"].mean()
    
    # 平均预测集大小
    avg_set_size = df["set_size"].mean()
    
    # 不确定性分数统计
    avg_uncertainty = df["uncertainty_score"].mean()
    std_uncertainty = df["uncertainty_score"].std()
    
    metrics = {
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "avg_uncertainty": avg_uncertainty,
        "std_uncertainty": std_uncertainty
    }
    
    return metrics