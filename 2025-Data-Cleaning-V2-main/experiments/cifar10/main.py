import click
import sys
from pathlib import Path
from airbench import train94, train96, infer, evaluate, CifarLoader
import pandas as pd
import torch
import numpy as np
import copy
import os
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal import split_data, train_calibrate_model, generate_prediction_sets, compute_uncertainty_scores, save_results, compute_metrics
from sampling import dynamic_sampling_scheduler, calculate_sampling_probabilities, apply_sampling

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir / "src"))
from utils import set_seed, cross_validation_loop, GPUTaskScheduler
from cp_utils import CLF_NCSCORE_MAP

DATASET = "CIFAR-10"


def _get_cifar_loader():
    loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=1024,
        aug={"flip": True, "translate": 2},
        altflip=True,
    )
    return loader


def _train_cifar_model(loader, target, verbose=2):
    if target == "94":
        net, result = train94(
            loader, epochs=16, label_smoothing=0, verbose=verbose
        )  # train this network without label smoothing to get a better confidence signal
    elif target == "96":
        net, result = train96(loader, verbose=verbose)
    return net, result


@click.group()
def cli():
    pass


@cli.command()
@click.option("--seed", type=int, default=40)
@click.option("--target", type=click.Choice(["94", "96"]), default="94")
def baseline(seed, target):
    """
    Target 94 => 94.5%
    Target 96 => 95.93%
    """
    algo = sys._getframe().f_code.co_name

    def setup(algo):
        dataset = DATASET + f"-{target}"
        output_dir = current_dir / "results" / algo
        output_dir.mkdir(parents=True, exist_ok=True)
        identifier = f"{dataset}_{algo}_{seed}"
        result_fn = output_dir / f"{identifier}.csv"
        mask_fn = output_dir / f"{identifier}.pkl"
        set_seed(seed)
        return result_fn, mask_fn

    result_fn, mask_fn = setup(algo)
    loader = _get_cifar_loader()
    net, result = _train_cifar_model(loader, target, verbose=2)
    mask = torch.ones(len(loader.labels), dtype=torch.bool)
    torch.save(mask, mask_fn)
    pd.DataFrame(result).to_csv(result_fn)


def _get_algo1_identifier(target, algo, confname, k, alpha, threshold, seed):
    return f"{DATASET}-{target}_{algo}_{confname}_{k}_{alpha}_{threshold}_{seed}"


@cli.command()
@click.option("--force", is_flag=True, default=False)
@click.option("--target", type=click.Choice(["94", "96"]), default="94")
@click.option("--confname", type=click.Choice(["CLF-Logits"]), default="CLF-Logits")
@click.option("--k", type=int, default=5)
@click.option("--alpha", type=float, default=0.1)
@click.option("--threshold", type=float, default=2)
@click.option("--save-model", is_flag=True, default=False)
@click.option("--seed", type=int, default=40)
@click.option("--verbose", type=int, default=2)
def algo1(force, target, confname, k, alpha, threshold, save_model, seed, verbose):
    """
    target: 94 or 96 target accuracy
    force: if True, re-run the experiment even if the result and mask already exist
    """
    algo = sys._getframe().f_code.co_name
    dataset = DATASET + f"-{target}"
    output_dir = current_dir / "results" / algo
    output_dir.mkdir(parents=True, exist_ok=True)

    identifier = _get_algo1_identifier(
        target, algo, confname, k, alpha, threshold, seed
    )
    result_fn = output_dir / f"{identifier}.csv"
    mask_fn = output_dir / f"{identifier}.pkl"
    if not force and result_fn.exists() and mask_fn.exists():
        return

    set_seed(seed)
    loader = _get_cifar_loader()
    loader0 = copy.deepcopy(loader)

    n = len(loader0.labels)
    mask = torch.BoolTensor(n, device="cpu")
    all_logits = torch.zeros(
        size=(n, len(loader0.classes)),
        dtype=torch.float16,
        device="cpu",
    )
    for train_mask, calib_mask, test_mask in cross_validation_loop(k, n):
        train_loader = copy.deepcopy(loader0)
        train_loader.images = train_loader.images[train_mask]
        train_loader.labels = train_loader.labels[train_mask]
        net, _ = _train_cifar_model(train_loader, target, verbose=verbose)

        calib_loader = copy.deepcopy(loader0)
        calib_loader.images = calib_loader.images[calib_mask]
        calib_loader.labels = calib_loader.labels[calib_mask]
        logits = infer(net, calib_loader).softmax(1)

        # Get conformal prediction
        gt = logits[np.arange(len(calib_loader.labels)), calib_loader.labels]
        gt = gt.to(torch.float32)
        q = torch.quantile(gt, alpha)

        test_loader = copy.deepcopy(loader0)
        test_loader.images = test_loader.images[test_mask]
        test_loader.labels = test_loader.labels[test_mask]
        test_logits = infer(net, test_loader).softmax(1)
        all_logits[test_mask] = test_logits.to("cpu")

        if confname == "CLF-Logits":
            mask[test_mask] = ((test_logits > q).sum(axis=1) <= threshold).to("cpu")
        else:
            raise NotImplementedError(f"NCScore {confname} not implemented")

    mask = mask.to(loader0.images.device)
    loader0.images = loader0.images[mask]
    loader0.labels = loader0.labels[mask]

    _, result = _train_cifar_model(loader0, target, verbose=verbose)
    pd.DataFrame(result).to_csv(result_fn)
    torch.save(mask.to("cpu"), mask_fn)
    torch.save(all_logits, output_dir / f"{identifier}_logits.pth")
    if save_model:
        torch.save(net, output_dir / f"{identifier}.pth")


@cli.command()
@click.option("--force", is_flag=True, default=False)
@click.option("--identifiers-only", is_flag=True, default=False)
def algo1_exp(force, identifiers_only):
    """
    target: 94 or 96 target accuracy
    force: if True, re-run the experiment even if the result and mask already exist
    parallel: if True, run the experiment in parallel
    identifiers_only: if True, only print the identifiers of the experiments
    """
    algo = sys._getframe().f_code.co_name.split("_")[0]

    def _loop():
        for K in [5]:
            for confname in ["CLF-Logits"]:
                for alpha in [0.2, 0.15, 0.1, 0.05]:
                    for threshold in [2, 3]:
                        for seed in [40, 41, 42, 43, 44]:
                            for target in ["94", "96"]:
                                # for target in ["96"]:
                                yield K, confname, alpha, float(threshold), seed, target

    def _get_identifier():
        for K, confname, alpha, threshold, seed, target in _loop():
            yield _get_algo1_identifier(
                target,
                algo,
                confname,
                K,
                alpha,
                threshold,
                seed,
            )

    def _get_params():
        for K, confname, alpha, threshold, seed, target in _loop():
            yield [
                "--confname",
                confname,
                "--threshold",
                str(threshold),
                "--k",
                str(K),
                "--alpha",
                str(alpha),
                "--seed",
                str(seed),
                "--target",
                target,
            ]

    if identifiers_only:
        print(list(_get_identifier()))
        return

    cmd = ["python", "main.py", algo]
    if force:
        cmd += ["--force"]
    scheduler = GPUTaskScheduler(gpu_min_memory_mb=5000, sleep=10)
    scheduler.start([cmd + params for params in _get_params()], cwd=current_dir)


@cli.command()
@click.option("--force", is_flag=True, default=False)
@click.option("--target", type=click.Choice(["94", "96"]), default="94")
@click.option("--confname", type=click.Choice(["CLF-Conf"]), default="CLF-Conf")
@click.option("--confthreshold", type=float, default=0.1)
@click.option("--seed", type=int, default=40)
@click.option("--verbose", type=int, default=2)
def algo2(force, target, confname, confthreshold, seed, verbose):
    # setups
    algo = sys._getframe().f_code.co_name
    dataset = DATASET + f"-{target}"
    output_dir = current_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    identifier = f"{dataset}_{algo}_{confname}_{confthreshold}_{seed}"
    result_fn = output_dir / f"{identifier}.csv"
    mask_fn = output_dir / f"{identifier}.pkl"
    if not force and result_fn.exists() and mask_fn.exists():
        return

    set_seed(seed)
    loader = _get_cifar_loader()
    net, _ = _train_cifar_model(copy.deepcopy(loader), target, verbose=verbose)
    logits = infer(net, loader)
    conf = logits.log_softmax(1).amax(1)
    mask = conf < conf.float().quantile(confthreshold)

    loader.images = loader.images[mask]
    loader.labels = loader.labels[mask]

    try:
        _, result = _train_cifar_model(loader, target, verbose=verbose)
        pd.DataFrame(result).to_csv(result_fn)
        torch.save(mask, mask_fn)
    except Exception as e:
        result_fn.touch(exist_ok=True)
        with open(result_fn, "w") as f:
            f.write(str(e))
        torch.save(mask, mask_fn)


@cli.command()
@click.option("--force", is_flag=True, default=False)
@click.option("--identifiers-only", is_flag=True, default=False)
def algo2_exp(force, identifiers_only):
    """
    target: 94 or 96 target accuracy
    force: if True, re-run the experiment even if the result and mask already exist
    parallel: if True, run the experiment in parallel
    identifiers_only: if True, only print the identifiers of the experiments
    """
    algo = sys._getframe().f_code.co_name.split("_")[0]

    def _get_params():
        for confname in ["CLF-Conf"]:
            for confthreshold in [0.2, 0.15, 0.1, 0.05, 0.01]:
                for seed in [3, 4, 5]:
                    for target in ["94", "96"]:
                        yield [
                            "--confname",
                            confname,
                            "--confthreshold",
                            str(confthreshold),
                            "--seed",
                            str(seed),
                            "--target",
                            target,
                        ]

    cmd = ["python", "main.py", algo]
    if force:
        cmd += ["--force"]
    if identifiers_only:
        cmd += ["--identifiers-only"]
    scheduler = GPUTaskScheduler(gpu_min_memory_mb=4000, sleep=10)
    scheduler.start([cmd + params for params in _get_params()], cwd=current_dir)


########################################################################
@cli.command()
@click.option("--target", type=click.Choice(["94", "96"]), default="94")
def analyze_predictive_uncertainty_against_accuracy(target):
    """
    For a trained model, log its predictive uncertainty on test set and prediction.
    """
    algo = sys._getframe().f_code.co_name
    output_dir = current_dir / "results" / algo / f"target{target}"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = _get_cifar_loader()
    device = loader.images.device
    n = len(loader.labels)
    K = len(loader.classes)

    clf_logits_ncscore = torch.zeros_like(loader.labels, dtype=torch.float16)
    clf_cumul_ncscore = torch.zeros_like(loader.labels, dtype=torch.float16)
    all_logits = torch.zeros(size=(n, K), dtype=torch.float16, device=device)
    for train_mask, calib_mask, test_mask in cross_validation_loop(5, n):
        train_mask = train_mask | calib_mask
        train_loader = copy.deepcopy(loader)
        train_loader.images = train_loader.images[train_mask]
        train_loader.labels = train_loader.labels[train_mask]
        net, _ = _train_cifar_model(train_loader, target, verbose=2)

        test_loader = copy.deepcopy(loader)
        test_loader.images = test_loader.images[test_mask]
        test_loader.labels = test_loader.labels[test_mask]
        logits = infer(net, test_loader)

        clf_logits_ncscore[test_mask] = CLF_NCSCORE_MAP["CLF-Logits"](
            logits, test_loader.labels
        )
        clf_cumul_ncscore[test_mask] = CLF_NCSCORE_MAP["CLF-Cumulative"](
            logits, test_loader.labels
        )
        all_logits[test_mask] = logits

    logits = all_logits.type(torch.float32)

    torch.save(clf_logits_ncscore, output_dir / f"clf_logits_ncscore.pth")
    torch.save(clf_cumul_ncscore, output_dir / f"clf_cumul_ncscore.pth")
    torch.save(logits, output_dir / f"logits.pth")
    torch.save(loader.labels, output_dir / f"labels.pth")


class CifarLoaderWithScore(CifarLoader):
    def __init__(
        self,
        path,
        train=True,
        batch_size=500,
        aug=None,
        drop_last=None,
        shuffle=None,
        altflip=False,
        scorepath=None,
    ):
        super().__init__(path, train, batch_size, aug, drop_last, shuffle, altflip)
        self.score = torch.load(scorepath)

    def get_indices(self, n, device):
        # Use the score function for weighted sampling and update the __len__
        weights = self.score / self.score.sum()
        indices = torch.multinomial(weights, n, replacement=True)

        indices = (torch.randperm if self.shuffle else torch.arange)(n, device=device)

        return indices

    def __len__(self):
        from math import ceil

        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):
        from airbench.utils import batch_flip_lr, batch_crop, batch_cutout
        import torch.nn.functional as F

        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.altflip:
                if self.epoch % 2 == 1:
                    images = images.flip(-1)
            else:
                images = batch_flip_lr(images)
        if self.aug.get("cutout", 0) > 0:
            images = batch_cutout(images, self.aug["cutout"])

        self.epoch += 1

        indices = self.get_indices(len(images), images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


@cli.command()
@click.option("--target", type=click.Choice(["94", "96"]), default="94")
@click.option("--scorepath", type=str, default=None)
@click.option("--outputpath", type=str, default=None)
@click.option("--seed", type=int, default=40)
@click.option("--verbose", type=int, default=2)
def score_weighted_training(target, scorepath, outputpath, seed, verbose):
    """
    python main.py score_weighted_training --target 94 --scorepath results/analyze_predictive_uncertainty_against_accuracy/target94/clf_logits_ncscore.pth --outputpath results/score_weighted_training/target94
    """
    # output_dir = current_dir / "results" / algo
    # output_dir.mkdir(parents=True, exist_ok=True)
    # identifier = _get_algo1_identifier(
    #     target, algo, confname, k, alpha, threshold, seed
    # )
    # result_fn = output_dir / f"{identifier}.csv"
    # mask_fn = output_dir / f"{identifier}.pkl"
    # if not force and result_fn.exists() and mask_fn.exists():
    #     return
    set_seed(seed)
    loader = CifarLoaderWithScore(
        "cifar10",
        train=True,
        batch_size=1024,
        aug={"flip": True, "translate": 2},
        altflip=True,
        scorepath=scorepath,
    )
    print(loader.score.shape, loader.score[:10])
    net, result = _train_cifar_model(loader, target, verbose=verbose)

    pass


@cli.command()
@click.option("--target", type=click.Choice(["94", "96"]), default="94")
def exp3_exp(target):
    path = (
        current_dir
        / "results"
        / "analyze_predictive_uncertainty_against_accuracy"
        / f"target{target}"
    )


# 新命令
@cli.command()
@click.option("--target", type=click.Choice(["94", "96"]), default="94")
@click.option("--alpha", type=float, default=0.1)
@click.option("--output-dir", type=str, default="results/conformal")
@click.option("--seed", type=int, default=40)
def conformal_evaluate(target, alpha, output_dir, seed):
    """
    使用 Split Conformal Prediction 评估数据
    """
    set_seed(seed)
    
    # 加载原始数据
    loader = CifarLoader('cifar10', train=True, batch_size=1024, aug=None)
    
    # 划分数据
    train_loader, calib_loader, test_loader = split_data(loader)
    
    # 训练和校准模型
    net, calib_scores = train_calibrate_model(train_loader, calib_loader, target)
    
    # 生成预测集合
    prediction_sets, threshold = generate_prediction_sets(net, test_loader, calib_scores, alpha)
    
    # 计算不确定性分数
    uncertainty_scores = compute_uncertainty_scores(prediction_sets)
    
    # 保存结果
    df = save_results(test_loader, prediction_sets, uncertainty_scores, output_dir)
    
    # 计算指标
    metrics = compute_metrics(df)
    
    # 打印结果
    print("=== Conformal Prediction Evaluation Results ===")
    print(f"Target Coverage: {(1-alpha)*100:.1f}%")
    print(f"Alpha: {alpha}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Actual Coverage: {metrics['coverage']:.4f} ({metrics['coverage']*100:.2f}%)")
    print(f"Average Set Size: {metrics['avg_set_size']:.2f}")
    print(f"Average Uncertainty: {metrics['avg_uncertainty']:.4f}")
    print(f"Std Uncertainty: {metrics['std_uncertainty']:.4f}")
    print(f"Results saved to: {output_dir}")


# sampling_experiment 函数
@cli.command()
@click.option("--target", type=click.Choice(["94", "96"]), default="94")
@click.option("--sampling-method", type=click.Choice(["hard_threshold", "inverse_probability", "exponential"]), default="inverse_probability")
@click.option("--threshold", type=float, default=0.5)
@click.option("--temperature", type=float, default=1.0)
@click.option("--sample-size", type=int, default=None)
@click.option("--use-dynamic", is_flag=True, default=False)
@click.option("--initial-threshold", type=float, default=0.3)
@click.option("--final-threshold", type=float, default=0.6)  # 降低最终阈值
@click.option("--output-dir", type=str, default="results/sampling")
@click.option("--seed", type=int, default=40)
@click.option("--max-epochs", type=int, default=100)  # 添加最大训练轮次参数
@click.option("--patience", type=int, default=5)  # 添加早停耐心值参数
def sampling_experiment(target, sampling_method, threshold, temperature, sample_size, use_dynamic, initial_threshold, final_threshold, output_dir, seed, max_epochs, patience):
    """
    测试不同采样策略的效果
    """
    set_seed(seed)
    
    # 加载原始数据
    loader = CifarLoader('cifar10', train=True, batch_size=1024, aug={"flip": True, "translate": 2})
    
    # 加载不确定性分数（假设已通过 conformal_evaluate 生成）
    uncertainty_scores = torch.load("results/conformal/uncertainty_scores.pth")
    
    # 固定采样率为60%，确保公平对比
    if sample_size is None:
        sample_size = int(len(loader.labels) * 0.6)
    
    # 计算采样概率
    if use_dynamic:
        # 使用动态采样调度器
        scheduler = dynamic_sampling_scheduler(
            initial_threshold=initial_threshold,
            final_threshold=final_threshold,
            total_epochs=max_epochs
        )
        
        # 模拟训练过程
        results = []
        best_val_acc = 0.0
        best_net = None
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # 获取当前轮数的采样概率
            sampling_probabilities = scheduler.get_sampling_probabilities(uncertainty_scores, epoch)
            
            # 应用采样
            sampled_loader = apply_sampling(loader, uncertainty_scores, sampling_probabilities, sample_size)
            
            # 训练模型
            if target == "94":
                net, result = train94(sampled_loader, epochs=1, label_smoothing=0, verbose=1 if epoch % 5 == 0 else 0)
            elif target == "96":
                net, result = train96(sampled_loader, verbose=1 if epoch % 5 == 0 else 0)
            
            # 记录结果
            result_dict = {
                'epoch': epoch,
                'sampled_size': len(sampled_loader.labels),
                'threshold': scheduler.get_threshold(epoch)
            }
            # 合并原始结果
            if isinstance(result, dict):
                result_dict.update(result)
                # 记录最佳验证准确率和模型
                if 'val_acc' in result_dict:
                    if result_dict['val_acc'] > best_val_acc:
                        best_val_acc = result_dict['val_acc']
                        best_net = net
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
            results.append(result_dict)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Sampled size: {len(sampled_loader.labels)}, Threshold: {scheduler.get_threshold(epoch):.4f}")
        
        # 使用最佳模型进行最终评估
        print("\nEvaluating best model...")
        test_loader = CifarLoader('cifar10', train=False, batch_size=1024)
        if best_net is not None:
            from airbench import infer, evaluate
            # 进行推理
            logits = infer(best_net, test_loader)
            # 计算准确率
            predictions = logits.argmax(dim=1)
            correct = (predictions == test_loader.labels).sum().item()
            val_acc = correct / len(test_loader.labels)
            print(f"Final evaluation accuracy: {val_acc:.4f}")
            # 将评估结果添加到最后一个结果中
            if len(results) > 0:
                results[-1]['val_acc'] = val_acc
                # 计算泛化差距
                if 'train_acc' in results[-1]:
                    generalization_gap = results[-1]['train_acc'] - val_acc
                    results[-1]['generalization_gap'] = generalization_gap
                    print(f"Generalization Gap: {generalization_gap:.4f}")
        else:
            print("No best model found")
    else:
        # 使用固定采样策略
        if sampling_method == 'hard_threshold':
            sampling_probabilities = calculate_sampling_probabilities(uncertainty_scores, sampling_method, threshold=threshold)
        else:
            sampling_probabilities = calculate_sampling_probabilities(uncertainty_scores, sampling_method, temperature=temperature)
        
        # 应用采样
        sampled_loader = apply_sampling(loader, uncertainty_scores, sampling_probabilities, sample_size)
        
        # 训练模型
        best_val_acc = 0.0
        best_net = None
        patience_counter = 0
        results = []
        
        for epoch in range(max_epochs):
            if target == "94":
                net, result = train94(sampled_loader, epochs=1, label_smoothing=0, verbose=1 if epoch % 5 == 0 else 0)
            elif target == "96":
                net, result = train96(sampled_loader, verbose=1 if epoch % 5 == 0 else 0)
            
            # 记录结果
            result_dict = {
                'epoch': epoch,
                'sampled_size': len(sampled_loader.labels),
                'sampling_method': sampling_method
            }
            # 合并原始结果
            if isinstance(result, dict):
                result_dict.update(result)
                # 记录最佳验证准确率和模型
                if 'val_acc' in result_dict:
                    if result_dict['val_acc'] > best_val_acc:
                        best_val_acc = result_dict['val_acc']
                        best_net = net
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
            results.append(result_dict)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Sampled size: {len(sampled_loader.labels)}")
        
        # 使用最佳模型进行最终评估
        if best_net is not None:
            test_loader = CifarLoader('cifar10', train=False, batch_size=1024)
            from airbench import infer, evaluate
            # 进行推理
            logits = infer(best_net, test_loader)
            # 计算准确率
            predictions = logits.argmax(dim=1)
            correct = (predictions == test_loader.labels).sum().item()
            val_acc = correct / len(test_loader.labels)
            print(f"Final evaluation accuracy: {val_acc:.4f}")
            # 将评估结果添加到最后一个结果中
            if len(results) > 0:
                results[-1]['val_acc'] = val_acc
                # 计算泛化差距
                if 'train_acc' in results[-1]:
                    generalization_gap = results[-1]['train_acc'] - val_acc
                    results[-1]['generalization_gap'] = generalization_gap
                    print(f"Generalization Gap: {generalization_gap:.4f}")
    
    # 保存结果
    import pandas as pd
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / f"{sampling_method}_results.csv", index=False)
    
    # 打印结果
    print("\n=== Sampling Experiment Results ===")
    if use_dynamic:
        print(f"Sampling Method: dynamic")
    else:
        print(f"Sampling Method: {sampling_method}")
    print(f"Target: {target}%")
    print(f"Sampled Size: {len(sampled_loader.labels)}")
    print(f"Max Epochs: {max_epochs}")
    print(f"Patience: {patience}")
    
    # 安全获取 val_acc 和 generalization_gap
    val_acc = 'N/A'
    generalization_gap = 'N/A'
    training_time = 'N/A'
    if len(results) > 0:
        # 遍历所有结果，找到包含 val_acc 的结果
        for result in reversed(results):
            if isinstance(result, dict):
                if 'val_acc' in result:
                    val_acc = result['val_acc']
                if 'generalization_gap' in result:
                    generalization_gap = result['generalization_gap']
                if 'total_time_seconds' in result:
                    training_time = result['total_time_seconds']
                if val_acc != 'N/A' and generalization_gap != 'N/A' and training_time != 'N/A':
                    break
    
    # 打印准确率、泛化差距和训练时间
    if val_acc != 'N/A':
        try:
            print(f"Final Accuracy: {float(val_acc):.4f}")
        except:
            print(f"Final Accuracy: {val_acc}")
    else:
        print(f"Final Accuracy: {val_acc}")
    
    if generalization_gap != 'N/A':
        try:
            print(f"Generalization Gap: {float(generalization_gap):.4f}")
        except:
            print(f"Generalization Gap: {generalization_gap}")
    else:
        print(f"Generalization Gap: {generalization_gap}")
    
    if training_time != 'N/A':
        try:
            print(f"Training Time: {float(training_time):.2f} seconds")
        except:
            print(f"Training Time: {training_time}")
    else:
        print(f"Training Time: {training_time}")
    
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    cli()
