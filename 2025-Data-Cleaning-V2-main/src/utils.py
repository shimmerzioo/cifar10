from click import progressbar
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import subprocess
from typing import Optional, Tuple, List

import torch
import pynvml


def find_gpu_with_largest_memory_nvml(min_memory_mb=0):
    """
    NVML version to find GPU with largest available memory.
    Returns:
        - best_gpu: Index of GPU with largest free memory (or None if no GPU meets min_memory_mb)
        - max_memory: Maximum free memory in MB (0 if no GPU found)
        - device_indices: List of all checked GPU indices
        - free_memory_list: List of free memory (MB) for each GPU
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        raise RuntimeError("NVML initialization failed (check NVIDIA driver)")

    # Handle CUDA_VISIBLE_DEVICES
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        device_indices = [int(x.strip()) for x in visible_devices.split(",")]
    else:
        device_indices = list(range(pynvml.nvmlDeviceGetCount()))

    best_gpu = None
    max_memory = 0
    free_memory_list = []

    for index in device_indices:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory_mb = mem_info.free / (1024 * 1024)  # Convert to MB
            free_memory_list.append(free_memory_mb)

            if free_memory_mb > min_memory_mb and free_memory_mb > max_memory:
                max_memory = free_memory_mb
                best_gpu = index
        except pynvml.NVMLError:
            free_memory_list.append(0)  # Mark as unavailable

    pynvml.nvmlShutdown()
    return best_gpu, max_memory, device_indices, free_memory_list


def set_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_command(command, cwd=None, env=None):
    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
    )
    return p


class GPUTaskScheduler:
    """1 Node, multi-GPUs"""

    def __init__(self, gpu_min_memory_mb=4000, sleep=10):
        self.gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        self.gpu_min_memory_mb = gpu_min_memory_mb
        self.sleep = sleep

    def _update_progressbar(
        self, progressbar, processes, device_indices=None, free_memory_list=None
    ):
        finished = [p for p in processes if p.poll() is not None]
        running = [p for p in processes if p.poll() is None]
        if progressbar.n < len(finished):
            progressbar.update(len(finished) - progressbar.n)

        if device_indices is not None and free_memory_list is not None:
            progressbar.set_postfix(
                device_indices=device_indices,
                free_memory_list=free_memory_list,
            )
        progressbar.set_description(
            f"Tasks: {len(running)} running, {len(finished)} finished"
        )

    def _next_gpu(self):
        gpu_id = None
        while gpu_id is None:
            gpu_id, max_memory, device_indices, free_memory_list = (
                find_gpu_with_largest_memory_nvml(self.gpu_min_memory_mb)
            )
            if gpu_id is None:
                time.sleep(self.sleep)
            if max_memory <= 2 * self.gpu_min_memory_mb:
                time.sleep(self.sleep)
        return gpu_id, device_indices, free_memory_list

    def _output_error_msg(self, processes):
        finished = [p for p in processes if p.poll() is not None]
        for p in finished:
            if p.poll() is not None and p.returncode != 0:
                print(p.stderr.read())

    def start(self, commands, cwd=None):
        progressbar = tqdm(total=len(commands), desc="Running tasks")
        processes = []
        while commands:
            command = commands.pop(0)
            gpu_id, device_indices, free_memory_list = self._next_gpu()

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            p = run_command(command, cwd=cwd, env=env)
            processes.append(p)
            time.sleep(self.sleep)

            # Check how many process have finished and how many are running
            self._update_progressbar(
                progressbar, processes, device_indices, free_memory_list
            )

            # if finshed process contain error message, print the error message
            self._output_error_msg(processes)

        finished = [p for p in processes if p.poll() is not None]
        while len(finished) < len(processes):
            time.sleep(self.sleep)
            self._update_progressbar(progressbar, processes)
            # if finshed process contain error message, print the error message
            self._output_error_msg(processes)
            finished = [p for p in processes if p.poll() is not None]


def cross_validation_loop(K, n):
    uniform_rand = torch.rand(n)

    for k in range(K - 1):
        q1, q2 = k / K, (k + 1) / K
        q3, q4 = (k + 1) / K, (k + 2) / K

        test_mask = (uniform_rand >= q1) & (uniform_rand < q2)
        calib_mask = (uniform_rand >= q3) & (uniform_rand < q4)
        train_mask = ~(test_mask | calib_mask)

        yield train_mask, calib_mask, test_mask

    q1, q2 = (K - 1) / K, 1.0
    q3, q4 = 0.0, 1.0 / K

    test_mask = (uniform_rand >= q1) & (uniform_rand < q2)
    calib_mask = (uniform_rand >= q3) & (uniform_rand < q4)
    train_mask = ~(test_mask | calib_mask)
    yield train_mask, calib_mask, test_mask
