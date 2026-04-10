"""
ROSE_gpu_utils.py — GPU Detection, Multi-GPU Dispatch, Memory Management

Provides automatic GPU detection, dynamic batch sizing based on available VRAM,
and multi-GPU workload distribution for the ROSE pipeline.

Supports: 1 or 2 NVIDIA GPUs (RTX 5060 Ti / Blackwell architecture)
Fallback: Automatic CPU multiprocessing when no GPU is available
"""

import os
import sys
import logging
import multiprocessing
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# GPU availability detection
# ---------------------------------------------------------------------------

GPU_AVAILABLE = False
CUPY_AVAILABLE = False
PYCUDA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None

try:
    import pycuda.driver as cuda_drv
    PYCUDA_AVAILABLE = True
except ImportError:
    cuda_drv = None

if PYCUDA_AVAILABLE:
    try:
        cuda_drv.init()
        if cuda_drv.Device.count() > 0:
            GPU_AVAILABLE = True
    except Exception:
        GPU_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("ROSE_GPU")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(name)s — %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# GPU Info Data Classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_mb: int
    free_memory_mb: int
    cuda_cores: int  # estimated


@dataclass
class GPUConfig:
    """Runtime configuration for GPU execution."""
    n_gpus: int
    gpu_infos: List[GPUInfo]
    block_size: int = 256        # CUDA threads per block
    max_batch_regions: int = 0   # dynamic — set by calibrate()
    max_batch_reads: int = 0     # dynamic — set by calibrate()
    use_gpu: bool = True


# ---------------------------------------------------------------------------
# GPU Detection & Enumeration
# ---------------------------------------------------------------------------

def detect_gpus() -> List[GPUInfo]:
    """Detect all available NVIDIA GPUs and return their specs."""
    if not GPU_AVAILABLE:
        logger.warning("No NVIDIA GPU detected. Will use CPU multiprocessing fallback.")
        return []

    gpus = []
    n_devices = cuda_drv.Device.count()

    for i in range(n_devices):
        dev = cuda_drv.Device(i)
        cc = dev.compute_capability()
        total_mem = dev.total_memory() // (1024 * 1024)

        # Estimate free memory (we'll refine when context is created)
        free_mem = int(total_mem * 0.85)  # assume 85% available

        # Estimate CUDA cores based on architecture
        sm_count = dev.get_attribute(cuda_drv.device_attribute.MULTIPROCESSOR_COUNT)
        # Blackwell (CC 10.x): 128 cores/SM; Ada (CC 8.9): 128; Ampere (CC 8.6): 128
        if cc[0] >= 10:
            cores_per_sm = 128  # Blackwell
        elif cc[0] >= 8:
            cores_per_sm = 128  # Ada/Ampere
        elif cc[0] >= 7:
            cores_per_sm = 64   # Turing/Volta
        else:
            cores_per_sm = 64

        gpu = GPUInfo(
            device_id=i,
            name=dev.name(),
            compute_capability=cc,
            total_memory_mb=total_mem,
            free_memory_mb=free_mem,
            cuda_cores=sm_count * cores_per_sm
        )
        gpus.append(gpu)
        logger.info(
            f"GPU {i}: {gpu.name} | CC {cc[0]}.{cc[1]} | "
            f"{total_mem} MB VRAM | {gpu.cuda_cores} CUDA cores | {sm_count} SMs"
        )

    return gpus


def get_free_memory_mb(device_id: int) -> int:
    """Get actual free memory on a GPU by creating a temporary context."""
    if not GPU_AVAILABLE:
        return 0
    try:
        dev = cuda_drv.Device(device_id)
        ctx = dev.make_context()
        free, total = cuda_drv.mem_get_info()
        ctx.pop()
        return free // (1024 * 1024)
    except Exception as e:
        logger.warning(f"Could not query GPU {device_id} memory: {e}")
        return 0


# ---------------------------------------------------------------------------
# Dynamic Batch Size Calibration
# ---------------------------------------------------------------------------

def calibrate_batch_sizes(gpu_info: GPUInfo, avg_reads_per_region: int = 5000,
                          region_size_bytes: int = 48) -> Tuple[int, int]:
    """
    Dynamically compute optimal batch sizes based on GPU VRAM.

    We need to fit in GPU memory:
      - read_starts array: n_reads * 4 bytes (int32)
      - read_ends array:   n_reads * 4 bytes (int32)
      - bin_density output: n_bins * 4 bytes (float32) per region
      - Overhead for kernel code, stack, etc: ~200 MB

    Returns:
        (max_batch_regions, max_batch_reads)
    """
    available_mb = gpu_info.free_memory_mb - 200  # reserve 200MB overhead
    available_bytes = available_mb * 1024 * 1024

    # Per-read cost: 8 bytes (start + end as int32)
    bytes_per_read = 8
    # Per-region cost: avg 50 bins * 4 bytes = 200 bytes output
    bytes_per_region = 50 * 4

    # Estimate: reads for one batch of regions
    max_reads = available_bytes // (bytes_per_read + 1)  # +1 for alignment
    max_reads = min(max_reads, 50_000_000)  # cap at 50M reads

    # Regions: based on remaining memory after reads
    reads_memory = avg_reads_per_region * bytes_per_read
    remaining = available_bytes - reads_memory
    max_regions = max(remaining // bytes_per_region, 100)
    max_regions = min(max_regions, 100_000)  # cap at 100k regions

    logger.info(
        f"GPU {gpu_info.device_id} batch calibration: "
        f"max_reads={max_reads:,}, max_regions={max_regions:,} "
        f"(avail VRAM: {available_mb} MB)"
    )

    return int(max_regions), int(max_reads)


def create_gpu_config(max_gpus: int = 2) -> GPUConfig:
    """
    Create GPU configuration: detect GPUs, calibrate batch sizes.
    If no GPU is available, returns a config with use_gpu=False.
    """
    gpus = detect_gpus()

    if len(gpus) == 0:
        logger.info("GPU mode DISABLED — using CPU multiprocessing fallback")
        return GPUConfig(
            n_gpus=0,
            gpu_infos=[],
            use_gpu=False,
            max_batch_regions=1000,  # CPU batch
            max_batch_reads=1_000_000
        )

    # Limit to max_gpus
    gpus = gpus[:max_gpus]

    # Calibrate using the smallest GPU (conservative)
    min_gpu = min(gpus, key=lambda g: g.free_memory_mb)
    max_regions, max_reads = calibrate_batch_sizes(min_gpu)

    config = GPUConfig(
        n_gpus=len(gpus),
        gpu_infos=gpus,
        use_gpu=True,
        max_batch_regions=max_regions,
        max_batch_reads=max_reads
    )

    logger.info(f"GPU config: {config.n_gpus} GPU(s), batch={max_regions} regions")
    return config


# ---------------------------------------------------------------------------
# Multi-GPU Work Distribution
# ---------------------------------------------------------------------------

def split_workload(items: list, n_splits: int) -> List[list]:
    """Split a list of work items into n roughly equal chunks."""
    if n_splits <= 1:
        return [items]

    chunk_size = len(items) // n_splits
    remainder = len(items) % n_splits
    chunks = []
    start = 0
    for i in range(n_splits):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


# ---------------------------------------------------------------------------
# CPU Multiprocessing Fallback
# ---------------------------------------------------------------------------

def get_cpu_worker_count() -> int:
    """Get optimal number of CPU workers for multiprocessing fallback."""
    n_cpus = multiprocessing.cpu_count()
    # Use all but 2 cores (leave room for OS + I/O)
    workers = max(1, n_cpus - 2)
    logger.info(f"CPU fallback: {workers} workers (of {n_cpus} available cores)")
    return workers


def cpu_parallel_map(func, items, n_workers: Optional[int] = None):
    """
    Execute a function over items in parallel using multiprocessing.
    Used as fallback when GPU is not available.
    """
    if n_workers is None:
        n_workers = get_cpu_worker_count()

    if n_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(func, items)
    return results


# ---------------------------------------------------------------------------
# Context Manager for Multi-GPU
# ---------------------------------------------------------------------------

class GPUContext:
    """
    Context manager for a specific GPU device.
    Creates and manages a PyCUDA context for the given GPU.
    """

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.context = None

    def __enter__(self):
        if not GPU_AVAILABLE:
            return self
        dev = cuda_drv.Device(self.device_id)
        self.context = dev.make_context()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context is not None:
            self.context.pop()
            self.context = None
        return False


# ---------------------------------------------------------------------------
# Module-level Summary
# ---------------------------------------------------------------------------

def print_system_summary():
    """Print a summary of the GPU/CPU compute environment."""
    print("=" * 60)
    print("ROSE CUDA — System Summary")
    print("=" * 60)
    print(f"  GPU Available:    {GPU_AVAILABLE}")
    print(f"  PyCUDA:           {PYCUDA_AVAILABLE}")
    print(f"  CuPy:             {CUPY_AVAILABLE}")
    if GPU_AVAILABLE:
        n = cuda_drv.Device.count()
        print(f"  GPU Count:        {n}")
        for i in range(n):
            dev = cuda_drv.Device(i)
            mem = dev.total_memory() // (1024 * 1024)
            cc = dev.compute_capability()
            print(f"    GPU {i}: {dev.name()} — {mem} MB — CC {cc[0]}.{cc[1]}")
    else:
        n_cpu = multiprocessing.cpu_count()
        print(f"  CPU Cores:        {n_cpu}")
        print(f"  CPU Workers:      {get_cpu_worker_count()}")
    print("=" * 60)


if __name__ == "__main__":
    print_system_summary()
