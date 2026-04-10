"""
ROSE_cuda_kernels.py — CUDA Kernels for ROSE Pipeline Acceleration

Contains PyCUDA-compiled kernels for:
  1. Read density computation (the main bottleneck)
  2. Interval overlap queries (batch locus overlap)

All kernels operate on int32/float32 arrays for GPU efficiency.
Each kernel has a corresponding CPU fallback function.
"""

import sys
import logging
import numpy as np
from typing import List, Tuple, Optional

# GPU imports (conditional)
try:
    import pycuda.driver as cuda_drv
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

logger = logging.getLogger("ROSE_GPU")

# ===================================================================
# CUDA KERNEL SOURCE CODE
# ===================================================================

DENSITY_KERNEL_SRC = r"""
extern "C" {
/**
 * compute_read_density — Main ROSE density kernel
 *
 * For each bin in a genomic region, count how many reads overlap that bin
 * and compute the normalized density.
 *
 * Grid layout:  1 block per region, blockDim.x threads per block
 *               Each thread handles one or more bins.
 *
 * @param read_starts   Array of read start positions (int32, length n_reads)
 * @param read_ends     Array of read end positions (int32, length n_reads)
 * @param n_reads       Number of reads
 * @param region_starts Array of region start positions (int32, length n_regions)
 * @param region_ends   Array of region end positions (int32, length n_regions)
 * @param n_bins        Number of bins per region
 * @param n_regions     Number of regions
 * @param mmr           Million Mapped Reads normalization factor (float)
 * @param floor_val     Minimum coverage to count (int)
 * @param bin_densities Output array (float32, length n_regions * n_bins)
 */
__global__ void compute_read_density(
    const int* __restrict__ read_starts,
    const int* __restrict__ read_ends,
    int n_reads,
    const int* __restrict__ region_starts,
    const int* __restrict__ region_ends,
    int n_bins,
    int n_regions,
    float mmr,
    int floor_val,
    float* __restrict__ bin_densities
) {
    int region_idx = blockIdx.x;
    if (region_idx >= n_regions) return;

    int r_start = region_starts[region_idx];
    int r_end   = region_ends[region_idx];
    int r_len   = r_end - r_start;
    if (r_len <= 0) return;

    int bin_size = r_len / n_bins;
    if (bin_size <= 0) bin_size = 1;

    // Each thread processes one or more bins
    for (int bin_idx = threadIdx.x; bin_idx < n_bins; bin_idx += blockDim.x) {
        int bin_start = r_start + bin_idx * bin_size;
        int bin_end   = bin_start + bin_size;
        if (bin_end > r_end) bin_end = r_end;

        // Count positions covered by reads within this bin
        // We use a coverage-counting approach
        float total_coverage = 0.0f;
        
        for (int i = 0; i < n_reads; i++) {
            // Compute overlap between read [read_starts[i], read_ends[i]] and bin [bin_start, bin_end]
            int ov_start = max(read_starts[i], bin_start);
            int ov_end   = min(read_ends[i], bin_end);
            if (ov_start < ov_end) {
                total_coverage += (float)(ov_end - ov_start);
            }
        }

        float density = total_coverage / (float)bin_size;
        if (floor_val > 0 && density < (float)floor_val) {
            density = 0.0f;
        }
        
        // Normalize by Million Mapped Reads
        bin_densities[region_idx * n_bins + bin_idx] = density / mmr;
    }
}


/**
 * batch_interval_overlap — GPU interval overlap kernel
 *
 * For each query interval, find how many target intervals overlap it.
 * Returns overlap count and cumulative overlap size.
 *
 * Targets are assumed sorted by start position (per chromosome).
 *
 * @param query_starts  Query interval starts (int32, length n_queries)
 * @param query_ends    Query interval ends (int32, length n_queries)
 * @param target_starts Target interval starts (int32, length n_targets) — sorted
 * @param target_ends   Target interval ends (int32, length n_targets)
 * @param n_queries     Number of query intervals
 * @param n_targets     Number of target intervals
 * @param overlap_counts Output: number of overlaps per query (int32, length n_queries)
 * @param overlap_sizes  Output: total overlap bp per query (int32, length n_queries)
 */
__global__ void batch_interval_overlap(
    const int* __restrict__ query_starts,
    const int* __restrict__ query_ends,
    const int* __restrict__ target_starts,
    const int* __restrict__ target_ends,
    int n_queries,
    int n_targets,
    int* __restrict__ overlap_counts,
    int* __restrict__ overlap_sizes
) {
    int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= n_queries) return;

    int q_start = query_starts[qidx];
    int q_end   = query_ends[qidx];
    int count   = 0;
    int total_size = 0;

    // Binary search: find first target that could overlap (target_end > q_start)
    int lo = 0, hi = n_targets - 1, first = n_targets;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (target_ends[mid] > q_start) {
            first = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    // Scan from first candidate
    for (int i = first; i < n_targets; i++) {
        if (target_starts[i] >= q_end) break;  // past query — done
        
        // Check overlap: target_start < q_end AND target_end > q_start
        if (target_starts[i] < q_end && target_ends[i] > q_start) {
            count++;
            int ov_start = max(target_starts[i], q_start);
            int ov_end   = min(target_ends[i], q_end);
            total_size += (ov_end - ov_start);
        }
    }

    overlap_counts[qidx] = count;
    overlap_sizes[qidx] = total_size;
}

/**
 * batch_interval_overlap_indices — GPU interval indices kernel
 *
 * Second pass: writes the actual target indices that overlap into a flattened array.
 */
__global__ void batch_interval_overlap_indices(
    const int* __restrict__ query_starts,
    const int* __restrict__ query_ends,
    const int* __restrict__ target_starts,
    const int* __restrict__ target_ends,
    int n_queries,
    int n_targets,
    const int* __restrict__ overlap_offsets,
    int* __restrict__ overlap_indices
) {
    int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= n_queries) return;

    int q_start = query_starts[qidx];
    int q_end   = query_ends[qidx];
    
    int offset = overlap_offsets[qidx];
    int out_idx = 0;

    int lo = 0, hi = n_targets - 1, first = n_targets;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (target_ends[mid] > q_start) {
            first = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    for (int i = first; i < n_targets; i++) {
        if (target_starts[i] >= q_end) break;
        
        if (target_starts[i] < q_end && target_ends[i] > q_start) {
            overlap_indices[offset + out_idx] = i;
            out_idx++;
        }
    }
}

} /* end extern "C" */
"""

# ===================================================================
# Compiled Kernel Cache
# ===================================================================

_compiled_module = None
_compiled_context_id = None

def _get_module():
    """Lazily compile the CUDA module. Recompiles if CUDA context changes."""
    global _compiled_module, _compiled_context_id
    
    if not HAS_PYCUDA:
        raise RuntimeError("PyCUDA is not available. Cannot compile CUDA kernels.")
    
    # Check if current context matches the one we compiled for
    try:
        current_ctx = cuda_drv.Context.get_current()
        current_id = id(current_ctx) if current_ctx else None
    except Exception:
        current_id = None

    if _compiled_module is None or _compiled_context_id != current_id:
        _compiled_module = SourceModule(DENSITY_KERNEL_SRC)
        _compiled_context_id = current_id
        logger.info("CUDA kernels compiled successfully")
    
    return _compiled_module


def get_density_kernel():
    """Get the compiled density kernel function."""
    mod = _get_module()
    return mod.get_function("compute_read_density")


def get_overlap_kernel():
    """Get the compiled interval overlap kernel function."""
    mod = _get_module()
    return mod.get_function("batch_interval_overlap")


def get_overlap_indices_kernel():
    """Get the compiled interval overlap indices kernel function."""
    mod = _get_module()
    return mod.get_function("batch_interval_overlap_indices")


# ===================================================================
# GPU Density Computation (High-Level)
# ===================================================================

def compute_density_gpu(
    read_starts: np.ndarray,
    read_ends: np.ndarray,
    region_starts: np.ndarray,
    region_ends: np.ndarray,
    n_bins: int,
    mmr: float = 1.0,
    floor_val: int = 0,
    block_size: int = 256
) -> np.ndarray:
    """
    Compute read density for multiple regions using GPU.

    Args:
        read_starts:   1D int32 array of read start positions
        read_ends:     1D int32 array of read end positions
        region_starts: 1D int32 array of region start positions
        region_ends:   1D int32 array of region end positions
        n_bins:        Number of bins per region
        mmr:           Million mapped reads normalization
        floor_val:     Minimum coverage floor
        block_size:    CUDA threads per block

    Returns:
        2D float32 array of shape (n_regions, n_bins) with density values
    """
    n_reads = len(read_starts)
    n_regions = len(region_starts)

    if n_reads == 0 or n_regions == 0:
        return np.zeros((n_regions, n_bins), dtype=np.float32)

    # Ensure correct dtypes
    read_starts = np.ascontiguousarray(read_starts, dtype=np.int32)
    read_ends = np.ascontiguousarray(read_ends, dtype=np.int32)
    region_starts = np.ascontiguousarray(region_starts, dtype=np.int32)
    region_ends = np.ascontiguousarray(region_ends, dtype=np.int32)

    # Output array
    bin_densities = np.zeros(n_regions * n_bins, dtype=np.float32)

    # Get kernel
    kernel = get_density_kernel()

    # Launch: one block per region, block_size threads per block
    grid = (n_regions, 1, 1)
    block = (min(block_size, n_bins), 1, 1)

    kernel(
        cuda_drv.In(read_starts),
        cuda_drv.In(read_ends),
        np.int32(n_reads),
        cuda_drv.In(region_starts),
        cuda_drv.In(region_ends),
        np.int32(n_bins),
        np.int32(n_regions),
        np.float32(mmr),
        np.int32(floor_val),
        cuda_drv.Out(bin_densities),
        grid=grid,
        block=block
    )

    return bin_densities.reshape(n_regions, n_bins)


# ===================================================================
# GPU Interval Overlap (High-Level)
# ===================================================================

def compute_overlap_gpu(
    query_starts: np.ndarray,
    query_ends: np.ndarray,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
    block_size: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute interval overlaps for multiple queries against sorted targets using GPU.

    Args:
        query_starts:  1D int32 array of query start positions
        query_ends:    1D int32 array of query end positions
        target_starts: 1D int32 array of target start positions (MUST BE SORTED)
        target_ends:   1D int32 array of target end positions
        block_size:    CUDA threads per block

    Returns:
        (overlap_counts, overlap_sizes, overlap_indices)
    """
    n_queries = len(query_starts)
    n_targets = len(target_starts)

    if n_queries == 0 or n_targets == 0:
        return np.zeros(n_queries, dtype=np.int32), np.zeros(n_queries, dtype=np.int32), np.zeros(0, dtype=np.int32)

    query_starts = np.ascontiguousarray(query_starts, dtype=np.int32)
    query_ends = np.ascontiguousarray(query_ends, dtype=np.int32)
    target_starts = np.ascontiguousarray(target_starts, dtype=np.int32)
    target_ends = np.ascontiguousarray(target_ends, dtype=np.int32)

    overlap_counts = np.zeros(n_queries, dtype=np.int32)
    overlap_sizes = np.zeros(n_queries, dtype=np.int32)

    kernel = get_overlap_kernel()

    grid = ((n_queries + block_size - 1) // block_size, 1, 1)
    block = (block_size, 1, 1)

    kernel(
        cuda_drv.In(query_starts),
        cuda_drv.In(query_ends),
        cuda_drv.In(target_starts),
        cuda_drv.In(target_ends),
        np.int32(n_queries),
        np.int32(n_targets),
        cuda_drv.Out(overlap_counts),
        cuda_drv.Out(overlap_sizes),
        grid=grid,
        block=block
    )

    total_overlaps = int(np.sum(overlap_counts))
    if total_overlaps == 0:
        return overlap_counts, overlap_sizes, np.zeros(0, dtype=np.int32)

    offsets = np.zeros(n_queries, dtype=np.int32)
    if n_queries > 1:
        offsets[1:] = np.cumsum(overlap_counts)[:-1]

    overlap_indices = np.zeros(total_overlaps, dtype=np.int32)
    kernel_indices = get_overlap_indices_kernel()

    kernel_indices(
        cuda_drv.In(query_starts),
        cuda_drv.In(query_ends),
        cuda_drv.In(target_starts),
        cuda_drv.In(target_ends),
        np.int32(n_queries),
        np.int32(n_targets),
        cuda_drv.In(offsets),
        cuda_drv.Out(overlap_indices),
        grid=grid,
        block=block
    )

    return overlap_counts, overlap_sizes, overlap_indices


# ===================================================================
# CPU FALLBACK — Density (Multiprocessing)
# ===================================================================

def _compute_density_single_region_cpu(args):
    """CPU worker: compute density for a single region."""
    read_starts, read_ends, r_start, r_end, n_bins, mmr, floor_val = args

    r_len = r_end - r_start
    if r_len <= 0:
        return np.zeros(n_bins, dtype=np.float32)

    bin_size = r_len // n_bins
    if bin_size <= 0:
        bin_size = 1

    densities = np.zeros(n_bins, dtype=np.float32)

    for bin_idx in range(n_bins):
        bin_start = r_start + bin_idx * bin_size
        bin_end = bin_start + bin_size
        if bin_end > r_end:
            bin_end = r_end

        # Vectorized overlap computation using numpy
        ov_starts = np.maximum(read_starts, bin_start)
        ov_ends = np.minimum(read_ends, bin_end)
        overlaps = np.maximum(ov_ends - ov_starts, 0)
        total_coverage = float(np.sum(overlaps))

        density = total_coverage / bin_size
        if floor_val > 0 and density < floor_val:
            density = 0.0

        densities[bin_idx] = density / mmr

    return densities


def compute_density_cpu(
    read_starts: np.ndarray,
    read_ends: np.ndarray,
    region_starts: np.ndarray,
    region_ends: np.ndarray,
    n_bins: int,
    mmr: float = 1.0,
    floor_val: int = 0,
    n_workers: int = 1
) -> np.ndarray:
    """
    CPU fallback: compute read density using numpy + multiprocessing.
    Same interface as compute_density_gpu.
    """
    n_regions = len(region_starts)
    read_starts = np.asarray(read_starts, dtype=np.int32)
    read_ends = np.asarray(read_ends, dtype=np.int32)

    if n_regions == 0:
        return np.zeros((0, n_bins), dtype=np.float32)

    args_list = [
        (read_starts, read_ends, int(region_starts[i]), int(region_ends[i]),
         n_bins, mmr, floor_val)
        for i in range(n_regions)
    ]

    if n_workers > 1 and n_regions > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            results = pool.map(_compute_density_single_region_cpu, args_list)
    else:
        results = [_compute_density_single_region_cpu(a) for a in args_list]

    return np.array(results, dtype=np.float32)


# ===================================================================
# CPU FALLBACK — Interval Overlap
# ===================================================================

def compute_overlap_cpu(
    query_starts: np.ndarray,
    query_ends: np.ndarray,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CPU fallback: compute interval overlaps using numpy vectorization.
    """
    n_queries = len(query_starts)
    n_targets = len(target_starts)
    
    overlap_counts = np.zeros(n_queries, dtype=np.int32)
    overlap_sizes = np.zeros(n_queries, dtype=np.int32)
    indices_list = []

    target_starts = np.asarray(target_starts, dtype=np.int32)
    target_ends = np.asarray(target_ends, dtype=np.int32)

    for i in range(n_queries):
        q_s = int(query_starts[i])
        q_e = int(query_ends[i])
        
        # Vectorized: check overlap condition
        mask = (target_starts < q_e) & (target_ends > q_s)
        overlap_counts[i] = int(np.sum(mask))
        
        if overlap_counts[i] > 0:
            ov_s = np.maximum(target_starts[mask], q_s)
            ov_e = np.minimum(target_ends[mask], q_e)
            overlap_sizes[i] = int(np.sum(ov_e - ov_s))
            indices_list.append(np.where(mask)[0].astype(np.int32))

    if indices_list:
        overlap_indices = np.concatenate(indices_list)
    else:
        overlap_indices = np.zeros(0, dtype=np.int32)

    return overlap_counts, overlap_sizes, overlap_indices


# ===================================================================
# Unified API — Auto-selects GPU or CPU
# ===================================================================

def compute_density(
    read_starts: np.ndarray,
    read_ends: np.ndarray,
    region_starts: np.ndarray,
    region_ends: np.ndarray,
    n_bins: int,
    mmr: float = 1.0,
    floor_val: int = 0,
    use_gpu: bool = True,
    block_size: int = 256,
    n_cpu_workers: int = 4
) -> np.ndarray:
    """
    Unified density computation — automatically selects GPU or CPU.
    """
    if use_gpu and HAS_PYCUDA and cuda_drv.Device.count() > 0:
        try:
            return compute_density_gpu(
                read_starts, read_ends, region_starts, region_ends,
                n_bins, mmr, floor_val, block_size
            )
        except Exception as e:
            logger.warning(f"GPU density failed, falling back to CPU: {e}")

    return compute_density_cpu(
        read_starts, read_ends, region_starts, region_ends,
        n_bins, mmr, floor_val, n_cpu_workers
    )


def compute_overlap(
    query_starts: np.ndarray,
    query_ends: np.ndarray,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
    use_gpu: bool = True,
    block_size: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified interval overlap — automatically selects GPU or CPU.
    """
    if use_gpu and HAS_PYCUDA and cuda_drv.Device.count() > 0:
        try:
            return compute_overlap_gpu(
                query_starts, query_ends, target_starts, target_ends, block_size
            )
        except Exception as e:
            logger.warning(f"GPU overlap failed, falling back to CPU: {e}")

    return compute_overlap_cpu(
        query_starts, query_ends, target_starts, target_ends
    )
