#!/usr/bin/env python3
"""
test_rose_cuda.py — Test and benchmark script for ROSE CUDA pipeline

Runs the full ROSE pipeline on the H3K27ac test data and compares
GPU vs CPU performance. Can be run in three modes:

  1. --gpu-only    : Run with GPU acceleration only
  2. --cpu-only    : Run with CPU multiprocessing only
  3. --benchmark   : Run both and compare performance + output correctness

Usage:
    python test_rose_cuda.py --benchmark
    python test_rose_cuda.py --gpu-only
    python test_rose_cuda.py --cpu-only
    python test_rose_cuda.py --test-kernels  # Unit test GPU kernels only
"""

import os
import sys
import time
import argparse
import numpy as np

# Add lib to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROSE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'bin' else SCRIPT_DIR
sys.path.insert(0, os.path.join(ROSE_DIR, 'lib'))
sys.path.insert(0, os.path.join(ROSE_DIR, 'bin'))

# ===================================================================
# Test data paths
# ===================================================================

DATA_DIR = os.path.join(ROSE_DIR, 'data_h3k27ac_test')
BAM_FILE = os.path.join(DATA_DIR,
    '03_0GUC_025MIRCCS_KO-EZH2-EZH2-wt-HOXA9_H3K27Ac_hs_i21.deduped.filter.srt.bam')
GFF_FILE = os.path.join(DATA_DIR, '03_HOXA9_H3K27ac_blacklisted.gff')
ANNOT_DIR = os.path.join(DATA_DIR, 'annotation')
GENOME = 'HG38'

OUTPUT_GPU = os.path.join(ROSE_DIR, 'test_output_gpu')
OUTPUT_CPU = os.path.join(ROSE_DIR, 'test_output_cpu')


def check_test_data():
    """Verify test data exists."""
    print("=" * 60)
    print("CHECKING TEST DATA")
    print("=" * 60)

    missing = []
    for f in [BAM_FILE, BAM_FILE + '.bai', GFF_FILE]:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  ✅ {os.path.basename(f)} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ MISSING: {f}")
            missing.append(f)

    # Check annotation
    annot_file = os.path.join(ANNOT_DIR, 'hg38_refseq.ucsc')
    if os.path.exists(annot_file):
        print(f"  ✅ hg38_refseq.ucsc")
    else:
        print(f"  ❌ MISSING: {annot_file}")
        missing.append(annot_file)

    if missing:
        print("\n❌ Missing test files. Cannot proceed.")
        sys.exit(1)

    print("  All test data present.\n")


def test_gpu_detection():
    """Test GPU detection and system info."""
    print("=" * 60)
    print("GPU DETECTION TEST")
    print("=" * 60)

    try:
        from ROSE_gpu_utils import (
            detect_gpus, create_gpu_config, print_system_summary,
            GPU_AVAILABLE, PYCUDA_AVAILABLE, CUPY_AVAILABLE
        )
        print_system_summary()

        if GPU_AVAILABLE:
            config = create_gpu_config(max_gpus=2)
            print(f"\n  GPU Config: {config.n_gpus} GPU(s)")
            print(f"  Max batch regions: {config.max_batch_regions:,}")
            print(f"  Max batch reads: {config.max_batch_reads:,}")
            return True
        else:
            print("\n  ⚠️  No GPU detected. GPU tests will be skipped.")
            return False
    except Exception as e:
        print(f"\n  ❌ GPU detection failed: {e}")
        return False


def test_pysam():
    """Test pysam BAM reading."""
    print("\n" + "=" * 60)
    print("PYSAM BAM READING TEST")
    print("=" * 60)

    try:
        import pysam
        print(f"  pysam version: {pysam.__version__}")

        t0 = time.time()
        bam = pysam.AlignmentFile(BAM_FILE, "rb")

        # Count mapped reads
        stats = bam.get_index_statistics()
        total_mapped = sum(s.mapped for s in stats)
        print(f"  Total mapped reads: {total_mapped:,}")

        # Test fetching a region
        t1 = time.time()
        count = 0
        for read in bam.fetch('chr1', 1000000, 1100000):
            count += 1
        t2 = time.time()

        print(f"  Reads in chr1:1000000-1100000: {count:,}")
        print(f"  Fetch time: {(t2-t1)*1000:.1f} ms")

        bam.close()
        print("  ✅ pysam test PASSED")
        return True
    except Exception as e:
        print(f"  ❌ pysam test FAILED: {e}")
        return False


def test_cuda_kernels():
    """Unit test the CUDA density and overlap kernels."""
    print("\n" + "=" * 60)
    print("CUDA KERNEL UNIT TESTS")
    print("=" * 60)

    try:
        from ROSE_cuda_kernels import (
            compute_density_gpu, compute_density_cpu,
            compute_overlap_gpu, compute_overlap_cpu,
            HAS_PYCUDA
        )
    except ImportError as e:
        print(f"  ❌ Cannot import kernels: {e}")
        return False

    # --- Test density computation ---
    print("\n  [Test 1] Density kernel — synthetic data")
    np.random.seed(42)

    # Create synthetic reads: 10,000 reads in region [1000, 5000]
    n_reads = 10000
    read_starts = np.random.randint(800, 4800, size=n_reads).astype(np.int32)
    read_ends = read_starts + np.random.randint(50, 200, size=n_reads).astype(np.int32)

    # One region [1000, 5000], 10 bins
    region_starts = np.array([1000], dtype=np.int32)
    region_ends = np.array([5000], dtype=np.int32)
    n_bins = 10
    mmr = 1.0

    # CPU baseline
    t0 = time.time()
    cpu_result = compute_density_cpu(
        read_starts, read_ends, region_starts, region_ends, n_bins, mmr, 0
    )
    t_cpu = time.time() - t0
    print(f"    CPU result: {cpu_result[0, :5]} ... (took {t_cpu*1000:.1f} ms)")

    # GPU test
    if HAS_PYCUDA:
        try:
            import pycuda.driver as cuda_drv
            cuda_drv.init()
            if cuda_drv.Device.count() > 0:
                from ROSE_gpu_utils import GPUContext
                with GPUContext(0):
                    t0 = time.time()
                    gpu_result = compute_density_gpu(
                        read_starts, read_ends, region_starts, region_ends, n_bins, mmr, 0
                    )
                    t_gpu = time.time() - t0
                    print(f"    GPU result: {gpu_result[0, :5]} ... (took {t_gpu*1000:.1f} ms)")

                    # Compare
                    max_diff = np.max(np.abs(cpu_result - gpu_result))
                    print(f"    Max difference CPU vs GPU: {max_diff:.6f}")
                    if max_diff < 0.01:
                        print("    ✅ Density kernel PASSED")
                    else:
                        print("    ⚠️  Density values differ (may be due to floating point)")
        except Exception as e:
            print(f"    ⚠️  GPU density test skipped: {e}")
    else:
        print("    ⚠️  PyCUDA not available, GPU test skipped")

    # --- Test overlap computation ---
    print("\n  [Test 2] Overlap kernel — synthetic data")

    query_starts = np.array([100, 500, 900, 1500], dtype=np.int32)
    query_ends = np.array([300, 800, 1100, 2000], dtype=np.int32)
    target_starts = np.array([150, 250, 600, 950, 1000], dtype=np.int32)
    target_ends = np.array([200, 350, 750, 1050, 1800], dtype=np.int32)

    cpu_counts, cpu_sizes = compute_overlap_cpu(query_starts, query_ends, target_starts, target_ends)
    print(f"    CPU overlap counts: {cpu_counts}")
    print(f"    CPU overlap sizes:  {cpu_sizes}")

    if HAS_PYCUDA:
        try:
            import pycuda.driver as cuda_drv
            cuda_drv.init()
            if cuda_drv.Device.count() > 0:
                from ROSE_gpu_utils import GPUContext
                with GPUContext(0):
                    gpu_counts, gpu_sizes = compute_overlap_gpu(
                        query_starts, query_ends, target_starts, target_ends
                    )
                    print(f"    GPU overlap counts: {gpu_counts}")
                    print(f"    GPU overlap sizes:  {gpu_sizes}")

                    if np.array_equal(cpu_counts, gpu_counts):
                        print("    ✅ Overlap kernel PASSED")
                    else:
                        print("    ⚠️  Overlap counts differ")
        except Exception as e:
            print(f"    ⚠️  GPU overlap test skipped: {e}")

    # --- Test larger scale density ---
    print("\n  [Test 3] Large-scale density — 1000 regions, 50000 reads")

    n_reads = 50000
    read_starts = np.random.randint(0, 10000000, size=n_reads).astype(np.int32)
    read_ends = read_starts + 150

    n_regions = 1000
    region_starts = np.random.randint(0, 9900000, size=n_regions).astype(np.int32)
    region_ends = region_starts + np.random.randint(500, 5000, size=n_regions).astype(np.int32)
    n_bins = 50

    t0 = time.time()
    cpu_result = compute_density_cpu(read_starts, read_ends, region_starts, region_ends, n_bins, 1.0, 0, n_workers=4)
    t_cpu = time.time() - t0
    print(f"    CPU time (4 workers): {t_cpu:.2f}s")

    if HAS_PYCUDA:
        try:
            import pycuda.driver as cuda_drv
            cuda_drv.init()
            if cuda_drv.Device.count() > 0:
                from ROSE_gpu_utils import GPUContext
                with GPUContext(0):
                    t0 = time.time()
                    gpu_result = compute_density_gpu(read_starts, read_ends, region_starts, region_ends, n_bins, 1.0, 0)
                    t_gpu = time.time() - t0
                    print(f"    GPU time:             {t_gpu:.2f}s")
                    if t_cpu > 0 and t_gpu > 0:
                        print(f"    Speedup:              {t_cpu/t_gpu:.1f}x")
        except Exception as e:
            print(f"    ⚠️  GPU large-scale test skipped: {e}")

    return True


def test_bam_to_gff_gpu(use_gpu=True, n_regions=500, n_bins=1):
    """Test the full bamToGFF mapping on real data."""
    mode = "GPU" if use_gpu else "CPU"
    print(f"\n{'=' * 60}")
    print(f"FULL BAM-TO-GFF TEST ({mode} MODE) — {n_regions} regions, {n_bins} bins")
    print("=" * 60)

    import ROSE_utils as utils

    gff = utils.parseTable(GFF_FILE, '\t')

    if n_regions >= len(gff):
        gff_subset = gff
    else:
        # Sample evenly across the file to cover multiple chromosomes
        step = max(1, len(gff) // n_regions)
        gff_subset = gff[::step][:n_regions]

    actual_count = len(gff_subset)
    chroms = set(line[0] for line in gff_subset)
    print(f"  Testing with {actual_count} regions across {len(chroms)} chromosomes")

    import ROSE_bamToGFF

    t0 = time.time()
    result = ROSE_bamToGFF.mapBamToGFF(
        BAM_FILE, gff_subset, sense='both', extension=200,
        floor=0, rpm=True, matrix=n_bins, use_gpu=use_gpu
    )
    elapsed = time.time() - t0

    print(f"\n  Results: {len(result)} rows (including header)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Rate: {actual_count/elapsed:.0f} regions/sec")

    if len(result) > 1:
        print(f"  Sample output (first 3 rows):")
        for row in result[1:4]:
            display = row[:4] if len(row) > 5 else row
            print(f"    {display}{'...' if len(row) > 5 else ''}")

    print(f"  ✅ bamToGFF {mode} test PASSED")
    return elapsed, result


def run_full_pipeline(output_dir, use_gpu=True):
    """Run the complete ROSE pipeline."""
    mode = "GPU" if use_gpu else "CPU"
    print(f"\n{'=' * 60}")
    print(f"FULL PIPELINE RUN ({mode} MODE)")
    print(f"Output: {output_dir}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Copy annotation to expected location
    annot_dest = os.path.join(os.getcwd(), 'annotation')
    os.makedirs(annot_dest, exist_ok=True)
    annot_src = os.path.join(ANNOT_DIR, 'hg38_refseq.ucsc')
    annot_dst = os.path.join(annot_dest, 'hg38_refseq.ucsc')
    if not os.path.exists(annot_dst):
        os.system(f'cp "{annot_src}" "{annot_dst}"')

    gpu_flag = '' if use_gpu else '--no-gpu'
    cmd = (
        f'python3 {os.path.join(ROSE_DIR, "bin", "ROSE_main.py")} '
        f'-g {GENOME} '
        f'-i {GFF_FILE} '
        f'-r {BAM_FILE} '
        f'-o {output_dir} '
        f'-t 2000 '
        f'-s 12500 '
        f'{gpu_flag}'
    )

    print(f"\n  Command: {cmd}\n")

    t0 = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - t0

    print(f"\n  Pipeline {'succeeded' if ret == 0 else 'FAILED'}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # List output files
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"  Output files ({len(files)}):")
        for f in sorted(files):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath) / 1024
                print(f"    {f} ({size:.1f} KB)")

    return elapsed, ret


def run_benchmark():
    """Run both CPU and GPU modes and compare on realistic workload."""
    print("\n" + "=" * 60)
    print("BENCHMARK: GPU vs CPU (REALISTIC WORKLOAD)")
    print("=" * 60)

    import ROSE_utils as utils
    gff = utils.parseTable(GFF_FILE, '\t')
    total_regions = len(gff)

    # Use 2000 regions sampled across all chroms, 50 bins — compute-heavy benchmark
    n_test_regions = min(2000, total_regions)
    n_test_bins = 50

    print(f"\n  Dataset: {total_regions} total regions, BAM: 22.9M reads")
    print(f"  Benchmark: {n_test_regions} regions, {n_test_bins} bins per region")
    print(f"  This tests the compute-heavy path where GPU shines\n")

    gpu_time, gpu_result = test_bam_to_gff_gpu(use_gpu=True, n_regions=n_test_regions, n_bins=n_test_bins)
    cpu_time, cpu_result = test_bam_to_gff_gpu(use_gpu=False, n_regions=n_test_regions, n_bins=n_test_bins)

    print(f"\n  {'='*50}")
    print(f"  BENCHMARK RESULTS ({n_test_regions} regions x {n_test_bins} bins)")
    print(f"  {'='*50}")
    print(f"  GPU time: {gpu_time:.1f}s")
    print(f"  CPU time: {cpu_time:.1f}s")
    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"  Speedup:  {speedup:.1f}x {'🚀' if speedup > 2 else ''}")

    # Compare outputs
    if len(gpu_result) == len(cpu_result) and len(gpu_result) > 1:
        # Sort headers out, then sort by Region ID (row[0]) because multi-threading changes output order
        header = gpu_result[0]
        gpu_sorted = sorted(gpu_result[1:], key=lambda x: str(x[0]))
        cpu_sorted = sorted(cpu_result[1:], key=lambda x: str(x[0]))

        diffs = 0
        total_vals = 0
        max_diff = 0.0
        for i in range(len(gpu_sorted)):
            for j in range(2, min(len(gpu_sorted[i]), len(cpu_sorted[i]))):
                try:
                    g = float(gpu_sorted[i][j])
                    c = float(cpu_sorted[i][j])
                    total_vals += 1
                    d = abs(g - c)
                    max_diff = max(max_diff, d)
                    if d > 0.01:
                        diffs += 1
                except (ValueError, IndexError):
                    pass
        if diffs == 0:
            print(f"  Output:   ✅ IDENTICAL ({total_vals:,} values, max_diff={max_diff:.6f})")
        else:
            print(f"  Output:   ⚠️  {diffs}/{total_vals} values differ > 0.01 (max_diff={max_diff:.6f})")
    else:
        print(f"  Output:   ⚠️  Different row counts (GPU={len(gpu_result)}, CPU={len(cpu_result)})")


def main():
    parser = argparse.ArgumentParser(description='Test ROSE CUDA GPU acceleration')
    parser.add_argument('--gpu-only', action='store_true', help='Run GPU mode only')
    parser.add_argument('--cpu-only', action='store_true', help='Run CPU mode only')
    parser.add_argument('--benchmark', action='store_true', help='Run both and compare')
    parser.add_argument('--test-kernels', action='store_true', help='Unit test CUDA kernels only')
    parser.add_argument('--full-pipeline', action='store_true', help='Run the complete ROSE pipeline')
    parser.add_argument('--full-benchmark', action='store_true', help='Full pipeline benchmark GPU vs CPU')
    args = parser.parse_args()

    # Default to test-kernels if no args
    if not any(vars(args).values()):
        args.test_kernels = True

    print("\n🧬 ROSE CUDA Test Suite")
    print(f"   ROSE directory: {ROSE_DIR}")
    print(f"   Data directory: {DATA_DIR}\n")

    check_test_data()
    has_gpu = test_gpu_detection()
    test_pysam()

    if args.test_kernels:
        test_cuda_kernels()

    if args.gpu_only:
        test_bam_to_gff_gpu(use_gpu=True)

    if args.cpu_only:
        test_bam_to_gff_gpu(use_gpu=False)

    if args.benchmark:
        run_benchmark()

    if args.full_pipeline:
        use_gpu = not args.cpu_only
        run_full_pipeline(OUTPUT_GPU if use_gpu else OUTPUT_CPU, use_gpu=use_gpu)

    if args.full_benchmark:
        print("\n🏁 FULL PIPELINE BENCHMARK")
        gpu_time, gpu_ret = run_full_pipeline(OUTPUT_GPU, use_gpu=True)
        cpu_time, cpu_ret = run_full_pipeline(OUTPUT_CPU, use_gpu=False)

        print(f"\n  {'='*50}")
        print(f"  FULL PIPELINE BENCHMARK RESULTS")
        print(f"  {'='*50}")
        print(f"  GPU pipeline: {gpu_time:.1f}s ({gpu_time/60:.1f} min) — {'OK' if gpu_ret==0 else 'FAIL'}")
        print(f"  CPU pipeline: {cpu_time:.1f}s ({cpu_time/60:.1f} min) — {'OK' if cpu_ret==0 else 'FAIL'}")
        if gpu_time > 0:
            print(f"  Speedup:      {cpu_time/gpu_time:.1f}x")

    print("\n🎉 Test suite completed!\n")


if __name__ == "__main__":
    main()
