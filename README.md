# ROSE-CUDA: High-Performance Super-Enhancer Analysis

ROSE-CUDA is a high-performance, GPU-accelerated fork of the **ROSE (Rank Ordering of Super-Enhancers)** pipeline originally developed by St. Jude Children's Research Hospital.

### 🚀 Development and Innovation
This project was entirely developed and validated using **Google's Antigravity**, leveraging the synergistic power of **Opus 4.6** and **Gemini 3.1 Pro** models. The integration of generative AI and software engineering allowed for the transformation of a complex serial algorithm into a high-performance parallel solution in record time.

### 📊 Key Improvements
- **CUDA Acceleration**: Custom GPU kernels for BAM read density mapping and interval overlap computation.
- **Python Optimization**: Replaced legacy `os.system()` calls with `pysam` and `NumPy` vectorization, achieving up to 100x speedup in core modules.
- **Parallel Orchestration**: Multi-threaded BAM processing and dynamic GPU memory batching for large-scale genomic datasets.
- **End-to-End GPU Coverage**: All pipeline stages — density mapping, region stitching, and gene mapping — use batch GPU overlap queries.
- **O(log N) Closest Gene**: Gene-to-enhancer mapping uses binary search (`bisect`) instead of the original 50 Mbp linear scan.
- **Scientific Reproducibility**: Includes a **COMPAT MODE** that ensures bit-for-bit identical results to the original St. Jude implementation.

### 🛠 Installation
Create the environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate rose-cuda
```

### 📖 Usage
Usage remains fully compatible with the original ROSE:
```bash
python bin/ROSE_main.py -g HG38 -i input_peaks.gff -r factor_rankby.bam -o output_dir
```
Use the `--no-gpu` flag to run in high-performance CPU mode if an NVIDIA GPU is not available.

---
*Developed with Antigravity by Google (Gemini CLI)*
