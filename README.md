# ROSE-CUDA: High-Performance Super-Enhancer Analysis

ROSE-CUDA is a high-performance, GPU-accelerated fork of the **ROSE (Rank Ordering of Super-Enhancers)** pipeline originally developed by the St. Jude Children's Research Hospital.

### 🚀 Sviluppo e Innovazione
Questo progetto è stato interamente sviluppato e validato utilizzando **Antigravity di Google**, sfruttando la potenza sinergica dei modelli **Opus 4.6** e **Gemini 3.1 Pro**. L'integrazione tra intelligenza artificiale generativa e ingegneria del software ha permesso di trasformare un algoritmo seriale complesso in una soluzione parallela ad alte prestazioni in tempi record.

### 📊 Key Improvements
- **CUDA Acceleration**: Custom GPU kernels for BAM read density mapping and interval overlap computation.
- **Python Optimization**: Replaced legacy `os.system()` calls with `pysam` and `NumPy` vectorization for up to 100x speedup in specific modules.
- **Parallel Orchestration**: Multi-threaded processing of BAM files and dynamic GPU memory batching.
- **Scientific Reproducibility**: Includes a **COMPAT MODE** that ensures bit-for-bit identical results to the original St. Jude version.

### 🛠 Installation
Create the environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate rose-cuda
```

### 📖 Usage
The usage remains compatible with the original ROSE:
```bash
python bin/ROSE_main.py -g HG38 -i input_peaks.gff -r factor_rankby.bam -o output_dir
```
Use `--no-gpu` to run in high-performance CPU mode if no NVIDIA GPU is available.

---
*Developed with Antigravity by Google (Gemini CLI)*
