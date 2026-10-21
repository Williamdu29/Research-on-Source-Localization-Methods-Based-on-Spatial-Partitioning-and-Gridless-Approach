# Source Localization Based on Spatial Partitioning and Gridless Approach

A PyTorch implementation of a two-stage deep learning framework for near-field source localization. The system first estimates the number of signal sources, then reconstructs a clean covariance matrix for DoA and distance estimation — avoiding the grid mismatch problem inherent in traditional spatial spectrum methods.

## Overview

Classical source localization algorithms (MUSIC, ESPRIT) require prior knowledge of the number of sources and suffer from grid-based spatial discretization artifacts. This project addresses both limitations:

1. **Source Number Estimation** — a classification network (LSN) predicts how many sources are present
2. **Covariance Reconstruction** — a regression network (LCN) reconstructs a denoised covariance matrix, enabling gridless parameter extraction via eigendecomposition or root-finding

Both networks operate on the sample covariance matrix of received array signals, split into real and imaginary channels as a 2-channel image input.

## Architecture

### LSN — Localization Source Number Network

Classifies the number of active signal sources from the observed covariance matrix.

```
Input: [2, N, N]  (Re/Im channels of sample covariance)
  │
  ├── Conv2d(2→64, k=2) → BN → ReLU
  ├── DSC(64→64, k=2) → BN → ReLU
  ├── DSC(64→64, k=2) → BN → ReLU
  ├── Flatten → FC(1024) → Dropout(0.5) → ReLU
  ├── FC(256) → Dropout → ReLU
  ├── FC(128) → Dropout → ReLU
  ├── FC(64) → Dropout → ReLU
  └── FC(num_sources) → Softmax
      └── Output: source count probabilities
```

### LCN — Localization Covariance Network

Reconstructs the clean covariance matrix representation for downstream DoA/distance estimation.

```
Input: [2, N, N]  (Re/Im channels of sample covariance)
  │
  ├── Conv2d(2→128, k=2) → BN → LeakyReLU
  ├── DSC(128→128, k=2) → BN → LeakyReLU
  ├── DSC(128→128, k=2) → BN → LeakyReLU
  ├── Global Average Pooling
  └── FC(128 → 3N−5)
      └── Output: covariance feature vector
```

Both networks use Depthwise Separable Convolutions (DSC) for parameter efficiency — each DSC layer decomposes into a per-channel spatial convolution followed by a 1x1 pointwise convolution.

## Signal Model

The system uses a near-field spherical wave propagation model. For a Uniform Linear Array (ULA) with N elements and inter-element spacing d = λ/4:

```
Steering vector:  a_n = exp(-j·2π/λ · (r_n − r))

where r_n = √(r² + (n·d)² − 2·n·d·r·sin(θ))   is the distance from source to the n-th element
```

The received signal at K snapshots:

```
X = a · sᵀ + W    →    R̂ = XXᴴ / K    →    Input: [Re(R̂), Im(R̂)]
```

## Project Structure

```
├── model/
│   ├── LSN.py             # Source number estimation network
│   └── LCN.py             # Covariance reconstruction network
├── data/
│   ├── single_source.py   # Near-field signal generation & dataset creation
│   └── dataset.npz        # Pre-generated dataset (26 SNR levels × 1000 samples)
└── README.md
```

## Two-Stage Pipeline

```
Received Signal X (N×K)
    → Sample Covariance R̂ = XXᴴ/K
    → Split into [Re(R̂), Im(R̂)]  →  [2, N, N] tensor
    │
    ├── Stage 1: LSN  →  Estimated source count M̂
    │
    └── Stage 2: LCN  →  Reconstructed covariance features
                          → Classical estimator (MUSIC / Root-MUSIC)
                          → DoA (θ̂) and distance (r̂) estimates
```

## Usage

### 1. Generate Dataset

```bash
python data/single_source.py
```

Generates near-field covariance samples across multiple SNR levels:
- Array: 5-element ULA, d = λ/4, carrier frequency 2 MHz
- DoA range: [−60°, 60°], distance range: [1 mm, λ]
- SNR range: [−10, 15] dB (26 levels, 1000 samples each)
- Output: `dataset.npz` containing covariance tensors and (θ, r) labels

### 2. Train Models

```bash
python model/LSN.py   # Train source number estimator
python model/LCN.py   # Train covariance reconstruction network
```

Both scripts include self-contained training loops with example data for quick validation.

## Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 5 | Number of antenna elements |
| f_c | 2 MHz | Carrier frequency |
| d | λ/4 | Inter-element spacing |
| K | 1024 | Snapshots per sample |
| θ range | [−60°, 60°] | DoA sampling range |
| r range | [1 mm, λ] | Source distance range (near-field) |
| SNR range | [−10, 15] dB | Training SNR levels |
| Max sources | 6 | Maximum number of signal sources |

## Dependencies

- Python 3.8+
- PyTorch ≥ 1.9
- NumPy
- SciPy

```bash
pip install torch numpy scipy
```

## Key Design Choices

- **Depthwise Separable Convolutions**: Reduce parameter count significantly compared to standard convolutions while maintaining performance on structured covariance inputs
- **Covariance as 2-channel image**: Treating the real and imaginary parts of the covariance matrix as a 2-channel image allows standard CNN architectures to process complex-valued data
- **Gridless estimation**: By reconstructing covariance features rather than discretizing the angular space, the approach avoids basis mismatch errors common in grid-based methods
- **Two-stage decoupling**: Separating source enumeration from localization allows each network to specialize and simplifies training

## License

This is an academic implementation for research and educational purposes.
