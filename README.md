
# NEGATIVA_ML

![example workflow](https://github.com/negativa-ai/negativa-ml/actions/workflows/main.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---


This repository provides the open-source implementation of our paper **“The Hidden Bloat in Machine Learning Systems.”**

The project includes the components described in the paper, the **kernel detector** and **kernel locator**.
The **compaction** component is not included in this release and will be made public upon acceptance of our companion paper.

## Overview

**NEGATIVA_ML** analyzes machine learning (ML) workloads to identify unused GPU code segments in shared libraries.
Given an ML workload, `negativa_ml` outputs:

* A list of **shared libraries** loaded during execution.
* The **unused GPU code segments** within those libraries.

See the [Quick Start](#quick-start) section below for usage instructions.

---

## Installation

**Tested environment:** Ubuntu 20.04

**Requirements:** CUDA must be properly installed and configured.
Refer to the official CUDA installation guide:  [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

### Step-by-step setup

1. **Install Rust:**
   [https://rust-lang.org/tools/install/](https://rust-lang.org/tools/install/)

2. **Install dependencies:**

   ```bash
   sudo apt update
   sudo apt install libspdlog-dev cmake
   ```

3. **Set LD_LIBRARY_PATH:**

   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
   ```

4. **Run tests:**

   ```bash
   make test
   ```

   All tests should pass successfully.

5. **Install and verify:**

   ```bash
   make install
   negativa_ml --help
   ```

---

## Quick Start

Example ML workloads are available in the `examples` directory.
Here’s how to analyze the `demo` example:

1. **Build the demo:**

   ```bash
   cd examples/demo && ./build.sh
   ```
   The `demo` example compiles a shared library `libdemo.so` and a main executable `main` that uses it.
   The shared library contains two GPU kernels: `matrixMulGPU` and `setScalarItems`.

2. **Verify the demo:**

   ```bash
   cd build && ./main matmul
   ```
    This should output `SUCCESS`.

3. **Run NEGATIVA_ML analysis:**

   ```bash
   negativa_ml debloat -- $PWD/main
   ```

   *Note:* The executable path must be **absolute**.

4. **View the results:**

   The analysis results are stored in the `nml_workspace` directory:

   ```
   nml_workspace
   ├── trace.json        # Traced kernels and loaded shared libraries
   ├── spans/            # Located unused GPU code segments
   │   └── libdemo.so.json  # Lists unused code segments in `libdemo.so`
   ```

   The `spans` directory contains the unused GPU code segments for each shared library.
   These can be used as input for the `compaction` component (not yet released).

---

## Reconstructing Shared Libraries

You can verify the detected code segments are unused by modifying them and rerunning the workload.
The following command reconstructs a shared library by overwriting unused code regions with `0x1`:

```bash
negativa_ml reconstruct \
  --span-path nml_workspace/spans/libdemo.so.json \
  --output-dir ./reconstructed
```

This will produce a reconstructed version of the shared library in `./reconstructed/`.
You may replace the original shared library with this version to verify correctness. **Remember to back up the original file first**.

For convenience, a helper script `debloat.sh` is provided under the demo example to automate this process.

---

## Command Line Usage

The main executable is `negativa_ml`, which supports four subcommands:

| Command       | Description                                                                       |
| ------------- | --------------------------------------------------------------------------------- |
| `trace`       | Traces kernel launches and loaded shared libraries during the workload execution. |
| `locate`      | Identifies unused GPU code segments in shared libraries based on trace results.   |
| `debloat`     | Runs `trace` and `locate` sequentially, producing final analysis results.         |
| `reconstruct` | Rebuilds shared libraries with unused code segments set to `0x1`.                 |

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{zhanghidden,
  title={The Hidden Bloat in Machine Learning Systems},
  author={Zhang, Huaifeng and Ali-Eldin, Ahmed},
  booktitle={Eighth Conference on Machine Learning and Systems (MLSys)},
  year={2025}
}
```
