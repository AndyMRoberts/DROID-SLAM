# ONNX mode for DROID-SLAM

This folder contains tools to export DROID's neural networks to ONNX and run the pipeline using ONNXRuntime (GPU) instead of PyTorch for those subnets. Correlation and bundle adjustment stay in PyTorch/CUDA.

## 1. Export ONNX models

From the project root, open and run **`andy/onnx_conversion.ipynb`** (with the same conda env you use for DROID, e.g. `droidenv`).

The notebook:

- Loads `droid.pth` and builds `DroidNet`
- Exports **fnet**, **cnet**, and **update_core** to ONNX with the expected input/output names and dynamic axes
- Writes next to the notebook (or paths you set):
  - `fnet.onnx`
  - `cnet.onnx`
  - `update_core.onnx`

Run all cells; the last cell checks the ONNX files. For running from the repo root, either copy these three files into the project root or note their paths for the next step.

### Single ONNX (whole neural backbone)

For a **single ONNX** that combines fnet, cnet, and update (including graph aggregation) in one model, use **`andy/onnx_droid_net_full.py`**:

```bash
python andy/onnx_droid_net_full.py --pth droid.pth --out droid_net.onnx
```

This produces `droid_net.onnx` with inputs `(images, corr, flow, ii, jj)` and outputs `(fmaps, net_out, inp, delta, weight, eta, upmask)`. CorrBlock output (`corr`) and motion (`flow`) are provided as inputs because CorrBlock uses custom CUDA backends and cannot be exported. Use `--try-full` to attempt (and document) a full DroidNet.forward export; it fails on SE3/lietorch and CorrBlock as expected.

## 2. Run in ONNX mode

From the **project root** (where `demo.py` lives), run the demo with ONNX enabled:

```bash
python demo.py \
  --imagedir /path/to/images \
  --calib /path/to/calib.txt \
  --weights droid.pth \
  --use_onnx
```

If the ONNX files are not in the current directory, pass their paths:

```bash
python demo.py \
  --imagedir /path/to/images \
  --calib /path/to/calib.txt \
  --weights droid.pth \
  --use_onnx \
  --onnx_fnet /path/to/fnet.onnx \
  --onnx_cnet /path/to/cnet.onnx \
  --onnx_update /path/to/update_core.onnx
```

Optional:

- **`--onnx_tensorrt`** – use TensorRT execution provider if available (otherwise falls back to CUDA).
- Other `demo.py` options (e.g. `--disable_vis`, `--stride`, `--reconstruction_path`) work as usual.

## Prerequisites

- **Conda env** with PyTorch and DROID dependencies (e.g. `droidenv`).
- **onnxruntime-gpu** in that env, e.g.  
  `pip install onnxruntime-gpu`
- **cuDNN 9** available to ONNXRuntime for the CUDA execution provider (e.g. install into the env via conda: `conda install -c nvidia cudnn=9`). The ONNX backend in `droid_slam/onnx_backend.py` will preload cuDNN from the env's `lib/` when needed.

Without GPU/cuDNN, ONNXRuntime will fall back to CPU (slower).

## What runs on ONNX vs PyTorch

| Component        | Runtime        |
|-----------------|----------------|
| fnet, cnet      | ONNXRuntime    |
| update (core)   | ONNXRuntime    |
| eta / upmask    | PyTorch (GraphAgg) |
| CorrBlock       | PyTorch (custom CUDA) |
| Bundle adjustment | PyTorch       |

So the heavy encoders and update core use the exported ONNX models; correlation and BA stay as in the original DROID-SLAM pipeline.

## 3. Run TartanAir evaluation (test_tartanair)

From the **project root**, run the TartanAir evaluation using the launcher (recommended) or the test script directly.

**Launcher (recommended)** — creates `andy/runs/YYYYMMDD_HHMM_<test_run_name>/` with `metadata.txt`, ATE results, and plots:

PyTorch:

```bash
sudo ls # enables sudo access needed for the power logger to have rapl access
python launch_tartanair.py \
  --test_run_name tartanair_mono \
  --datapath /mnt/data/datasets/agricultural/tartanair/tartanair_mono_track/ \
  --gt_path /mnt/data/datasets/agricultural/tartanair/mono_gt/ \
  --asynchronous \
  --disable_vis \
  --power_log
```

ONNX:

```bash
sudo ls # enables sudo access needed for the power logger to have rapl access
python launch_tartanair.py \
  --test_run_name tartanair_mono_onnx \
  --datapath /mnt/data/datasets/agricultural/tartanair/tartanair_mono_track/ \
  --gt_path /mnt/data/datasets/agricultural/tartanair/mono_gt/ \
  --asynchronous \
  --disable_vis \
  --use_onnx \
  --power_log
```

**Direct test script** (no run directory):

```bash
python evaluation_scripts/test_tartanair_andy.py \
  --datapath /mnt/data/datasets/agricultural/tartanair/tartanair_mono_track/ \
  --gt_path /mnt/data/datasets/agricultural/tartanair/tartanair/mono_gt/ \
  --asynchronous \
  --disable_vis
```

Add `--use_onnx` for ONNX runtime.

## 4. Power logging (optional)

Power logging is integrated into `launch_tartanair` via `--power_log`. It uses the C++ tool in `andy/metric_measurement/power.cc` (CPU RAPL + nvidia-smi for GPU power and memory).

1. Compile once: `g++ -o power andy/metric_measurement/power.cc` (creates `power` in project root)
2. Run with power logging:
   ```bash
   python launch_tartanair.py --power_log --test_run_name tartanair_mono --datapath ... --gt_path ...
   ```

The launcher starts the power logger before the test, stops it when done, and writes to the run directory:
- `power_log.csv` – raw power samples
- `power_summary.json` / `power_summary.txt` – run duration, total energy (J), mean power (W), energy per frame (J and mJ)

Use `andy/metric_measurement/view_power_log.ipynb` to visualize the CSV.
