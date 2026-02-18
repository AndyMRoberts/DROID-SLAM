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

## Findings: Full ONNX Model Attempt (abandoned)

An attempt was made to export and run the whole DroidNet as a single ONNX model (or as `droid_net_features.onnx` + `droid_net_update.onnx`) via `andy/onnx_droid_net_full.py`. This approach was abandoned. Summary:

**What was tried**
- Exporting a combined model that includes fnet, cnet, and the full update (including GraphAgg with `scatter_mean`).
- Integrating this into the pipeline with `--use_onnx_full`.

**Why it didn't work**
1. **GraphAgg shape baking**: The update's GraphAgg uses `torch_scatter.scatter_mean`, which aggregates over variable-sized groups determined by unique indices (`ii`). During ONNX export with `num_edges=3`, the output shapes were effectively fixed to `num_unique=3`. At runtime, the pipeline uses different numbers of edges (e.g. 1 for MotionFilter, 12+ for FactorGraph). The ONNX graph then produced shape mismatches such as `{1,1,48,64}` vs `{1,3,48,64}` or `{1,12,48,64}` vs `{1,3,48,64}`.
2. **No benefit over split models**: Even when falling back to PyTorch for the update (to avoid the shape issues), full-ONNX mode only used ONNX for features. The individual models (`fnet.onnx`, `cnet.onnx`, `update_core.onnx`) already provide ONNX for the update core (GRU, delta, weight) and only use PyTorch for GraphAgg (eta, upmask). That gives better ONNX coverage than the full model approach.

**Recommendation**: Use `--use_onnx` with the individual ONNX models from `onnx_conversion.ipynb`, or run purely in PyTorch. The full-ONNX export script (`onnx_droid_net_full.py`) remains for reference or profiling but is not wired into the pipeline.

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
  --gt_path /mnt/data/datasets/agricultural/tartanair/mono_gt/ \
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
