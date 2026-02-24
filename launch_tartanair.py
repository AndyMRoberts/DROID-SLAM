#!/usr/bin/env python3
"""
Generic launcher for TartanAir evaluation.

Run from the project root (where demo.py lives). Creates a run directory
andy/runs/YYYYMMDD_HHMM_<test_run_name>, writes metadata with all parameters,
then runs evaluation_scripts/test_tartanair_andy.py with outputs directed there.

Optional profiling (GPU/power etc): use --power_log and the 'profiler' package
  (from profiler import Profiler). Run directory is created by the profiler.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

try:
    from profiler import Profiler
except ImportError:
    Profiler = None


def _format_with_stdev(value, stddev=None):
    """Format a numeric value with optional ± stddev on the same line."""
    if value is None:
        return "N/A"
    if stddev is not None and stddev is not False:
        try:
            s = float(stddev)
            if s == s:  # not nan
                return f"{value} ± {s}"
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_power_summary_txt(run_dir, meta):
    """
    Write power_summary.txt from profiler metadata.json content (meta dict).
    Uses ± stdev on the same line when _stddev is present.
    """
    txt_path = os.path.join(run_dir, "power_summary.txt")
    av = meta.get("averages") or {}
    ef = meta.get("energy_per_frame_j") or {}

    run_time_s = meta.get("run_time_s")
    num_frames = meta.get("num_frames")

    cpu_w = av.get("cpu_power_w")
    gpu_w = av.get("gpu_power_w")
    cpu_std = av.get("cpu_power_w_stddev")
    gpu_std = av.get("gpu_power_w_stddev")
    mean_power = (cpu_w or 0) + (gpu_w or 0)
    # Approximate stddev of sum (independent): sqrt(sigma_cpu^2 + sigma_gpu^2)
    mean_power_std = None
    if cpu_std is not None and gpu_std is not None:
        mean_power_std = (float(cpu_std) ** 2 + float(gpu_std) ** 2) ** 0.5
    elif cpu_std is not None:
        mean_power_std = float(cpu_std)
    elif gpu_std is not None:
        mean_power_std = float(gpu_std)

    total_energy_j = (run_time_s * mean_power) if (run_time_s and mean_power) else None
    total_energy_std = (run_time_s * mean_power_std) if (run_time_s and mean_power_std is not None) else None
    total_energy_kj = round(total_energy_j / 1000, 2) if total_energy_j is not None else None
    total_energy_kj_std = round(total_energy_std / 1000, 2) if total_energy_std else None

    gpu_mem_gb = av.get("gpu_memory_gb")
    gpu_mem_std = av.get("gpu_memory_gb_stddev")
    mean_gpu_memory_mib = round(gpu_mem_gb * 1024, 2) if gpu_mem_gb is not None else None
    mean_gpu_memory_mib_std = round(gpu_mem_std * 1024, 2) if gpu_mem_std is not None else None

    data_csv = os.path.join(run_dir, "data.csv")
    max_gpu_memory_gb = _max_gpu_memory_gb_from_csv(data_csv)
    max_gpu_memory_mib = round(max_gpu_memory_gb * 1024, 2) if max_gpu_memory_gb is not None else None

    ef_avg = ef.get("avg")
    ef_avg_std = ef.get("avg_stddev")
    ef_mj = round(ef_avg * 1000, 2) if ef_avg is not None else None
    ef_mj_std = round(ef_avg_std * 1000, 2) if ef_avg_std is not None else None

    lines = [
        "Power and timing summary (from profiler)",
        "=" * 50,
        "",
        f"run_duration_s: {_format_with_stdev(run_time_s)}",
        f"total_energy_J: {_format_with_stdev(total_energy_j, total_energy_std)}",
        f"total_energy_kJ: {_format_with_stdev(total_energy_kj, total_energy_kj_std)}",
        f"mean_power_W: {_format_with_stdev(round(mean_power, 2) if mean_power else None, round(mean_power_std, 2) if mean_power_std is not None else None)}",
        f"mean_gpu_memory_MiB: {_format_with_stdev(mean_gpu_memory_mib, mean_gpu_memory_mib_std)}",
        f"max_gpu_memory_MiB: {max_gpu_memory_mib if max_gpu_memory_mib is not None else 'N/A'}",
        f"total_frames: {num_frames if num_frames is not None else 'N/A'}",
        f"energy_per_frame_J: {_format_with_stdev(ef_avg, ef_avg_std)}",
        f"energy_per_frame_mJ: {_format_with_stdev(ef_mj, ef_mj_std)}",
        f"frames_per_second: {num_frames/run_time_s if (num_frames is not None and run_time_s is not None and run_time_s > 0) else 'N/A'}"
    ]
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return txt_path


def _max_gpu_memory_gb_from_csv(csv_path):
    """Read profiler data.csv and return max gpu_memory_gb (column index 5)."""
    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            vals = []
            for row in reader:
                if len(row) > 5 and row[5].strip():
                    try:
                        vals.append(float(row[5]))
                    except ValueError:
                        pass
            return max(vals) if vals else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Launch TartanAir evaluation with optional ONNX runtime"
    )
    parser.add_argument("--test_run_name", type=str, required=True,
                        help="Name for this test run (used in run directory)")
    parser.add_argument("--use_onnx", action="store_true",
                        help="Use ONNXRuntime for fnet/cnet/update")
    parser.add_argument("--onnx_dir", type=str, default="andy/onnx/")
    parser.add_argument("--onnx_tensorrt", action="store_true")

    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--weights", type=str, default="droid.pth")
    parser.add_argument("--buffer", type=int, default=2048, help="Max frames in video buffer; use >= sequence length (e.g. TartanAir ~2k)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Process only first N frames per scene then backend+eval (e.g. 1000 to avoid OOM)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--scene", type=str, default=None)

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.5)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.0)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--motion_damping", type=float, default=0.5)

    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    parser.add_argument("--power_log", action="store_true",
                        help="Profile CPU/GPU power and metrics during run. Requires 'profiler' package.")

    args = parser.parse_args()

    project_root = os.path.abspath(os.getcwd())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.test_run_name)
    runs_base = os.path.join(project_root, "andy", "runs")
    os.makedirs(runs_base, exist_ok=True)

    profiler_instance = None
    if args.power_log:
        if Profiler is None:
            print("Error: --power_log requires the 'profiler' package. Install it or add it to PYTHONPATH.", file=sys.stderr)
            sys.exit(1)
        # Run directory is created by the profiler (andy/runs/YYYY_MM_DD_HHMM_<title>)
        # profiler_instance = Profiler(runs_base, frequency_hz=2.0, title=safe_name)
        profiler_instance = Profiler(runs_base, frequency_hz=2.0, title=safe_name,
             cpu_power_max_w=200.0, 
             gpu_power_max_w=200.0,
             gpu_memory_total_gb= 16.376)
        ref_dir = os.path.join(runs_base, "reference")
        run_dir = profiler_instance.start(use_reference=True)
        print(f"Profiler started. Run directory: {run_dir}")
    else:
        run_dirname = f"{timestamp}_{safe_name}"
        run_dir = os.path.join(runs_base, run_dirname)
        os.makedirs(run_dir, exist_ok=True)

    # Build parameter dict for metadata (run_dir may be from profiler or our own)
    params = {
        "test_run_name": args.test_run_name,
        "run_dir": run_dir,
        "use_onnx": args.use_onnx,
        "onnx_dir": args.onnx_dir,
        "onnx_tensorrt": args.onnx_tensorrt,
        "datapath": args.datapath,
        "gt_path": args.gt_path,
        "weights": args.weights,
        "buffer": args.buffer,
        "max_frames": args.max_frames,
        "image_size": list(args.image_size),
        "stereo": args.stereo,
        "disable_vis": args.disable_vis,
        "plot_curve": args.plot_curve,
        "scene": args.scene,
        "beta": args.beta,
        "filter_thresh": args.filter_thresh,
        "warmup": args.warmup,
        "keyframe_thresh": args.keyframe_thresh,
        "frontend_thresh": args.frontend_thresh,
        "frontend_window": args.frontend_window,
        "frontend_radius": args.frontend_radius,
        "frontend_nms": args.frontend_nms,
        "backend_thresh": args.backend_thresh,
        "backend_radius": args.backend_radius,
        "backend_nms": args.backend_nms,
        "motion_damping": args.motion_damping,
        "upsample": args.upsample,
        "asynchronous": args.asynchronous,
        "frontend_device": args.frontend_device,
        "backend_device": args.backend_device,
        "timestamp": timestamp,
    }

    # Write human-readable metadata.txt (launch params) so run_dir has it from the start
    metadata_txt_path = os.path.join(run_dir, "metadata.txt")
    with open(metadata_txt_path, "w") as f:
        f.write("TartanAir evaluation run metadata\n")
        f.write("=" * 60 + "\n\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

    # Build command for test_tartanair_andy.py
    test_script = os.path.join(project_root, "evaluation_scripts", "test_tartanair_andy.py")
    cmd = [
        sys.executable, test_script,
        "--run_dir", run_dir,
        "--datapath", args.datapath,
        "--gt_path", args.gt_path,
        "--weights", args.weights,
        "--buffer", str(args.buffer),
        "--image_size", str(args.image_size[0]), str(args.image_size[1]),
        "--beta", str(args.beta),
        "--filter_thresh", str(args.filter_thresh),
        "--warmup", str(args.warmup),
        "--keyframe_thresh", str(args.keyframe_thresh),
        "--frontend_thresh", str(args.frontend_thresh),
        "--frontend_window", str(args.frontend_window),
        "--frontend_radius", str(args.frontend_radius),
        "--frontend_nms", str(args.frontend_nms),
        "--backend_thresh", str(args.backend_thresh),
        "--backend_radius", str(args.backend_radius),
        "--backend_nms", str(args.backend_nms),
        "--motion_damping", str(args.motion_damping),
        "--frontend_device", args.frontend_device,
        "--backend_device", args.backend_device,
    ]
    if args.stereo:
        cmd.append("--stereo")
    if args.disable_vis:
        cmd.append("--disable_vis")
    if args.plot_curve:
        cmd.append("--plot_curve")
    if args.scene:
        cmd.extend(["--scene", args.scene])
    if getattr(args, "max_frames", None) is not None:
        cmd.extend(["--max_frames", str(args.max_frames)])
    if args.upsample:
        cmd.append("--upsample")
    if args.asynchronous:
        cmd.append("--asynchronous")
    if args.use_onnx:
        cmd.append("--use_onnx")
        cmd.extend(["--onnx_fnet", os.path.join(args.onnx_dir, "fnet.onnx")])
        cmd.extend(["--onnx_cnet", os.path.join(args.onnx_dir, "cnet.onnx")])
        cmd.extend(["--onnx_update", os.path.join(args.onnx_dir, "update_core.onnx")])
    if args.onnx_tensorrt:
        cmd.append("--onnx_tensorrt")

    print(f"Run directory: {run_dir}")
    print(f"Metadata written to {metadata_txt_path}")
    print("Launching test_tartanair_andy.py...")
    print(" ".join(cmd))

    result = subprocess.run(cmd, cwd=project_root)

    # If we used the profiler, stop it and merge launch params into metadata
    total_frames = None
    ate_path = os.path.join(run_dir, "ate_results.json")
    if os.path.isfile(ate_path):
        with open(ate_path) as f:
            ate_data = json.load(f)
        total_frames = ate_data.get("total_frames")

    if profiler_instance is not None:
        profiler_instance.stop(num_frames=total_frames)
        print("Profiler stopped. Power log and plot written by profiler (data.csv, plot.png).")

        # Merge launch params into metadata.json so one file has both profiler and run params
        meta_path = os.path.join(run_dir, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["launch_params"] = params
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            # Note in metadata.txt where profiler data lives (metadata.json, power_summary.txt)
            with open(metadata_txt_path, "a") as f:
                f.write("(Profiler data: metadata.json; human-readable power summary: power_summary.txt)\n")
            power_summary_txt = _write_power_summary_txt(run_dir, meta)
            print(f"Power summary (human-readable) saved to {power_summary_txt}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
