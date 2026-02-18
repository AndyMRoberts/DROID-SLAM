#!/usr/bin/env python3
"""
Generic launcher for TartanAir evaluation.

Run from the project root (where demo.py lives). Creates a run directory
andy/runs/YYYYMMDD_HHMM_<test_run_name>, writes metadata.txt with all parameters,
then runs evaluation_scripts/test_tartanair_andy.py with outputs directed there.

Optional power logging: compile andy/metric_measurement/power.cc first:
  g++ -o power andy/metric_measurement/power.cc
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime


def _parse_power_log(csv_path, run_duration_s, total_frames):
    """Parse power log CSV and compute run duration, total energy, energy per frame."""
    summary = {}
    try:
        with open(csv_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) < 2:
            summary["error"] = "Power log empty or header only"
            return summary

        header = lines[0]
        rows = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    t = float(parts[0])
                    gpu_pwr = float(parts[1])
                    cpu_pwr = float(parts[2])
                    total_pwr = float(parts[3])
                    gpu_mem = float(parts[5]) if len(parts) >= 6 else None
                    rows.append((t, gpu_pwr, cpu_pwr, total_pwr, gpu_mem))
                except ValueError:
                    continue

        if not rows:
            summary["error"] = "No valid power readings"
            return summary

        # Power log: Time (s), GPU Power, CPU Power, Total Power [, GPU Memory (MiB)]
        # C++ outputs in W (nvidia-smi and RAPL-derived). Integrate for energy.
        duration_from_log = rows[-1][0] - rows[0][0]
        total_energy_j = 0.0
        for i in range(len(rows) - 1):
            dt = rows[i + 1][0] - rows[i][0]
            avg_pwr = (rows[i][3] + rows[i + 1][3]) / 2.0
            total_energy_j += avg_pwr * dt

        summary["run_duration_s"] = round(run_duration_s, 2)
        summary["power_log_duration_s"] = round(duration_from_log, 2)
        summary["total_energy_J"] = round(total_energy_j, 2)
        summary["total_energy_kJ"] = round(total_energy_j / 1000, 2)
        summary["mean_power_W"] = round(total_energy_j / duration_from_log, 2) if duration_from_log > 0 else 0

        gpu_mem_vals = [r[4] for r in rows if r[4] is not None]
        if gpu_mem_vals:
            summary["mean_gpu_memory_MiB"] = round(sum(gpu_mem_vals) / len(gpu_mem_vals), 2)
            summary["max_gpu_memory_MiB"] = round(max(gpu_mem_vals), 2)

        if total_frames and total_frames > 0:
            energy_per_frame_J = total_energy_j / total_frames
            summary["total_frames"] = total_frames
            summary["energy_per_frame_J"] = round(energy_per_frame_J, 4)
            summary["energy_per_frame_mJ"] = round(energy_per_frame_J * 1000, 2)
        else:
            summary["total_frames"] = "unknown (ate_results.json not found or empty)"
            summary["energy_per_frame_J"] = "N/A"
            summary["energy_per_frame_mJ"] = "N/A"

    except Exception as e:
        summary["error"] = str(e)
    return summary


def _plot_power_log(csv_path, png_path):
    """Create power plot similar to view_power_log.ipynb and save to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(csv_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) < 2:
            return False

        # Parse CSV: Time (s), GPU Power, CPU Power, Total Power, Average Power [, GPU Memory (MiB)]
        x, gpu_power, cpu_power, total_power, avg_power, gpu_memory = [], [], [], [], [], []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 5:
                try:
                    x.append(float(parts[0]))
                    gpu_power.append(float(parts[1]))
                    cpu_power.append(float(parts[2]))
                    total_power.append(float(parts[3]))
                    avg_power.append(float(parts[4]))
                    gpu_memory.append(float(parts[5]) if len(parts) >= 6 else float("nan"))
                except ValueError:
                    continue

        if not x:
            return False

        plt.style.use("dark_background")
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(x, gpu_power, label="GPU Power")
        ax1.plot(x, cpu_power, label="CPU Power")
        ax1.plot(x, avg_power, label="Avg Power")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Power (W)")
        ax1.set_ylim(0, max(500, max(total_power) * 1.1) if total_power else 500)
        ax1.legend(loc="upper left")

        if gpu_memory and any(m == m for m in gpu_memory):  # any non-nan (nan != nan)
            ax2 = ax1.twinx()
            ax2.plot(x, gpu_memory, color="cyan", label="GPU Memory", linestyle="--", alpha=0.8)
            ax2.set_ylabel("GPU Memory (MiB)")
            ax2.legend(loc="upper right")

        plt.title("Power and GPU memory during run")
        fig.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Launch TartanAir evaluation with optional ONNX runtime"
    )
    parser.add_argument("--test_run_name", type=str, required=True,
                        help="Name for this test run (used in run directory)")
    parser.add_argument("--use_onnx", action="store_true",
                        help="Use ONNXRuntime for fnet/cnet/update")
    parser.add_argument("--onnx_fnet", type=str, default="fnet.onnx")
    parser.add_argument("--onnx_cnet", type=str, default="cnet.onnx")
    parser.add_argument("--onnx_update", type=str, default="update_core.onnx")
    parser.add_argument("--onnx_tensorrt", action="store_true")

    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--weights", type=str, default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
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
                        help="Log CPU/GPU power during run. Requires compiled power binary (g++ -o power andy/metric_measurement/power.cc)")

    args = parser.parse_args()

    # Create run directory (project root = cwd when launcher is run)
    project_root = os.path.abspath(os.getcwd())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.test_run_name)
    run_dirname = f"{timestamp}_{safe_name}"
    runs_base = os.path.join(project_root, "andy", "runs")
    run_dir = os.path.join(runs_base, run_dirname)
    os.makedirs(runs_base, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # Build parameter dict for metadata
    params = {
        "test_run_name": args.test_run_name,
        "run_dir": run_dir,
        "use_onnx": args.use_onnx,
        "onnx_fnet": args.onnx_fnet,
        "onnx_cnet": args.onnx_cnet,
        "onnx_update": args.onnx_update,
        "onnx_tensorrt": args.onnx_tensorrt,
        "datapath": args.datapath,
        "gt_path": args.gt_path,
        "weights": args.weights,
        "buffer": args.buffer,
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

    # Write metadata.txt
    with open(os.path.join(run_dir, "metadata.txt"), "w") as f:
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
    if args.upsample:
        cmd.append("--upsample")
    if args.asynchronous:
        cmd.append("--asynchronous")
    if args.use_onnx:
        cmd.append("--use_onnx")
        cmd.extend(["--onnx_fnet", args.onnx_fnet])
        cmd.extend(["--onnx_cnet", args.onnx_cnet])
        cmd.extend(["--onnx_update", args.onnx_update])
    if args.onnx_tensorrt:
        cmd.append("--onnx_tensorrt")

    print(f"Run directory: {run_dir}")
    print(f"Metadata written to {os.path.join(run_dir, 'metadata.txt')}")

    power_proc = None
    power_log_path = os.path.join(run_dir, "power_log.csv")
    if args.power_log:
        power_bin = os.path.join(project_root, "power")
        if not os.path.isfile(power_bin):
            power_bin = os.path.join(project_root, "andy", "metric_measurement", "power")
        if os.path.isfile(power_bin):
            print("Starting power logger...")
            power_proc = subprocess.Popen(
                ['sudo', power_bin, power_log_path],
                cwd=project_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.5)  # let power logger initialize
            if power_proc.poll() is not None:
                _, err = power_proc.communicate()
                print(f"Power logger failed to start: {err.decode()}", file=sys.stderr)
                power_proc = None
        else:
            print("Power binary not found. Compile with: g++ -o power andy/metric_measurement/power.cc", file=sys.stderr)
            power_proc = None

    print("Launching test_tartanair_andy.py...")
    print(" ".join(cmd))

    run_start = time.perf_counter()
    result = subprocess.run(cmd, cwd=project_root)
    run_duration_s = time.perf_counter() - run_start

    if power_proc is not None:
        power_proc.send_signal(signal.SIGINT)
        try:
            power_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            power_proc.kill()
        print("Power logger stopped.")

    # Load results and compute power metrics
    total_frames = None
    ate_path = os.path.join(run_dir, "ate_results.json")
    if os.path.isfile(ate_path):
        with open(ate_path) as f:
            ate_data = json.load(f)
        total_frames = ate_data.get("total_frames")

    power_summary = {}
    if os.path.isfile(power_log_path):
        power_summary = _parse_power_log(power_log_path, run_duration_s, total_frames)
        summary_path = os.path.join(run_dir, "power_summary.json")
        with open(summary_path, "w") as f:
            json.dump(power_summary, f, indent=2)
        txt_path = os.path.join(run_dir, "power_summary.txt")
        with open(txt_path, "w") as f:
            f.write("Power and timing summary\n")
            f.write("=" * 50 + "\n\n")
            for k, v in power_summary.items():
                f.write(f"{k}: {v}\n")
        print(f"Power summary saved to {summary_path}")

        plot_path = os.path.join(run_dir, "power_plot.png")
        if _plot_power_log(power_log_path, plot_path):
            print(f"Power plot saved to {plot_path}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
