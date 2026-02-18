#!/usr/bin/env python3
"""
Attempt to convert the whole DroidNet model (from droid.pth / droid_net.py)
into a single ONNX model.

Unlike onnx_conversion.ipynb which exports fnet, cnet, and update_core as
three separate ONNX files, this script tries to produce one ONNX that contains
all exportable neural components.

EXPORT BLOCKERS (cannot be in ONNX):
  - CorrBlock: uses droid_backends (custom CUDA) for correlation indexing
  - BA (bundle adjustment): uses lietorch SE3, projective_ops with jacobians,
    torch_scatter, chol (Schur solve)
  - projective_transform: uses lietorch SE3/Sim3 for Lie group action

APPROACH:
  Export a single "DroidNetNeuralBackbone" ONNX that combines:
    - fnet + cnet (feature extraction)
    - update (including GraphAgg with eta, upmask)
  with correlation and motion as external inputs (CorrBlock output is fed in).
  BA and projective ops remain outside ONNX.

Usage:
  From project root: python andy/onnx_droid_net_full.py
  Or from andy/:     python onnx_droid_net_full.py
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "droid_slam"))

import torch
import torch.nn as nn
import onnx


def _load_droid_state_dict(pth_path: str) -> dict:
    """Load and preprocess state dict from droid.pth."""
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ["state_dict", "model"]):
        state_dict = ckpt.get("model", ckpt.get("state_dict"))
    else:
        state_dict = ckpt

    new_state_dict = collections.OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in state_dict.items()
    ])

    for key in [
        "update.weight.2.weight", "update.weight.2.bias",
        "update.delta.2.weight", "update.delta.2.bias",
    ]:
        if key in new_state_dict:
            new_state_dict[key] = new_state_dict[key][:2]

    return new_state_dict


# --- Normalization (matches extract_features) ---
class _NormalizeImages(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, images):
        images = images[:, :, [2, 1, 0]] / 255.0
        return (images - self.mean) / self.std


# --- DroidNet neural backbone: fnet + cnet + update (single ONNX) ---
class DroidNetNeuralBackbone(nn.Module):
    """
    Single ONNX-compatible model combining:
      - fnet: feature maps
      - cnet: context net -> net, inp
      - update: net, inp, corr, flow, ii, jj -> net_out, delta, weight, eta, upmask

    CorrBlock output (corr) and motion (flow) are inputs; CorrBlock uses
    droid_backends and cannot be exported.
    """

    def __init__(self, droid_net):
        super().__init__()
        self.norm = _NormalizeImages()
        self.fnet = droid_net.fnet
        self.cnet = droid_net.cnet
        self.update = droid_net.update

    def forward(self, images, corr, flow, ii, jj):
        # Feature extraction (matches DroidNet.extract_features)
        x = self.norm(images)
        fmaps = self.fnet(x)
        cnet_out = self.cnet(x)
        net, inp = cnet_out.split([128, 128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # Index by source frame (ii) - update expects net[:,ii], inp[:,ii] per edge
        net = net[:, ii]
        inp = inp[:, ii]
        net, delta, weight, eta, upmask = self.update(net, inp, corr, flow, ii, jj)

        return fmaps, net, inp, delta, weight, eta, upmask


# --- Features only: fnet + cnet (for pipeline fnet/cnet replacement) ---
class DroidNetFeaturesOnly(nn.Module):
    """Features extraction only: images -> fmaps, net, inp. For pipeline use."""

    def __init__(self, droid_net):
        super().__init__()
        self.norm = _NormalizeImages()
        self.fnet = droid_net.fnet
        self.cnet = droid_net.cnet

    def forward(self, images):
        x = self.norm(images)
        fmaps = self.fnet(x)
        cnet_out = self.cnet(x)
        net, inp = cnet_out.split([128, 128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


# --- Update only with GraphAgg (for pipeline update replacement) ---
class DroidNetUpdateOnly(nn.Module):
    """Update only: net, inp, corr, flow, ii, jj -> net_out, delta, weight, eta, upmask."""

    def __init__(self, droid_net):
        super().__init__()
        self.update = droid_net.update

    def forward(self, net, inp, corr, flow, ii, jj):
        return self.update(net, inp, corr, flow, ii, jj)


# --- Variant without graph aggregation (no scatter_mean) ---
class DroidNetNeuralBackboneNoGraphAgg(nn.Module):
    """
    Same as above but update is called with ii=None, jj=None to skip GraphAgg.
    Use when torch_scatter.scatter_mean is not ONNX-exportable.
    Outputs: fmaps, net_out, delta, weight (no eta, upmask).
    """

    def __init__(self, droid_net):
        super().__init__()
        self.norm = _NormalizeImages()
        self.fnet = droid_net.fnet
        self.cnet = droid_net.cnet
        self.update = droid_net.update

    def forward(self, images, corr, flow, ii):
        x = self.norm(images)
        fmaps = self.fnet(x)
        cnet_out = self.cnet(x)
        net, inp = cnet_out.split([128, 128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        net = net[:, ii]
        inp = inp[:, ii]

        net_out, delta, weight = self.update(net, inp, corr, flow)
        return fmaps, net_out, delta, weight


def create_dummy_inputs(batch=1, num_frames=3, num_edges=3, h=240, w=320, device="cpu"):
    """Create dummy inputs for ONNX tracing."""
    h8, w8 = h // 8, w // 8
    corr_ch = 4 * (2 * 3 + 1) ** 2  # 196

    images = (torch.rand(batch, num_frames, 3, h, w, device=device) * 255.0).to(torch.float32)
    corr = torch.randn(batch, num_edges, corr_ch, h8, w8, device=device)
    flow = torch.randn(batch, num_edges, 4, h8, w8, device=device)
    ii = torch.arange(num_edges, dtype=torch.long, device=device)
    jj = torch.arange(num_edges, dtype=torch.long, device=device)

    return images, corr, flow, ii, jj


def attempt_full_forward_export(model, device, output_dir):
    """
    Attempt to export the full DroidNet.forward().
    Expected to fail on CorrBlock or BA/projective ops.
    """
    print("\n" + "=" * 60)
    print("Attempt 1: Full DroidNet.forward() export")
    print("=" * 60)
    try:
        from geom.graph_utils import graph_to_edge_list
        from lietorch import SE3
        import geom.projective_ops as pops

        B, N = 1, 3
        H, W = 240, 320
        graph = {0: [1], 1: [2], 2: [0]}
        ii, jj, _ = graph_to_edge_list(graph)
        ii = ii.to(device)
        jj = jj.to(device)

        images = (torch.rand(B, N, 3, H, W, device=device) * 255.0).float()
        intrinsics = torch.tensor([[[320.0, 320.0, 160.0, 120.0]]] * B, device=device).float()
        Gs = SE3(torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1))
        disps = torch.ones(B, N, H // 8, W // 8, device=device) * 0.5

        out_path = os.path.join(output_dir, "droid_net_full_attempt.onnx")
        torch.onnx.export(
            model,
            (Gs, images, disps, intrinsics),
            out_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["Gs", "images", "disps", "intrinsics"],
            output_names=["Gs_list", "disp_list", "residual_list"],
            dynamic_axes={"images": {0: "batch", 1: "frames", 3: "height", 4: "width"}},
        )
        print(f"  SUCCESS: Saved {out_path}")
        return True
    except Exception as e:
        print(f"  FAILED (expected): {e}")
        print("  Reason: CorrBlock uses droid_backends; BA uses lietorch/torch_scatter.")
        return False


def export_neural_backbone(model, device, output_path, use_graph_agg=True):
    """
    Export DroidNetNeuralBackbone (or NoGraphAgg variant) to a single ONNX.
    """
    B, num_frames, num_edges = 1, 3, 3
    H, W = 240, 320
    images, corr, flow, ii, jj = create_dummy_inputs(
        batch=B, num_frames=num_frames, num_edges=num_edges,
        h=H, w=W, device=device,
    )

    if use_graph_agg:
        export_model = DroidNetNeuralBackbone(model)
        dummy = (images, corr, flow, ii, jj)
        input_names = ["images", "corr", "flow", "ii", "jj"]
        output_names = ["fmaps", "net_out", "inp", "delta", "weight", "eta", "upmask"]
        dynamic_axes = {
            "images": {0: "batch", 1: "frames", 3: "height", 4: "width"},
            "corr": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "flow": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "ii": {0: "edges"},
            "jj": {0: "edges"},
            "fmaps": {0: "batch", 1: "frames", 3: "h8", 4: "w8"},
            "net_out": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "inp": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "delta": {0: "batch", 1: "edges", 2: "h8", 3: "w8"},
            "weight": {0: "batch", 1: "edges", 2: "h8", 3: "w8"},
            "eta": {0: "batch", 1: "keyframes", 3: "h8", 4: "w8"},
            "upmask": {0: "batch", 1: "keyframes", 3: "h8", 4: "w8"},
        }
    else:
        export_model = DroidNetNeuralBackboneNoGraphAgg(model)
        dummy = (images, corr, flow, ii)
        input_names = ["images", "corr", "flow", "ii"]
        output_names = ["fmaps", "net_out", "delta", "weight"]
        dynamic_axes = {
            "images": {0: "batch", 1: "frames", 3: "height", 4: "width"},
            "corr": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "flow": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "ii": {0: "edges"},
            "fmaps": {0: "batch", 1: "frames", 3: "h8", 4: "w8"},
            "net_out": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "delta": {0: "batch", 1: "edges", 2: "h8", 3: "w8"},
            "weight": {0: "batch", 1: "edges", 2: "h8", 3: "w8"},
        }

    export_model = export_model.to(device).eval()

    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return output_path


def export_features_only(model, device, output_path):
    """Export features-only ONNX: images -> fmaps, net, inp."""
    B, N, H, W = 1, 3, 240, 320
    images = (torch.rand(B, N, 3, H, W, device=device) * 255.0).float()
    export_model = DroidNetFeaturesOnly(model).to(device).eval()
    torch.onnx.export(
        export_model,
        (images,),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["fmaps", "net", "inp"],
        dynamic_axes={
            "images": {0: "batch", 1: "frames", 3: "height", 4: "width"},
            "fmaps": {0: "batch", 1: "frames", 3: "h8", 4: "w8"},
            "net": {0: "batch", 1: "frames", 3: "h8", 4: "w8"},
            "inp": {0: "batch", 1: "frames", 3: "h8", 4: "w8"},
        },
    )
    return output_path


def export_update_only(model, device, output_path):
    """Export update-only ONNX: net, inp, corr, flow, ii, jj -> net_out, delta, weight, eta, upmask."""
    B, num_edges, h8, w8 = 1, 3, 30, 40
    corr_ch = 4 * (2 * 3 + 1) ** 2
    net = torch.randn(B, num_edges, 128, h8, w8, device=device)
    inp = torch.randn(B, num_edges, 128, h8, w8, device=device)
    corr = torch.randn(B, num_edges, corr_ch, h8, w8, device=device)
    flow = torch.randn(B, num_edges, 4, h8, w8, device=device)
    ii = torch.arange(num_edges, dtype=torch.long, device=device)
    jj = torch.arange(num_edges, dtype=torch.long, device=device)

    export_model = DroidNetUpdateOnly(model).to(device).eval()
    torch.onnx.export(
        export_model,
        (net, inp, corr, flow, ii, jj),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["net", "inp", "corr", "flow", "ii", "jj"],
        output_names=["net_out", "delta", "weight", "eta", "upmask"],
        dynamic_axes={
            "net": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "inp": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "corr": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "flow": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "ii": {0: "edges"},
            "jj": {0: "edges"},
            "net_out": {0: "batch", 1: "edges", 3: "h8", 4: "w8"},
            "delta": {0: "batch", 1: "edges", 2: "h8", 3: "w8"},
            "weight": {0: "batch", 1: "edges", 2: "h8", 3: "w8"},
            "eta": {0: "batch", 1: "keyframes", 3: "h8", 4: "w8"},
            "upmask": {0: "batch", 1: "keyframes", 3: "h8", 4: "w8"},
        },
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert DroidNet to single ONNX")
    parser.add_argument(
        "--pth",
        type=str,
        default=os.path.join(PROJECT_ROOT, "droid.pth"),
        help="Path to droid.pth checkpoint",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(PROJECT_ROOT, "droid_net.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument(
        "--no-graph-agg",
        action="store_true",
        help="Skip graph aggregation (no eta, upmask) if scatter_mean fails",
    )
    parser.add_argument(
        "--try-full",
        action="store_true",
        help="Also attempt full DroidNet.forward export (will fail)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for export (CPU recommended for ONNX compatibility)",
    )
    parser.add_argument(
        "--no-export-split",
        action="store_true",
        help="Skip exporting droid_net_features.onnx and droid_net_update.onnx",
    )
    args = parser.parse_args()
    export_split = not args.no_export_split

    device = torch.device(args.device)
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Loading DroidNet from", args.pth)
    from droid_slam.droid_net import DroidNet

    model = DroidNet()
    state_dict = _load_droid_state_dict(args.pth)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model.eval().to(device)

    if args.try_full:
        attempt_full_forward_export(model, device, output_dir)

    print("\n" + "=" * 60)
    print("Attempt 2: DroidNetNeuralBackbone (single ONNX)")
    print("=" * 60)

    use_graph_agg = not args.no_graph_agg
    try:
        path = export_neural_backbone(model, device, args.out, use_graph_agg=use_graph_agg)
        print(f"  Saved: {path}")
        m = onnx.load(path)
        onnx.checker.check_model(m)
        print("  ONNX check passed.")
    except Exception as e:
        if use_graph_agg and "scatter" in str(e).lower():
            print(f"  Failed (likely scatter_mean): {e}")
            print("  Retrying without graph aggregation (--no-graph-agg)...")
            fallback_path = args.out.replace(".onnx", "_no_graph_agg.onnx")
            path = export_neural_backbone(model, device, fallback_path, use_graph_agg=False)
            print(f"  Saved fallback: {path}")
            m = onnx.load(path)
            onnx.checker.check_model(m)
            print("  ONNX check passed.")
        else:
            raise

    if export_split:
        print("\n" + "=" * 60)
        print("Export split models (for pipeline use)")
        print("=" * 60)
        out_dir = os.path.dirname(args.out) or "."
        features_path = os.path.join(out_dir, "droid_net_features.onnx")
        update_path = os.path.join(out_dir, "droid_net_update.onnx")
        try:
            export_features_only(model, device, features_path)
            print(f"  Saved: {features_path}")
            onnx.checker.check_model(onnx.load(features_path))
            export_update_only(model, device, update_path)
            print(f"  Saved: {update_path}")
            onnx.checker.check_model(onnx.load(update_path))
        except Exception as e:
            if "scatter" in str(e).lower():
                print(f"  Update export failed (scatter_mean): {e}")
                print("  Use --use_onnx with fnet.onnx, cnet.onnx, update_core.onnx instead.")
            else:
                raise

    print("\nDone.")


if __name__ == "__main__":
    main()
