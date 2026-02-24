#!/usr/bin/env python3
"""
Quantize DROID-SLAM ONNX models (fnet, cnet, update_core) from andy/onnx/ and save
to a new folder with a quantization suffix (e.g. andy/onnx_dynamic_int8/ or andy/onnx_static_int8/).

Supports:
  - int8/uint8 dynamic: ONNX Runtime dynamic quantization (no calibration).
    Uses DynamicQuantizeLinear; not supported by TensorRT.
  - int8 static: Static INT8 quantization with calibration. Uses QuantizeLinear
    with constant scale/zero_point (no DynamicQuantizeLinear), suitable for
    TensorRT execution provider.
  - fp16: Float16 conversion via onnxconverter-common (pip install onnxconverter-common).

Use the output directory as --onnx_dir when running launch_tartanair.py with --use_onnx.
"""

import argparse
import os
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class DROIDImageCalibrationDataReader(CalibrationDataReader):
    """Yields calibration batches for fnet/cnet: input 'images' with shape (1, N, 3, H, W), float32 in [0, 255]."""

    def __init__(self, model_path, shape=(1, 1, 3, 384, 512), num_batches=20, rng=None):
        self.model_path = model_path
        self.shape = shape
        self.num_batches = num_batches
        self.rng = np.random.default_rng(rng)
        self._input_name = "images"
        self._batch_index = 0
        model = onnx.load(model_path)
        if model.graph.input:
            self._input_name = model.graph.input[0].name

    def get_next(self):
        if self._batch_index >= self.num_batches:
            return None
        # DROID expects images in [0, 255] (BGR, OpenCV convention)
        data = self.rng.uniform(low=0.0, high=255.0, size=self.shape).astype(np.float32)
        self._batch_index += 1
        return {self._input_name: data}

    def rewind(self):
        self._batch_index = 0


# Default shapes for update_core: batch=1, edges=2, h8=48, w8=64 (1/8 of 384x512), corr_channels=196
UPDATE_CORE_DEFAULT_SHAPE = {
    "net": (1, 2, 128, 48, 64),
    "inp": (1, 2, 128, 48, 64),
    "corr": (1, 2, 196, 48, 64),
    "flow": (1, 2, 4, 48, 64),
}


class DROIDUpdateCalibrationDataReader(CalibrationDataReader):
    """Yields calibration batches for update_core: inputs net, inp, corr, flow."""

    def __init__(self, model_path, shapes=None, num_batches=20, rng=None):
        self.model_path = model_path
        self.shapes = shapes or UPDATE_CORE_DEFAULT_SHAPE.copy()
        self.num_batches = num_batches
        self.rng = np.random.default_rng(rng)
        self._batch_index = 0
        self._input_names = []
        model = onnx.load(model_path)
        for inp in model.graph.input:
            self._input_names.append(inp.name)
        for name in self._input_names:
            if name not in self.shapes:
                # fallback: use (1, 2, 128, 48, 64) for unknown inputs
                self.shapes[name] = (1, 2, 128, 48, 64)

    def get_next(self):
        if self._batch_index >= self.num_batches:
            return None
        feed = {}
        for name in self._input_names:
            shape = self.shapes[name]
            feed[name] = self.rng.standard_normal(size=shape).astype(np.float32)
        self._batch_index += 1
        return feed

    def rewind(self):
        self._batch_index = 0


def main():
    parser = argparse.ArgumentParser(
        description="Quantize fnet.onnx, cnet.onnx, and update_core.onnx from an ONNX directory."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing fnet.onnx, cnet.onnx, update_core.onnx (default: andy/onnx)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="dynamic_int8",
        help="Suffix for output directory: input dir parent + onnx_<suffix> (default: dynamic_int8)",
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        choices=["uint8", "int8", "fp16"],
        default="int8",
        help="Weight type: int8/uint8 = dynamic quantization; fp16 = float16 conversion (default: int8)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static INT8 quantization (requires calibration). Produces QuantizeLinear with constant "
        "scale/zero_point, suitable for TensorRT. Ignored if --weight-type is fp16.",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=20,
        help="Number of calibration batches for static quantization (default: 20)",
    )
    parser.add_argument(
        "--calibration-shape",
        type=str,
        default="1,1,3,384,512",
        help="Comma-separated shape for calibration input [batch,frames,channels,height,width] for fnet/cnet (default: 1,1,3,384,512)",
    )
    args = parser.parse_args()
    if args.static and args.weight_type in ("int8", "uint8") and args.suffix == "dynamic_int8":
        args.suffix = "static_int8"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = os.path.join(script_dir, "onnx")
    input_dir = os.path.abspath(input_dir)

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    fnet_path = os.path.join(input_dir, "fnet.onnx")
    cnet_path = os.path.join(input_dir, "cnet.onnx")
    update_core_path = os.path.join(input_dir, "update_core.onnx")
    for name, p in [("fnet", fnet_path), ("cnet", cnet_path), ("update_core", update_core_path)]:
        if not os.path.isfile(p):
            raise SystemExit(f"Missing {name} model: {p}")

    # Output dir: same parent as input_dir, name = basename_input + "_" + suffix
    # e.g. andy/onnx -> andy/onnx_dynamic_int8
    parent = os.path.dirname(input_dir)
    base = os.path.basename(input_dir.rstrip(os.sep))
    output_dir = os.path.join(parent, f"{base}_{args.suffix}")
    os.makedirs(output_dir, exist_ok=True)

    models = [
        ("fnet", fnet_path, os.path.join(output_dir, "fnet.onnx")),
        ("cnet", cnet_path, os.path.join(output_dir, "cnet.onnx")),
        ("update_core", update_core_path, os.path.join(output_dir, "update_core.onnx")),
    ]

    if args.weight_type == "fp16":
        try:
            from onnxconverter_common import float16 as float16_conv  # type: ignore[import-untyped]
        except ImportError:
            raise SystemExit(
                "FP16 conversion requires onnxconverter-common. Install with: pip install onnxconverter-common"
            )
        for name, src, dst in models:
            print(f"Converting {name} to FP16: {src} -> {dst}")
            model = onnx.load(src)
            model_fp16 = float16_conv.convert_float_to_float16(model, keep_io_types=True)
            onnx.save(model_fp16, dst)
            print(f"  Saved: {dst}")
    else:
        weight_type = QuantType.QInt8 if args.weight_type == "int8" else QuantType.QUInt8
        if args.static:
            # Static INT8: calibration + QuantizeLinear with constant scale/zero_point (no DynamicQuantizeLinear).
            # TensorRT-friendly. Use QDQ format and symmetric QInt8 for GPU/TRT.
            calib_shape = tuple(int(x) for x in args.calibration_shape.split(","))
            if len(calib_shape) != 5:
                raise SystemExit(
                    f"--calibration-shape must be 5 values (batch,frames,channels,height,width), got {args.calibration_shape}"
                )
            for name, src, dst in models:
                print(f"Static quantizing {name}: {src} -> {dst}")
                with tempfile.TemporaryDirectory(prefix="droid_quant_") as tmpdir:
                    shape_inferred = os.path.join(tmpdir, "shape_inferred.onnx")
                    quant_pre_process(
                        src,
                        shape_inferred,
                        skip_optimization=False,
                        skip_onnx_shape=False,
                    )
                    if name == "update_core":
                        calib_reader = DROIDUpdateCalibrationDataReader(
                            shape_inferred,
                            num_batches=args.calibration_batches,
                        )
                    else:
                        calib_reader = DROIDImageCalibrationDataReader(
                            shape_inferred,
                            shape=calib_shape,
                            num_batches=args.calibration_batches,
                        )
                    quantize_static(
                        shape_inferred,
                        dst,
                        calib_reader,
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=True,
                        reduce_range=False,
                        calibrate_method=CalibrationMethod.MinMax,
                        extra_options={
                            "ActivationSymmetric": True,
                            "WeightSymmetric": True,
                            # TensorRT's DequantizeLinear only accepts Int8 (not Int32). ORT quantizes
                            # biases as Int32 by default, which causes TRT import to fail. Keeping
                            # biases in float avoids that and is standard for TensorRT-friendly QDQ.
                            "QuantizeBias": False,
                        },
                    )
                print(f"  Saved: {dst}")
        else:
            for name, src, dst in models:
                print(f"Dynamic quantizing {name}: {src} -> {dst}")
                quantize_dynamic(
                    src,
                    dst,
                    weight_type=weight_type,
                    per_channel=True,
                    reduce_range=False,
                )
                print(f"  Saved: {dst}")

    print(f"Done. Output models in: {output_dir}")
    print(f"Run with: --use_onnx --onnx_dir {output_dir}")


if __name__ == "__main__":
    main()
