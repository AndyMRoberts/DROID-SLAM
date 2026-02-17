import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


def _ensure_env_lib_in_ld_library_path() -> None:
    """Ensure $CONDA_PREFIX/lib (or sys.executable env) is visible to dlopen().

    This keeps the install 'safe' (env-scoped) while making sure ONNXRuntime can
    locate cuDNN / CUDA libs that were installed into the conda env.
    """
    # Prefer the environment of the running interpreter
    py_prefix = os.path.dirname(os.path.dirname(sys.executable))
    lib_dir = os.path.join(py_prefix, "lib")

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(":") if p]
    if lib_dir not in parts and os.path.isdir(lib_dir):
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld if ld else "")


def _preload_env_cudnn() -> None:
    """Preload cuDNN from the running env so ORT CUDA EP can resolve it.

    In some setups, dlopen() of ORT's CUDA provider fails to *locate* libcudnn
    via the loader search path. Preloading by absolute path is env-scoped and
    avoids changing any system CUDA install.
    """
    import ctypes

    py_prefix = os.path.dirname(os.path.dirname(sys.executable))
    cudnn_path = os.path.join(py_prefix, "lib", "libcudnn.so.9")
    if os.path.exists(cudnn_path):
        # RTLD_GLOBAL helps satisfy downstream NEEDED dependencies by SONAME.
        ctypes.CDLL(cudnn_path, mode=ctypes.RTLD_GLOBAL)


def _parse_cuda_device_id(device: Union[str, torch.device]) -> int:
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return 0
        return 0 if device.index is None else int(device.index)

    # strings like: "cuda", "cuda:0", "cuda:1"
    if isinstance(device, str) and device.startswith("cuda"):
        if ":" in device:
            return int(device.split(":", 1)[1])
        return 0

    return 0


def _as_fp32_contig(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    return x.contiguous()


class _OrtSession:
    """Small helper around onnxruntime with CUDA I/O binding."""

    def __init__(
        self,
        onnx_path: str,
        device_id: int = 0,
        prefer_tensorrt: bool = False,
        providers: Optional[Sequence] = None,
    ) -> None:
        _ensure_env_lib_in_ld_library_path()
        _preload_env_cudnn()
        import onnxruntime as ort

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.ort = ort
        self.onnx_path = onnx_path
        self.device_id = int(device_id)

        sess_options = ort.SessionOptions()
        # keep defaults; users can tune ORT separately if desired

        if providers is None:
            cuda = ("CUDAExecutionProvider", {"device_id": self.device_id})
            trt = ("TensorrtExecutionProvider", {"device_id": self.device_id})
            cpu = "CPUExecutionProvider"
            providers = [trt, cuda, cpu] if prefer_tensorrt else [cuda, cpu]

        self.sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=list(providers))

        # Cache input/output names to avoid repeated graph introspection
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def run(
        self,
        inputs: Dict[str, torch.Tensor],
        output_buffers: Dict[str, torch.Tensor],
        output_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run session with CUDA I/O binding.

        Note: ORT Python in this environment does not expose DLPack helpers, so
        we bind raw CUDA pointers from pre-allocated torch tensors.
        """
        import numpy as np

        if output_names is None:
            output_names = self.output_names

        io = self.sess.io_binding()

        # bind inputs (CUDA pointers)
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing ORT input '{name}' for {self.onnx_path}")
            t = inputs[name]
            if not t.is_cuda:
                raise ValueError(f"ORT CUDA backend expects CUDA tensor for '{name}', got device={t.device}")
            t = _as_fp32_contig(t)
            io.bind_input(
                name=name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=np.float32,
                shape=list(t.shape),
                buffer_ptr=t.data_ptr(),
            )

        # bind outputs to pre-allocated CUDA buffers
        for name in output_names:
            if name not in output_buffers:
                raise KeyError(f"Missing output buffer for '{name}'")
            out_t = output_buffers[name]
            if not out_t.is_cuda:
                raise ValueError(f"ORT CUDA backend expects CUDA output buffer for '{name}', got device={out_t.device}")
            out_t = _as_fp32_contig(out_t)
            io.bind_output(
                name=name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=np.float32,
                shape=list(out_t.shape),
                buffer_ptr=out_t.data_ptr(),
            )

        self.sess.run_with_iobinding(io)
        return {name: output_buffers[name] for name in output_names}


class ORTFNet(nn.Module):
    """ONNXRuntime wrapper for exported `fnet.onnx`.

    Expects input images in OpenCV BGR, range [0,255], shape [B, N, 3, H, W].
    """

    expects_raw_bgr_255 = True
    # backward-compatible flag (older notebook wording)
    expects_raw_rgb_255 = True

    def __init__(self, onnx_path: str, device: Union[str, torch.device] = "cuda:0", prefer_tensorrt: bool = False):
        super().__init__()
        device_id = _parse_cuda_device_id(device)
        self._sess = _OrtSession(onnx_path, device_id=device_id, prefer_tensorrt=prefer_tensorrt)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, n, _, h, w = images.shape
        out = torch.empty((b, n, 128, h // 8, w // 8), device=images.device, dtype=torch.float32)
        outs = self._sess.run({"images": images}, output_buffers={"fmaps": out}, output_names=["fmaps"])
        return outs["fmaps"]


class ORTCNet(nn.Module):
    """ONNXRuntime wrapper for exported `cnet.onnx`.

    Returns (net, inp) already split + activated.
    Expects input images in OpenCV BGR, range [0,255], shape [B, N, 3, H, W].
    """

    expects_raw_bgr_255 = True
    expects_raw_rgb_255 = True
    returns_split = True

    def __init__(self, onnx_path: str, device: Union[str, torch.device] = "cuda:0", prefer_tensorrt: bool = False):
        super().__init__()
        device_id = _parse_cuda_device_id(device)
        self._sess = _OrtSession(onnx_path, device_id=device_id, prefer_tensorrt=prefer_tensorrt)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, _, h, w = images.shape
        net = torch.empty((b, n, 128, h // 8, w // 8), device=images.device, dtype=torch.float32)
        inp = torch.empty((b, n, 128, h // 8, w // 8), device=images.device, dtype=torch.float32)
        outs = self._sess.run(
            {"images": images},
            output_buffers={"net": net, "inp": inp},
            output_names=["net", "inp"],
        )
        return outs["net"], outs["inp"]


class ORTUpdate(nn.Module):
    """ONNXRuntime wrapper for exported `update_core.onnx`.

    - Uses ONNX for (net_out, delta, weight)
    - Uses PyTorch `GraphAgg` for (eta, upmask) when `ii` is provided
    """

    def __init__(
        self,
        update_module: nn.Module,
        onnx_path: str,
        device: Union[str, torch.device] = "cuda:0",
        prefer_tensorrt: bool = False,
        output_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        device_id = _parse_cuda_device_id(device)
        self._sess = _OrtSession(onnx_path, device_id=device_id, prefer_tensorrt=prefer_tensorrt)

        # reuse PyTorch aggregation head (eta + upmask)
        self.agg = update_module.agg
        self.output_dtype = output_dtype

    @torch.no_grad()
    def forward(
        self,
        net: torch.Tensor,
        inp: torch.Tensor,
        corr: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        ii: Optional[torch.Tensor] = None,
        jj: Optional[torch.Tensor] = None,  # kept for API compatibility
    ):
        if flow is None:
            # flow is 4-channel motion features at 1/8 resolution
            b, n, _, ht, wd = net.shape
            flow = torch.zeros((b, n, 4, ht, wd), device=net.device, dtype=net.dtype)

        outs = self._sess.run(
            {"net": net, "inp": inp, "corr": corr, "flow": flow},
            output_buffers={
                "net_out": torch.empty_like(net, dtype=torch.float32),
                "delta": torch.empty((net.shape[0], net.shape[1], net.shape[3], net.shape[4], 2), device=net.device, dtype=torch.float32),
                "weight": torch.empty((net.shape[0], net.shape[1], net.shape[3], net.shape[4], 2), device=net.device, dtype=torch.float32),
            },
            output_names=["net_out", "delta", "weight"],
        )

        net_out = outs["net_out"].to(dtype=self.output_dtype)
        delta = outs["delta"].to(dtype=self.output_dtype)
        weight = outs["weight"].to(dtype=self.output_dtype)

        if ii is None:
            return net_out, delta, weight

        # GraphAgg expects ii for scatter aggregation
        eta, upmask = self.agg(net_out, ii.to(net_out.device))
        return net_out, delta, weight, eta, upmask


def enable_onnx_backend(
    net: nn.Module,
    fnet_onnx: str = "fnet.onnx",
    cnet_onnx: str = "cnet.onnx",
    update_onnx: str = "update_core.onnx",
    device: Union[str, torch.device] = "cuda:0",
    prefer_tensorrt: bool = False,
) -> nn.Module:
    """Replace DroidNet submodules with ONNXRuntime-backed versions.

    This is intended for inference (demo/eval). CorrBlock + BA remain PyTorch/CUDA.
    """

    net.fnet = ORTFNet(fnet_onnx, device=device, prefer_tensorrt=prefer_tensorrt)
    net.cnet = ORTCNet(cnet_onnx, device=device, prefer_tensorrt=prefer_tensorrt)
    net.update = ORTUpdate(net.update, update_onnx, device=device, prefer_tensorrt=prefer_tensorrt)
    return net

