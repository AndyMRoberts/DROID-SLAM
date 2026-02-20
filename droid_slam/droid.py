import gc
import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.args = args
        self.load_weights(args.weights)
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn
            self.visualizer = Process(target=visualization_fn, args=(self.video, None))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights, weights_only=True).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

        # Optional ONNXRuntime backend (GPU) for fnet/cnet/update
        if getattr(self.args, "use_onnx", False):
            from onnx_backend import enable_onnx_backend

            fnet_onnx = getattr(self.args, "onnx_fnet", "fnet.onnx")
            cnet_onnx = getattr(self.args, "onnx_cnet", "cnet.onnx")
            update_onnx = getattr(self.args, "onnx_update", "update_core.onnx")
            prefer_trt = bool(getattr(self.args, "onnx_tensorrt", False))

            enable_onnx_backend(
                self.net,
                fnet_onnx=fnet_onnx,
                cnet_onnx=cnet_onnx,
                update_onnx=update_onnx,
                device="cuda:0",
                prefer_tensorrt=prefer_trt,
            )

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend
        del self.filterx
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        # Free backend graph/correlation before trajectory filler (helps ONNX memory)
        del self.backend
        torch.cuda.empty_cache()

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

