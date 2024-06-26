import copy
import torch
import time
import os

import models

class Compile:
    def __init__(self, cfg, device, half_precision):
        self.cfg = cfg
        self.device = device
        self.half_precision = half_precision

        if device == "cpu" and half_precision:
            raise ValueError(f"Half precision does not work with device {device}")

        # load model
        checkpoint_dir = vars(cfg.model)
        checkpoint_dir = [f"{key}[{checkpoint_dir[key]}]" for key in checkpoint_dir]
        checkpoint_dir = "-".join(checkpoint_dir)

        self.checkpoint_dir = os.path.join("logs", checkpoint_dir)
        checkpoint_path = os.path.join(self.checkpoint_dir, "ckpt_finetuned.pth")

        model = models.load(cfg)
        model.load_checkpoint(checkpoint_path, device=device, load_opt=False)
        model.to(device).to(torch.float32)
        model.eval()
        self.model = model.half() if self.half_precision else model

    def run(self):
        dummy = torch.rand(1, 3, self.cfg.data.input_size, self.cfg.data.input_size).to(self.device)
        dummy = dummy.half() if self.half_precision else dummy.to(torch.float32)

        self.module = torch.jit.trace_module(self.model, {'forward_all': dummy})

    def save(self):
        torch.jit.save(self.module, os.path.join(self.checkpoint_dir, './packaged.pth'))

    def load(self):
        self.loaded_module = torch.jit.load(
            f=os.path.join(self.checkpoint_dir, './packaged.pth'),
            map_location=self.device
        )
        self.loaded_module.eval()

    def test(self):
        dummy = torch.rand(1, 3, self.cfg.data.input_size, self.cfg.data.input_size).to(self.device)
        dummy = dummy.half() if self.half_precision else dummy

        t0 = time.time()
        with torch.no_grad():
            out_jl = self.loaded_module.forward_all(copy.deepcopy(dummy)).detach()
        t1 = time.time()
        with torch.no_grad():
            out_nj = self.model.forward_all(dummy).detach()
        t2 = time.time()
        print(
            f"Out size: {out_nj.size()}\n"
            "Out diff:", (out_nj - out_jl).abs().mean().item(), "\n",
            "\n",
            f"  Torch JIT: {t1 - t0}\n",
            f"Torch Model: {t2 - t1}\n",
            "--------------------------"
        )

if __name__ == "__main__":
    from utils import load_config
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference on query images."
    )
    parser.add_argument(
        '--checkpoint_dir',
        default="arch[UnetPlusPlus]-backbone[resnet101]-pretrained_weights[imagenet]-out_layer_size[512]-in_channels[3]-version[coco_proxy_opt]",
        type=str,
        help='Directory containing the checkpoint.'
    )
    parser.add_argument(
        '--device',
        default="cuda",
        type=str,
        help='"cpu" or "cuda"'
    )
    parser.add_argument(
        '--half_precision',
        action='store_false',
        help='for half precision, do not use yet! it is not ready!'
    )

    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    device = args.device
    half_precision = args.half_precision

    cfg = load_config(os.path.join(
        "logs",
        checkpoint_dir,
        "./config.yml"
    ))

    instance = Compile(cfg, device, half_precision=half_precision)
    instance.run()
    instance.save()
    instance.load()
    instance.test()
    instance.test()
