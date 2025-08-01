''' 

Run with: 

DATA_DIR=/scratch/dannycharm-alt-REU/DRAIV/datasets
IMG=/scratch/dannycharm-alt-REU/DRAIV/draiv_ml.sif
REPO=/scratch/dannycharm-alt-REU/DRAIV      


    python /workspace/scripts/TwinLiteNetPlus_scripts/train.py --config medium --max_epochs 3 --ema 

'''



import math
import os
from argparse import ArgumentParser
from copy import deepcopy

import BDD100K
import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import yaml
from loss import TotalLoss
from model.model import TwinLiteNetPlus
from utils import netParams, poly_lr_scheduler, save_checkpoint, train, val


class ModelEMA:
    """Exponential Moving Average (EMA) for model parameters"""

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / 2000)
        )  # Exponential decay ramp
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA model parameters"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


def train_net(args, hyp):
    """Train the neural network model with given arguments and hyperparameters"""
    use_ema = args.ema
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()

    model = TwinLiteNetPlus(args)
    # if num_gpus > 1:
    #     model = torch.nn.DataParallel(model)

    os.makedirs(args.savedir, exist_ok=True)  # Ensure save directory exists

    trainLoader = torch.utils.data.DataLoader(
        BDD100K.Dataset(hyp, valid=False),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # I added this -----------------------
    #name_temp, img_temp, mask_temp = next(iter(trainLoader))
    #print("input image size:  ", img_temp[0].size() )
    #print("mask image size:  ", mask_temp[0].size() )
    #------------------------------


    valLoader = torch.utils.data.DataLoader(
        BDD100K.Dataset(hyp, valid=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    print(f"Total network parameters: {netParams(model)}")

    criteria = TotalLoss(hyp)
    start_epoch = 0
    lr = hyp["lr"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(hyp["momentum"], 0.999),
        eps=hyp["eps"],
        weight_decay=hyp["weight_decay"],
    )

    ema = ModelEMA(model) if use_ema else None

    # Resume training from checkpoint
    if args.resume and os.path.isfile(args.resume):
        if args.resume.endswith(".tar"):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            if use_ema:
                ema.ema.load_state_dict(checkpoint["ema_state_dict"])
                ema.updates = checkpoint["updates"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No valid checkpoint found at '{args.resume}'")

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, args.max_epochs):
        model_file_name = os.path.join(args.savedir, f"model_{epoch}.pth")
        poly_lr_scheduler(args, hyp, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        print(f"Learning rate: {lr}")

        model.train()
        ema = train(
            args,
            trainLoader,
            model,
            criteria,
            optimizer,
            epoch,
            scaler,
            args.verbose,
            ema if use_ema else None,
        )

        model.eval()
        da_segment_results, ll_segment_results = val(
            valLoader, ema.ema if use_ema else model, args=args
        )

        print(f"Driving Area Segment: mIOU({da_segment_results[2]:.3f})")
        print(
            f"Lane Line Segment: Acc({ll_segment_results[0]:.3f}) IOU({ll_segment_results[1]:.3f})"
        )

        (
            torch.save(ema.ema.state_dict(), model_file_name)
            if use_ema
            else torch.save(model.state_dict(), model_file_name)
        )

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema.ema.state_dict() if use_ema else None,
                "updates": ema.updates if use_ema else None,
                "optimizer": optimizer.state_dict(),
                "lr": lr,
            },
            os.path.join(args.savedir, "checkpoint.pth.tar"),
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Max number of epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Number of parallel threads"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--savedir", default="/workspace/runs/TwinLiteNetPlus_lane/test_dir", help="Directory to save the results"
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default="/workspace/scripts/TwinLiteNetPlus_scripts/hyperparameters/twinlitev2_hyper.yaml",
        help="Path to hyperparameters YAML",
    )
    parser.add_argument(
        "--resume", type=str, default="", help="Resume training from a checkpoint"
    )
    parser.add_argument("--config", default="nano", help="Model configuration")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--ema", action="store_true", help="Use Exponential Moving Average (EMA)"
    )
    args = parser.parse_args()

    with open(args.hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # Load hyperparameters

    train_net(args, hyp.copy())
