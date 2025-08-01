import os
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import yaml
from demoDataset import LoadImages, LoadStreams
from loss import TotalLoss
from model.model import TwinLiteNetPlus
from tqdm import tqdm
from utils import netParams, val


def show_seg_result(
    img, result, index, epoch, save_dir=None, is_ll=False, palette=None
):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3  # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

    # for label, color in enumerate(palette):
    #     color_area[result[0] == label, :] = color

    color_area[result[0] == 1] = [0, 255, 0]
    color_area[result[1] == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)

    return img


def detect(args):

    device = "cuda:0"
    half = True
    model = TwinLiteNetPlus(args)
    model = model.cuda()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if args.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(args.source, img_size=args.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(args.source, img_size=args.img_size)
        bs = 1  # batch_size
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, args.img_size, args.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device != "cpu" else None  # run once

    model.load_state_dict(torch.load(args.weight))
    model.eval()

    for i, (path, img, img_det, vid_cap, shapes) in tqdm(
        enumerate(dataset), total=len(dataset)
    ):
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.cuda().half() / 255.0 if half else img.cuda().float() / 255.0

        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]
        # Inference
        # t1 = time_synchronized()
        da_seg_out, ll_seg_out = model(img)
        # t2 = time_synchronized()

        save_path = (
            str(opt.save_dir + "/" + Path(path).name)
            if dataset.mode != "stream"
            else str(opt.save_dir + "/" + "web.mp4")
        )

        da_predict = da_seg_out[:, :, pad_h : (height - pad_h), pad_w : (width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(
            da_predict, scale_factor=int(1 / ratio), mode="bilinear"
        )
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        ll_predict = ll_seg_out[:, :, pad_h : (height - pad_h), pad_w : (width - pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(
            ll_predict, scale_factor=int(1 / ratio), mode="bilinear"
        )
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing

        img_vis = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _)

        if dataset.mode == "images":
            cv2.imwrite(save_path, img_vis)

        elif dataset.mode == "video":
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = "mp4v"  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h, w, _ = img_det.shape
                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                )
            vid_writer.write(img_det)

        else:
            cv2.imshow("image", img_det)
            cv2.waitKey(1)  # 1 millisecond


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--weight", type=str, default="pretrained/large.pth", help="model.pth path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="inference/videos", help="source"
    )  # file/folder   ex:inference/images
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["nano", "small", "medium", "large"],
        help="Model configuration",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="inference/output",
        help="directory to save results",
    )
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)
