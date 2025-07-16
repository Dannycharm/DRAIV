import torch
import numpy as np
import argparse
import cv2, time, os, shutil
import sys
from pathlib import Path
from tqdm import tqdm

# TwinLiteNetPlus
parent_dir = (Path(__file__).parent.parent / 'TwinLiteNetPlus_scripts').resolve()
sys.path.append(str(parent_dir))

from model.model import TwinLiteNetPlus

def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))


    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def show_seg_result(img, result, save_dir=None, is_ll=False,palette=None):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    

    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    
    # for label, color in enumerate(palette):
    #     color_area[result[0] == label, :] = color

    color_area[result[0] == 1] = [0, 255, 0]
    color_area[result[1] ==1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    return img

@torch.no_grad()
def segment_image(input_image, args):
    """Run TwinLiteNet+ once on one RGB image file."""
    device = torch.device('cuda:0')
    half   = True

    # ------------------------------------------------------------------ model
    model = TwinLiteNetPlus(args).to(device)
    if half:
        model.half()
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()

    # ----------------------------------------------------------- read + pre-process
    assert os.path.isfile(input_image), f"❌ {input_image} not found"
    img_bgr = cv2.imread(input_image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    h0, w0  = img_bgr.shape[:2]

    # letter-box resize
    img, ratio, pad = letterbox_for_img(img_bgr, new_shape=(384, 640), auto=True)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()                    # BGR→RGB, HWC→CHW
    img = torch.from_numpy(img).to(device).unsqueeze(0)         # add batch dim
    img = img.half() / 255.0 if half else img.float() / 255.0

    # ---------------------------------------------------------------- inference
    t0 = time.time()
    da_seg_out, ll_seg_out = model(img)
    print(f"Inference: {(time.time()-t0)*1000:.1f} ms")

    # ------------------------------------------------------------- post-process
    _, _, H, W = img.shape
    pad_w, pad_h = map(int, pad)          # pad = (pad_w, pad_h)
    da_pred = da_seg_out[:, :, pad_h:H-pad_h, pad_w:W-pad_w]
    ll_pred = ll_seg_out[:, :, pad_h:H-pad_h, pad_w:W-pad_w]

    da_mask = torch.nn.functional.interpolate(da_pred, scale_factor=1/ratio[0], mode='bilinear')
    ll_mask = torch.nn.functional.interpolate(ll_pred, scale_factor=1/ratio[0], mode='bilinear')

    da_mask = torch.argmax(da_mask, 1).int().squeeze().cpu().numpy()
    ll_mask = torch.argmax(ll_mask, 1).int().squeeze().cpu().numpy()

    return da_mask, ll_mask, img_bgr

if __name__ == "__main__":

    # --------------------- Model & Tracker Setup -----------------------
    # model = YOLO(args.weights).to(device)
    # tracker  = sv.ByteTrack()
    # box_anno = sv.BoxAnnotator()
    # lbl_anno = sv.LabelAnnotator()

    # --------------------- Model & segmentation Setup -----------------------
    parser = argparse.ArgumentParser()
    #parser.add_argument("--image", type=str, help="path/to/img.jpg")
    parser.add_argument("--weight", type=str, default="/workspace/models/TwinLiteNet_models/large.pth", help="model .pt")
   #parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--save_dir", type=str, default="/workspace/runs/segment_result")
    # any extra hyper-parameters the model constructor needs:
    parser.add_argument( "--hyp", type=str,default="/workspace/scripts/TwinLiteNetPlus_scripts/hyperparameters/twinlitev2_hyper.yaml", help="Path to hyperparameters YAML")
    parser.add_argument("--config", default="large", help="Model configuration")

    args = parser.parse_args()
    #input_image = "/datasets/bdd100k/images/100k/train/0000f77c-62c2a288.jpg" 
    input_image = "/datasets/bdd100k/images/100k/train/0001542f-7c670be8.jpg"
    #prediction:
    da_mask, ll_mask, img_bgr = segment_image (input_image, args)    
    print ("da_mask, ll_mask shapes:", da_mask.shape, ll_mask.shape)    
    # Visualise
    vis = show_seg_result(img_bgr, (da_mask, ll_mask))  

    # ------------------------------------------------------------------ save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = Path(args.save_dir) / Path(input_image).name
    cv2.imwrite(str(save_path), vis)
    print(f"✅ Saved to {save_path}")


    
