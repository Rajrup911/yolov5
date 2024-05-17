import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import colors
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import cv2, increment_path, non_max_suppression
from utils.torch_utils import smart_inference_mode

def draw_rectangular_region(img):
    pts = np.array([[(100, 420), (1220, 420), (1220, 340), (100, 340)]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return img #creates region of interest

def check_object_passes_region(x, y):
    region_points = [(100, 420), (1320, 420), (1320, 340), (100, 340)]
    crossings = 0
    x, y = x/2, y/2
    for i in range(len(region_points)):
        x1, y1 = region_points[i]
        x2, y2 = region_points[(i + 1) % len(region_points)]
        if (y1 <= y < y2) or (y2 <= y < y1):
            if x1 + (y - y1) / (y2 - y1) * (x2 - x1) < x:
                crossings += 1
    return crossings % 2 == 1 #checks if centroid of object passes through roi
    
def bboxes(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]) 
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  
    boxes[..., [0, 2]] -= pad[0] 
    boxes[..., [1, 3]] -= pad[1] 
    boxes[..., :4] /= gain
    
    boxes[..., 0].clamp_(0, img0_shape[1]) 
    boxes[..., 1].clamp_(0, img0_shape[0]) 
    boxes[..., 2].clamp_(0, img0_shape[1]) 
    boxes[..., 3].clamp_(0, img0_shape[0]) 
    return boxes
    
def box_label(im0, box, color=(128, 128, 128)):
    if isinstance(box, torch.Tensor):
        box = box.tolist()
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im0, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)


@smart_inference_mode()
def detection(weights=ROOT / "yolov5x.pt", source=ROOT / "data/images", imgsz=(640, 640), project=ROOT / "runs/detect"):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    name="exp"

    save_dir = increment_path(Path(project) / name, False)
    (save_dir).mkdir(parents=True)

    model = DetectMultiBackend(weights = 'yolov5x.pt')
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = [640, 640]

    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1) #utility function
    vid_path, vid_writer = [None] * bs, [None] * bs

    seen, windows = 0, []
    
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).cuda()
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if model.xml and im.shape[0] > 1:
            ims = torch.chunk(im, im.shape[0], 0)
                
        if model.xml and im.shape[0] > 1:
            pred = None
            for image in ims:
                if pred is None:
                    pred = model(image).unsqueeze(0)
                else:
                    pred = torch.cat((pred, model(image).unsqueeze(0)), dim=0)
            pred = [pred, None]
        else:
            pred = model(im)
        
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000) #NMS-utility function

        for i, det in enumerate(pred):
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization
            im0 = draw_rectangular_region(im0)

            if len(det):
                det[:, :4] = bboxes(im.shape[2:], det[:, :4], im0.shape).round()
                object_count = 0
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    box_label(im0, box = xyxy, color=colors(c, True))
                        
                    x1 = int(xyxy[0].item())
                    y1 = int(xyxy[1].item())
                    x2 = int(xyxy[2].item())
                    y2 = int(xyxy[3].item())
                        
                    if c==2 and check_object_passes_region(x1+x2, y1+y2):
                        object_count += 1
                        
            cv2.putText(im0, f'Object Count: {object_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)

            im0 = np.asarray(im0)

            if dataset.mode == "image":
                cv2.imwrite(save_path, im0)
            else:
                if vid_path[i] != save_path: 
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else: 
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix(".mp4"))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                vid_writer[i].write(im0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5x.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

def main(opt):
    detection(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)