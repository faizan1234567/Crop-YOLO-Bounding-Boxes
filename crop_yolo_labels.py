import os 
import argparse
import numpy as np
import cv2
import torch

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default= "", type = str, help= "path to images directory to crop images from..")
    parser.add_argument("--save", default="", type = str, help="path to the save cropped images directory")
    parser.add_argument("--labels", default="", type = str, help="path to the save label files directory")
    opt = parser.parse_args()
    return opt

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """convert normalized boxes to xyxy format
    credit: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    
    Args:
        x: np.ndarray or torch.tensor -> an annotation tensor in xyhw format
        w: int, image width
        h: int, image height
        padw: int, padding if applied to width default no padding
        padh: int, padding if applied to height default no padding
    
    Return:
        y: np.ndarray or torch.tensor -> transofomed to unnormalized xyxy"""

    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def save_img(img, name, save_path):
    """save the cropped images to the given directory
    
    Args:
        img: np.ndarray
        save_path: path to save the cropped logo image"""
    save_file = os.path.join(save_path, name)
    cv2.imwrite(save_file, img)

def crop_logo(images_path, save, labels):
    """crop an logo image and save it to the ssave directory
    
    Args:
        images: str, path to images dir
        save: str, path to save croped images dir
        labels:, str, path to the annotation dir.
        """
    images = os.listdir(images_path)
    annotations = os.listdir(labels)
    for annot in annotations:
        image_name = annot.split('.')[-2]
        image_name = image_name + ".jpg".strip()
        image_path = os.path.join(images_path, image_name)
        annot_path = os.path.join(labels, annot)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            with open(annot_path) as f:
                annotation = []
                line = f.readline() 
                annot = [float(num) for num in line.split(" ")]
                annotation.append(annot[1:])
            annotation = np.array(annotation)
            boxes = xywhn2xyxy(annotation, w=width, h=height)
            for box in boxes:
                x_top, y_top, x_bottom, y_bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cropped_img = img[y_top: y_bottom, x_top: x_bottom]
                save_img(cropped_img, image_name, save)
        else:
            print(f"{image_path} doesn't exists!!")

def main():
    args = read_args()
    crop_logo(args.images, args.save, args.labels)
    print("done!!!")

if __name__ == "__main__":
    main()

