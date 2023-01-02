import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import matplotlib.cm as cm
import matplotlib.colors as co
import argparse

from SegFormer import segformer_mit_b3
from utils import train_id_to_color as CMAP

def pipeline(model, tf, img_var):
    model.eval()
    img_var = tf(img_var).unsqueeze(0).float()
    with torch.no_grad():
        if torch.cuda.is_available():
            img_var = img_var.cuda()
        segm, depth = model(img_var)
        segm, depth = segm.cpu(), depth.cpu().squeeze(0).permute(1,2,0).numpy()
        segm = CMAP[torch.argmax(segm, dim=1).squeeze(0)].astype(np.uint8)
        depth = np.abs(depth).reshape((512, 1024))
        return segm, depth

def depth_to_rgb(depth):
    normalizer = co.Normalize(vmin=0, vmax=4)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


def test_video(video_path, weights_path):
    print(f"[INFO]: Running inference on video {video_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = segformer_mit_b3(in_channels=3, num_classes=19).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    print("[INFO]: Model loaded, running inference")
    preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])
    video_writer = cv2.VideoWriter(f"results/{video_path[:-4]}_out.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20, (3*1024,512))
    video_reader = cv2.VideoCapture(video_path)
    ret, frame = video_reader.read()
    count=0
    while ret:
        count+=1
        ret, frame = video_reader.read()
        output_segm, output_depth = pipeline(model, preprocess, frame)
        output_depth = depth_to_rgb(output_depth)
        output_depth = cv2.cvtColor(output_depth, cv2.COLOR_RGB2BGR)
        output_frame = np.hstack((frame, output_segm, output_depth))
        video_writer.write(output_frame)
        print(f"[INFO]:Processing frame {count}", "\r")
    video_writer.release()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    test_video(args.video, args.weights)
