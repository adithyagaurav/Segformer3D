import os

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.colors as co
import matplotlib.pyplot as plt

from src.Segformer_output import segformer_mit_b3
from src.utils import train_id_to_color as CMAP

def pipeline(model, tf, img_var):
    model.eval()
    img_var = tf(img_var).unsqueeze(0).float()
    with torch.no_grad():
        if torch.cuda.is_available():
            img_var = img_var.cuda()
        segm, depth, attn = model(img_var)
        segm, depth = segm.cpu(), depth.cpu().squeeze(0).permute(1,2,0).numpy()
        segm = CMAP[torch.argmax(segm, dim=1).squeeze(0)].astype(np.uint8)
        depth = np.abs(depth).reshape((512, 1024))
        return segm, depth, attn

def depth_to_rgb(depth):
    normalizer = co.Normalize(vmin=0, vmax=4)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im

def display_output(stage_output, frame, count):
    fig, axes = plt.subplots(4,4, figsize=(15.5,8))
    axes = axes.flatten()
    fig.tight_layout()
    for i in range(len(axes)):
        axes[i].clear()
        axes[i].imshow(frame)
        axes[i].imshow(stage_output[i], cmap='inferno', alpha=0.5)
        axes[i].axis('off')
    fig.savefig(f'out_attn/{count}.png')
    del fig

def render_video():
    image_names = sorted([os.listdir('out_attn/')])
    video_writer = cv2.VideoWriter("attention_out_.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20, (3*1024,512))

    for img_name in image_names:
        img = cv2.imread(os.path.join('out_attn', img_name))
        video_writer.write(img)
    video_writer.release()
        
def get_attention_output(attn, stage_scale):
    stage_output = []
    for i, data in enumerate(attn):
        stage_data = data['attn']
        stage_nh = stage_data.shape[1]
        stage_data = stage_data[0,:,:,0]
        stage_h, stage_w = int(512 / stage_scale[i]), int(1024 / stage_scale[i])
        stage_data = stage_data.reshape(stage_nh, stage_h, stage_w)
        stage_data = F.interpolate(stage_data.unsqueeze(0), size=(512, 1024), mode='bilinear')[0].detach().cpu().numpy()
        stage_output.append(stage_data)
    stage_output = np.concatenate(stage_output, axis=0)
    return stage_output

def test_video(video_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = segformer_mit_b3(in_channels=3, num_classes=19).to(device)
    model.load_state_dict(torch.load('segformer3d_mit_b3_cs_pretrain_19CLS_224_224_CE_loss.pt', map_location=torch.device('cpu')))
    preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])
    # video_writer = cv2.VideoWriter(f"{video_path[:-4]}_out_.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20, (3*1024,512))
    video_reader = cv2.VideoCapture(video_path)
    ret, frame = video_reader.read()
    count = 0
    while ret:
        ret, frame = video_reader.read()
        _, _, attn = pipeline(model, preprocess, frame)
        stage_output = get_attention_output(attn, [4, 8, 16, 32])
        display_output(stage_output, frame, count)
        count+=1
    render_video()

    # video_writer.release()

test_video('stuttgart.mp4')
