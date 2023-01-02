import random
import os

import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

from SegFormer import segformer_mit_b3
from utils import train_id_to_color as CMAP

def pipeline(model, img_var):
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img_var = img_var.cuda()
        segm, depth = model(img_var)
        segm, depth = segm.cpu(), depth.cpu().squeeze(0).permute(1,2,0).numpy()
        segm = CMAP[torch.argmax(segm, dim=1).squeeze(0)].astype(np.uint8)
        depth = np.abs(depth).reshape((512, 1024))
        return segm, depth 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image_dir", required=True)
    args = parser.parse_args()
    test_imgs_dir = args.image_dir
    test_img_name = random.choice(os.listdir(test_imgs_dir))
    test_img_path = os.path.join(test_imgs_dir, test_img_name)
    print(f"[INFO]: Running inference on {test_img_path}")
    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    img = cv2.imread(test_img_path)#np.load(test_img_path)
    img_input = preprocess(img).unsqueeze(dim=0).float()
    print("[INFO]: Image loaded")

    model = segformer_mit_b3(in_channels=3, num_classes=19).to(device)
    model.load_state_dict(torch.load('weights/segformer3d_mit_b3_cs_pretrain_19CLS_224_224_CE_loss.pt', map_location=torch.device('cpu')))
    print("[INFO]: Model loaded, running inference")
    out_segm, out_depth = pipeline(model, img_input)
    print("[INFO]: Plot results")
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,10))
    ax1.imshow(img)
    ax1.set_title('Original', fontsize=20)
    ax2.imshow(out_segm)
    ax2.set_title('Predicted Segmentation', fontsize=20)
    ax3.imshow(out_depth, cmap="plasma")
    ax3.set_title("Predicted Depth", fontsize=20)
    plt.show()
