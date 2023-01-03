import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import numpy as np
from dataset import SegformerDataset
from SegFormer import segformer_mit_b3 as Segformer3D
from utils import meanIoU, plot_training_results, InvHuberLoss
import argparse

def validate_model(model, dataloader, criterions, metric_class, num_classes, device):
    model.eval()
    total_loss = 0.0
    metric = metric_class(num_classes)
    criterion_seg, criterion_depth = criterions

    with torch.no_grad():
        for inputs, seg_label, depth_label in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            seg_label = seg_label.to(device)
            depth_label = depth_label.to(device)

            seg_pred, depth_pred = model(inputs)
            seg_loss = criterion_seg(seg_pred, seg_label)
            depth_loss = criterion_depth(depth_pred, depth_label)
            loss = 1.0 * seg_loss + 1.0 * depth_loss
            total_loss += loss.item()
            metric.update(seg_pred.detach().cpu(), seg_label.detach().cpu())
    eval_loss = total_loss/len(dataloader)
    eval_score = metric.compute()
    return eval_loss, eval_score

def train_validate_model(model, num_epochs, model_name, criterions, optimizer, 
                         device, dataloader_train, dataloader_valid, 
                         metric_class, metric_name, num_classes, lr_scheduler = None,
                         output_path = '.'):
    results = []
    min_val_loss = np.Inf
    model.to(device)
    criterion_seg, criterion_depth = criterions
    for epoch in range(num_epochs):
        print(f"Running Epoch : {epoch+1}")
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_depth_loss = 0.0
        train_size = len(dataloader_train)
        for i, item in enumerate(tqdm(dataloader_train, total=train_size)):
            inputs, seg_label, depth_label = item
            inputs = inputs.to(device)
            seg_label = seg_label.to(device)
            depth_label = depth_label.to(device)

            seg_pred, depth_pred = model(inputs)
            seg_loss = criterion_seg(seg_pred, seg_label)
            depth_loss = criterion_depth(depth_pred, depth_label)
            loss = 1.0 * seg_loss + 1.0 * depth_loss
            train_loss += loss.item()
            train_seg_loss += seg_loss.item()
            train_depth_loss += depth_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step(i)
        train_loss = train_loss/len(dataloader_train)
        train_seg_loss = train_seg_loss/len(dataloader_train)
        train_depth_loss = train_depth_loss/len(dataloader_train)
        val_loss, val_score = validate_model(model, dataloader_valid, criterions, metric_class, num_classes, device)
        model.train()
        print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}, segLoss: {train_seg_loss:6.5f}, depthLoss:{train_depth_loss:6.5f}, validationLoss:{val_loss:6.5f}, {metric_name}:{val_score: 4.2f}')
        results.append({'epoch': epoch, 
                        'trainLoss': train_loss, 
                        'validationLoss': val_loss, 
                        f'{metric_name}': val_score})
        

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"results/segformer3d.pt")

    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    targetWidth = 256
    targetHeight = 128

    # MODEL HYPERPARAMETERS
    N_EPOCHS = 200
    NUM_CLASSES = 19
    MAX_LR = 1e-3
    MODEL_NAME = f'segformer_mit_b3_cs_pretrain_19CLS_{targetHeight}_{targetWidth}_CE_loss'

    criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
    criterion_depth = torch.nn.MSELoss().to(device)#InvHuberLoss(ignore_index=0).to(device)
    criterions = [criterion_seg, criterion_depth]
    print(f'[INFO]: Training job for {MODEL_NAME} received')
    model = Segformer3D(in_channels=3, num_classes=NUM_CLASSES).to(device)
    print(f'[INFO]: New Segformer model created')
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    print(f'[INFO]: Segformer model weights loaded')
    train_dataset = SegformerDataset(root_dir=args.data_dir, mode='train')
    val_dataset = SegformerDataset(root_dir=args.data_dir, mode='val')

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    valloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    print(f'[INFO]:Dataset created')
    lr_encoder = 1e-2
    lr_decoder = 1e-2
    momentum_encoder = 0.9
    momentum_decoder = 0.9
    weight_decay_encoder = 1e-5
    weight_decay_decoder = 1e-5
    n_epochs = 1000
    for param in model.backbone.parameters():
      param.requires_grad = False
    for param in model.decoder_head.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.depth_head.parameters(), lr=lr_decoder)
    scheduler = OneCycleLR(optimizer, max_lr= lr_decoder, epochs = N_EPOCHS,steps_per_epoch = len(trainloader), 
                        pct_start=0.3, div_factor=10, anneal_strategy='cos', verbose=True)
    print(f'[INFO]: Start training')
    results = train_validate_model(model, N_EPOCHS, MODEL_NAME, criterions, optimizer, device, trainloader, valloader, meanIoU, 'meanIoU', NUM_CLASSES, scheduler)

    plot_training_results(results, MODEL_NAME)