import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import os
from glob import glob
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from EnergeticMatDataPipeLine import EnergeticMatDataPipeLine
from neuralop.models.base_model import get_model
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from RecursiveFNO import RecursiveFNO


# Hyperparameter config files
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig("em_pino_config.yaml", config_name="default", config_folder="config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name
# Hyperparameter
SEQ_LEN = 15
BATCH_SIZE = 6
MAX_EPOCH = config["opt"]["n_epochs"]           # Max number of epochs to train
LR = config["opt"]["learning_rate"]             # Learning rate
model_save_path = './model/checkpoint%i.pt' % (MAX_EPOCH)


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print('Pretrained model loaded!')
    return model, optimizer, scheduler


# Train
train_ic, train_seq = torch.load("single_void_data/training_set.pt")
train_dataset = TensorDataset(train_ic, train_seq)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Test
test_ic, test_seq = torch.load("single_void_data/testing_set.pt")
test_dataset = TensorDataset(test_ic, test_seq)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Model
model_fno = get_model(config).cuda()
model = RecursiveFNO(model_fno, SEQ_LEN-1).cuda()
# Optimizer and scheduler
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0) 
scheduler = StepLR(optimizer, step_size=config["opt"]["step_size"], 
                   gamma=config["opt"]["gamma"])
# Training
best_loss = 1e12
for epoch in range(MAX_EPOCH):
    # Training
    train_loss = 0.0
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        model.train()
        ic, seq = data
        ic, seq = ic.cuda(), seq.cuda()
        optimizer.zero_grad()
        output = model(ic)
        loss = loss_func(output, seq)
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        optimizer.step()
    scheduler.step()
    train_loss /= len(train_dataloader)
    # Testing
    test_loss = 0.0
    with torch.no_grad():
        model.eval()
        for data in tqdm(test_dataloader, total=len(test_dataloader)):
            ic, seq = data
            ic, seq = ic.cuda(), seq.cuda()
            output = model(ic)
            loss = loss_func(output, seq)
            test_loss += loss.item()
        test_loss /= len(test_dataloader)
    # Report stuff
    print('[%d/%d %d%%] train: %.10f test: %.10f' % ((epoch+1), MAX_EPOCH, ((epoch+1)/MAX_EPOCH*100.0), train_loss, test_loss))
    if test_loss < best_loss:
        save_checkpoint(model, optimizer, scheduler, model_save_path)
        best_loss = test_loss