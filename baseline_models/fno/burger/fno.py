# FNO training script for Burgers Equation
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import os
from glob import glob
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from neuralop.models.base_model import get_model
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from RecursiveFNO import RecursiveFNO


# Hyperparameter config files
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig("burgers_pino_config.yaml", config_name="default", config_folder="config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name
# Hyperparameter
BATCH_SIZE = config["data"]["batch_size"]
time_steps = config["data"]["temporal_length"]  # Total time steps per simulation
dt = 2.0 / time_steps                           # Temporal resolution
dx = 6.0 / config["data"]["spatial_length"]     # Spatial resolution
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


def read_data(datapath):
    vel_seq_whole = []
    for each_file in glob(datapath):
        vis = np.float32(os.path.basename(each_file).split("_")[-3]) / 10000.0
        sim_data = np.float32(np.load(each_file))
        vis_data_shape = (sim_data.shape[0], sim_data.shape[1], sim_data.shape[2], 1)
        vis_data = np.empty(vis_data_shape)
        vis_data[:, :, :, :] = vis
        run_data = np.concatenate([sim_data, vis_data], axis=-1).transpose(2, 3, 1, 0)
        run_data = np.expand_dims(run_data, axis=0)
        vel_seq_whole.append(run_data)
    vel_seq_whole = np.concatenate(vel_seq_whole, axis=0)
    return vel_seq_whole


train_seq = read_data("data/train_data/*.npy")
test_seq = read_data("data/test_data/*.npy")
# Training set
train_ic = torch.Tensor(train_seq[:, 0, :, :, :])
train_seq = torch.Tensor(train_seq[:, 1:, :2, :, :])
train_dataset = TensorDataset(train_ic, train_seq)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Testing set
test_ic = torch.Tensor(test_seq[:, 0, :, :, :])
test_seq = torch.Tensor(test_seq[:, 1:, :2, :, :])
test_dataset = TensorDataset(test_ic, test_seq)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Model
model_fno = get_model(config).cuda()
model = RecursiveFNO(model_fno, time_steps+1).cuda()
# Optimizer and scheduler
loss_func = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5) 
scheduler = StepLR(optimizer, step_size=config["opt"]["step_size"], 
                   gamma=config["opt"]["gamma"])
# Training
best_loss = 1e12
train_loss_list = []
test_loss_list = []
for epoch in range(MAX_EPOCH):
    # Training
    train_loss = 0.0
    model.train()
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        ic, seq = data
        ic, seq = ic.cuda(), seq.cuda()
        optimizer.zero_grad()
        output = model(ic)
        # Loss
        loss = loss_func(output[:, :-1, :2, :, :], seq)
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        optimizer.step()
    scheduler.step()
    train_loss /= len(train_dataloader)
    train_loss_list.append(train_loss)
    # Testing
    test_loss = 0.0
    with torch.no_grad():
        model.eval()
        for data in tqdm(test_dataloader, total=len(test_dataloader)):
            ic, seq = data
            ic, seq = ic.cuda(), seq.cuda()
            output = model(ic)
            loss = loss_func(output[:, :-1, :2, :, :], seq)
            test_loss += loss.item()
        test_loss /= len(test_dataloader)
        test_loss_list.append(test_loss)
    # Report stuff
    print('[%d/%d %d%%] loss: %.10f %.10f' % ((epoch+1), MAX_EPOCH, ((epoch+1)/MAX_EPOCH*100.0), 
                                              train_loss, test_loss))
    if test_loss < best_loss:
        save_checkpoint(model, optimizer, scheduler, model_save_path)
        best_loss = test_loss