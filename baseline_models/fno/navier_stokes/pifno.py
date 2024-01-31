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
from LossGeneratorNew import loss_generator, compute_loss
from RecursiveFNO import RecursiveFNO


# Hyperparameter config files
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig("ns_pino_config.yaml", config_name="default", config_folder="config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name
# Hyperparameter
BATCH_SIZE = 1
time_steps = 38  # Total time steps per simulation
dt = 2.0 / time_steps                           # Temporal resolution
dx = 1.0 / 128     # Spatial resolution
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


def read_data(datapath, re_list):
    vel_seq_whole = []
    for vis in re_list:
        tmp = np.load(datapath + "Re_%i.npy" % vis)
        sim_data = tmp.reshape(128, 256, 39, 3)
        rho_data_shape = (sim_data.shape[0], sim_data.shape[1], sim_data.shape[2], 1)
        rho_data = np.empty(rho_data_shape)
        rho_data[:, :, :, :] = vis * 4.0
        run_data = np.concatenate([sim_data, rho_data], axis=-1)
        # Normalization
        run_data[:, :, :, 2] /= (vis * 4.0)
        run_data[:, :, :, 3] /= 40000
        run_data = np.transpose(run_data, (2, 3, 1, 0))
        vel_seq_whole.append(run_data)
    return np.array(vel_seq_whole)


train_list = [40, 80, 100, 150, 200, 250, 300, 400, 450, 500, 600, 650, 
              700, 800, 850, 900]
test_list = [15, 20, 30, 60, 120, 140, 350, 550, 750, 950, 1000]
train_seq_clipped = read_data("/scratch/xc7ts/pinn/navier_stokes/data/", train_list)
test_seq_clipped = read_data("/scratch/xc7ts/pinn/navier_stokes/data/", test_list)
# Train
train_ic = torch.Tensor(train_seq_clipped[:, 0, :, :, :])    # u, v, p, rho
train_seq = torch.Tensor(train_seq_clipped[:, 1:, :3, :, :])  # u, v, p
train_dataset = TensorDataset(train_ic, train_seq)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Test
test_ic = torch.Tensor(test_seq_clipped[:, 0, :, :, :])
test_seq = torch.Tensor(test_seq_clipped[:, 1:, :3, :, :])
test_dataset = TensorDataset(test_ic, test_seq)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Model
model_fno = get_model(config).cuda()
model = RecursiveFNO(model_fno, time_steps+1).cuda()
# Optimizer and scheduler
loss_func = loss_generator(dt, dx)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0) 
scheduler = StepLR(optimizer, step_size=config["opt"]["step_size"], 
                   gamma=config["opt"]["gamma"])
# Mask
rr = 0.125 + 5.0/128
x, y = np.linspace(0.0, 1.0, 128), np.linspace(0.0, 2.0, 256)
xx, yy = np.meshgrid(x, y)
mask = (((xx-0.5)**2+(yy-0.5)**2)<=rr*rr)
# Training
best_loss = 1e12
for epoch in range(MAX_EPOCH):
    # Training
    train_loss = 0.0
    train_fu2 = 0.0
    train_fv2 = 0.0
    train_fp2 = 0.0
    train_mse_data = 0.0
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        model.train()
        ic, seq = data
        ic, seq = ic.cuda(), seq.cuda()
        rho = ic[:, 3, 0, 0] * 40000
        optimizer.zero_grad()
        u0 = ic[:, :3, :, :]
        output = model(ic)
        output = torch.cat((u0.reshape(u0.shape[0], 1, u0.shape[1], 
                                       u0.shape[2], u0.shape[3]),
                            output), dim=1)
        loss, fu2, fv2, fp2, mse_data = compute_loss(output[:, :, :3, :, :], loss_func,
                                                     seq, rho, mask, 1.0, 1.0)
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        train_fu2 += fu2.item()
        train_fv2 += fv2.item()
        train_fp2 += fp2.item()
        train_mse_data += mse_data.item()
        optimizer.step()
    scheduler.step()
    train_loss /= len(train_dataloader)
    train_fu2 /= len(train_dataloader)
    train_fv2 /= len(train_dataloader)
    train_fp2 /= len(train_dataloader)
    train_mse_data /= len(train_dataloader)
    # Testing
    test_loss = 0.0
    test_fu2 = 0.0
    test_fv2 = 0.0
    test_fp2 = 0.0
    test_mse_data = 0.0
    with torch.no_grad():
        model.eval()
        for data in test_dataloader:
            ic, seq = data
            ic, seq = ic.cuda(), seq.cuda()
            rho = ic[:, 3, 0, 0] * 40000
            u0 = ic[:, :3, :, :]
            output = model(ic)
            output = torch.cat((u0.reshape(u0.shape[0], 1, u0.shape[1], u0.shape[2], u0.shape[3]), 
                                output), dim=1)
            loss, fu2, fv2, fp2, mse_data = compute_loss(output[:, :, :3, :, :], loss_func, seq, rho, mask, 1.0, 1.0)
            test_loss += loss.item()
            test_fu2 += fu2.item()
            test_fv2 += fv2.item()
            test_fp2 += fp2.item()
            test_mse_data += mse_data.item()
        test_loss /= len(test_dataloader)
        test_fu2 /= len(test_dataloader)
        test_fv2 /= len(test_dataloader)
        test_fp2 /= len(test_dataloader)
        test_mse_data /= len(test_dataloader)
    # Report stuff
    print('[%d/%d %d%%] train: %.10f %.10f %.10f %.10f %10.f test: %.10f %.10f %.10f %.10f %.10f' % ((epoch+1), MAX_EPOCH, ((epoch+1)/MAX_EPOCH*100.0), train_loss, train_fu2, train_fv2, train_fp2, train_mse_data, test_loss, test_fu2, test_fv2, test_fp2, test_mse_data))
    if test_loss < best_loss:
        save_checkpoint(model, optimizer, scheduler, model_save_path)
        best_loss = test_loss
