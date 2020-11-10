import os
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
import torch.optim as optim
import torch.nn.utils.prune

import tensorflow as tf
import tensorboard as tb
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

import warnings

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.vae import Autoencoder
from src.sylvesterVAE import SylvesterVAE
from src.utils.dataloader import MoADataset

from sklearn.preprocessing import Normalizer, MinMaxScaler

warnings.filterwarnings('ignore')

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 1000
EARLY_STOPPING_STEPS = 100
EARLY_STOP = True
SAVE_MODELS = True


def l1_regularization(model):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
        del param
    return l1_loss


def seed_everything(seed=1903):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device, l1_reg=0):
    model.train()
    final_loss = 0
    kl_loss = 0
    recon_loss = 0
    final_mse_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs = data['x'].to(device)

        outputs, kl, z = model(inputs)

        kl_div = torch.mean(kl)

        out = torch.abs(outputs - inputs)

        loss_recon = loss_fn(out, torch.zeros_like(outputs))
        mse_loss = torch.nn.MSELoss(reduction='sum')(outputs, inputs)
        loss = loss_recon + kl_div

        regularization_loss = l1_regularization(model)
        loss += l1_reg * regularization_loss
        loss.backward()
        optimizer.step()

        final_loss += loss.item() / len(data)
        final_mse_loss += mse_loss.item() / len(data)
        recon_loss += loss_recon.item() / len(data)
        kl_loss += kl_div.item() / len(data)
        scheduler.step()

        del loss, kl_div, loss_recon, mse_loss, outputs, inputs, data, kl, z, regularization_loss

    final_mse_loss /= len(dataloader)
    final_loss /= len(dataloader)
    kl_loss /= len(dataloader)
    recon_loss /= len(dataloader)

    del dataloader, model

    return final_loss, recon_loss, kl_loss, final_mse_loss


def valid_fn(model, loss_fn, dataloader, scheduler, device, epoch, writer):
    model.eval()
    final_loss = 0
    final_mse_loss = 0
    kl_loss = 0
    recon_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs = data['x'].to(device)
        outputs, kl, z = model(inputs)

        kl_div = torch.mean(kl)

        out = torch.abs(outputs - inputs)

        loss_recon = loss_fn(out, torch.zeros_like(outputs))
        mse_loss = torch.nn.MSELoss(reduction='sum')(outputs, inputs)
        loss = loss_recon + kl_div

        final_loss += loss.item() / len(data)
        recon_loss += loss_recon.item() / len(data)
        kl_loss += kl_div.item() / len(data)
        final_mse_loss += mse_loss.item() / len(data)
        valid_preds.append(outputs.detach().cpu().numpy())
        del loss, kl_div, loss_recon, data, kl, mse_loss

    final_loss /= len(dataloader)
    final_mse_loss /= len(dataloader)
    kl_loss /= len(dataloader)
    recon_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    try:
        writer.add_histogram('original', inputs, epoch)
        writer.add_histogram('recon', outputs, epoch)
    except:
        pass
    min_value = np.min([torch.min(inputs[0]).item(), torch.min(outputs[0]).item()])
    max_value = np.max([torch.max(inputs[0]).item(), torch.max(outputs[0]).item()])

    fig1, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 6))
    axs[0].plot(list(range(len(inputs[0]))), inputs.detach().cpu().numpy()[0])
    axs[1].plot(list(range(len(outputs[0]))), outputs.detach().cpu().numpy()[0])
    axs[0].set_ylim([min_value, max_value])
    axs[1].set_ylim([min_value, max_value])

    try:
        writer.add_figure('Original/Reconstructed inputs', fig1, epoch)
    except:
        pass

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    axs[0].violinplot(inputs.detach().cpu().numpy()[0])
    axs[1].violinplot(outputs.detach().cpu().numpy()[0])

    axs[0].set_ylim([min_value, max_value])
    axs[1].set_ylim([min_value, max_value])

    del outputs, inputs, dataloader, z

    # fig.suptitle("Violin Plotting Examples")
    # fig.subplots_adjust(hspace=0.4)

    try:
        writer.add_figure('Violin', fig, global_step=epoch)
    except:
        pass

    return final_loss, valid_preds, recon_loss, kl_loss, final_mse_loss


def pruning_model(model, ratio):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(module.weight)
            parameters_to_prune += [[module, 'weight']]
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=ratio
    )


def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time', 'cp_dose'])

    return data


seed_everything(seed=1903)


def main(writer,
         first,
         loss_function='mse',
         ftype='o-sylvester',
         optim_type='adam',
         nflows=10,
         lr=1e-4,
         pruning_ratio=0,
         wd=1e-5,
         l1_reg=0,
         z_dim=3,
         ):
    z_dim = z_dim
    # Clear any logs from previous runs

    train_features = pd.read_csv('input/lish-moa/train_features.csv')
    train_targets_scored = pd.read_csv('input/lish-moa/train_targets_scored.csv')

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    train_features[GENES + CELLS] = Normalizer().fit_transform(train_features[GENES + CELLS].values)
    train_features[GENES + CELLS] = MinMaxScaler().fit_transform(train_features[GENES + CELLS].values)

    train = train_features.merge(train_targets_scored, on='sig_id')
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    target = train[train_targets_scored.columns]

    train = train.drop('cp_type', axis=1)

    target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

    folds = train.copy()

    mskf = MultilabelStratifiedKFold(n_splits=6)

    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        folds.loc[v_idx, 'kfold'] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)

    if ftype not in ['o-sylvester', 'h-sylvester', 't-sylvester']:
        model = Autoencoder(
            flow_type=ftype,
            z_dim=z_dim,
            in_features=872,
            hidden_size=128,
            activation=nn.ReLU,
            batchnorm=True,
            n_flows=nflows
        )
    else:
        model = SylvesterVAE(
            flow_type=ftype,
            z_dim=z_dim,
            in_features=872,
            hidden_size=128,
            activation=nn.ReLU,
            batchnorm=True,
            n_flows=nflows
        )

    train_df = folds[folds['kfold'] != 0].reset_index(drop=True)
    valid_df = folds[folds['kfold'] == 0].reset_index(drop=True)

    feature_cols = [c for c in process_data(train_df).columns if c not in target_cols]
    feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]

    x_train, y_train = train_df[feature_cols[:-5]].values, train_df[target_cols].values
    x_valid, y_valid = valid_df[feature_cols[:-5]].values, valid_df[target_cols].values

    if first:
        pbar = tqdm(range(10), position=0, leave=True)
        for i, x in enumerate(x_train[:10]):
            # x = (x - np.min(x)) / (np.max(x) - np.min(x))
            fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
            ax.plot(list(range(len(x))), x)
            min_value = np.min(x_train)
            max_value = np.max(x_train)
            ax.set_ylim([min_value, max_value])
            ax.set_ylim([min_value, max_value])

            writer.add_figure('Original Inputs (individually)', fig1, i)
            pbar.update(1)

        fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))

        if first:
            for i, x in enumerate(x_train[:10]):
                # x = (x - np.min(x)) / (np.max(x) - np.min(x))
                ax.plot(list(range(len(x))), x)
                ax.set_ylim([min_value, max_value])
                ax.set_ylim([min_value, max_value])

        writer.add_figure('Original Inputs (each color is a sample)', fig1, 0)

        writer.add_embedding(
            x_valid,
            # y_valid,
            tag='Raw Values'
        )
        del ax, x, fig1

    pruning_model(model, pruning_ratio)

    # writer.add_graph(model, (torch.zeros_like(torch.Tensor(x_train))))

    model.to(DEVICE)
    # with SummaryWriter(comment='Autoencoder') as w:

    # IMPORTANT: THE DATA IS STANDARDIZED (MU=0, VAR=1) THEN NORMALIZED (MIN=-0.5, MAX=0.5)

    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_dataset))

    traces = {
        'train': {
            'losses': [],
            'kl_divs': [],
            'recons': [],
            'mse': [],
        },
        'valid': {
            'losses': [],
            'kl_divs': [],
            'recons': [],
            'mse': [],
        },
    }

    try:
        saved = torch.load(
            f"saved_models/vae_{loss_function}_{ftype}_{optim_type}_nflows{nflows}_lr{lr}_prune{pruning_ratio}_wd{wd}_l1{l1_reg}.pth")
        model.load_state_dict(saved['state'])
        start_epoch = saved['epoch']

        # If best epoch is 0, then it starts at 1
        # If EPOCHS is 100, the last epoch to be run is 99
        if (start_epoch + 1) >= (EPOCHS - 1):
            return None, None, None, None, None, None, None, None
        optimizer.load_state_dict(saved['optimizer'])
        best_loss = saved['best_loss']
        traces['train']['losses'] = saved['train_losses']
        traces['train']['mse'] = saved['train_mse_losses']
        traces['train']['kl_divs'] = saved['train_kl_divs']
        traces['train']['recons'] = saved['train_recons']
        traces['valid']['losses'] = saved['valid_losses']
        traces['valid']['mse'] = saved['valid_mse_losses']
        traces['valid']['kl_divs'] = saved['valid_kl_divs']
        traces['valid']['recons'] = saved['valid_recons']

        for epoch in range(len(saved['train_losses'])):
            writer.add_scalar('Train/Loss', saved['train_losses'][epoch], epoch)
            writer.add_scalar('Train/MSE Loss', saved['train_mse_losses'][epoch], epoch)
            writer.add_scalar('Train/KL DIV', saved['train_kl_divs'][epoch], epoch)
            writer.add_scalar('Train/Recon', saved['train_recons'][epoch], epoch)
            writer.add_scalar('Valid/Loss', saved['valid_losses'][epoch], epoch)
            writer.add_scalar('Valid/MSE Loss', saved['valid_mse_losses'][epoch], epoch)
            writer.add_scalar('Valid/KL DIV', saved['valid_kl_divs'][epoch], epoch)
            writer.add_scalar('Valid/Recon', saved['valid_recons'][epoch], epoch)
        del saved

        print('Model Loaded successfully')

    except:
        print('No saved model found.')
        start_epoch = 0
        best_loss = np.inf

    best_train_loss = np.inf
    best_train_mse_loss = np.inf
    best_train_kld_loss = np.inf
    best_train_recon_loss = np.inf

    best_valid_loss1 = np.inf
    best_valid_mse_loss = np.inf
    best_valid_kld_loss = np.inf
    best_valid_recon_loss = np.inf

    if loss_function == 'bce':
        loss_fn = nn.BCELoss(reduction='sum')
    elif loss_function == 'mse':
        loss_fn = nn.MSELoss(reduction='sum')
    elif loss_function == 'mae':
        loss_fn = nn.L1Loss(reduction='sum')
    early_step = 0

    for epoch in range(start_epoch, EPOCHS):

        train_loss, train_recon, train_kl, train_mse = train_fn(model, optimizer, scheduler, loss_fn, trainloader,
                                                                DEVICE, l1_reg)

        if np.isnan(train_loss):
            print('NaN loss. Next hyperparameters...')
            break

        print(f"EPOCH: {epoch}, train_loss: {train_loss}, kl: {train_kl}, recon: {train_recon}, mse: {train_mse}")

        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/MSE Loss', train_mse, epoch)
        writer.add_scalar('Train/KL DIV', train_kl, epoch)
        writer.add_scalar('Train/Recon', train_recon, epoch)
        traces['train']['losses'] += [train_loss]
        traces['train']['kl_divs'] += [train_kl]
        traces['train']['recons'] += [train_recon]
        traces['train']['mse'] += [train_mse]

        model.zs = torch.tensor([])

        valid_loss, valid_preds, valid_recon, valid_kl, valid_mse = valid_fn(model, loss_fn, validloader, scheduler,
                                                                             DEVICE, epoch,
                                                                             writer)

        print(f"EPOCH: {epoch}, valid_loss: {valid_loss}, kl: {valid_kl}, recon: {valid_recon}, mse: {valid_mse}")

        writer.add_scalar('Valid/Loss', valid_loss, epoch)
        writer.add_scalar('Valid/MSE Loss', valid_mse, epoch)
        writer.add_scalar('Valid/KL DIV', valid_kl, epoch)
        writer.add_scalar('Valid/Recon', valid_recon, epoch)
        traces['valid']['losses'] += [valid_loss]
        traces['valid']['recons'] += [valid_recon]
        traces['valid']['kl_divs'] += [valid_kl]
        traces['valid']['mse'] += [valid_mse]

        try:
            writer.add_histogram('hist_dense_1_1', model.dense1.weight, epoch)
            writer.add_histogram('hist_dense_1_2', model.dense11.weight, epoch)

            writer.add_histogram('hist_dense_2_1', model.dense2.weight, epoch)
            writer.add_histogram('hist_dense_2_2', model.dense21.weight, epoch)

            writer.add_histogram('hist_mu', model.GaussianSample.mu.weight, epoch)
            writer.add_histogram('hist_logvar', model.GaussianSample.log_var.weight, epoch)
        except:
            pass
        if valid_loss < best_loss:

            best_loss = valid_loss

            best_train_loss = traces['train']['losses'][-1]
            best_train_kld_loss = traces['train']['kl_divs'][-1]
            best_train_recon_loss = traces['train']['recons'][-1]
            best_train_mse_loss = traces['train']['mse'][-1]
            best_valid_loss1 = traces['valid']['losses'][-1]
            best_valid_kld_loss = traces['valid']['kl_divs'][-1]
            best_valid_recon_loss = traces['valid']['recons'][-1]
            best_valid_mse_loss = traces['valid']['mse'][-1]

            writer.add_embedding(
                model.zs,
                global_step=epoch,
                tag='z_' + str(epoch)
            )

            if SAVE_MODELS:
                torch.save(
                    {
                        'state': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'train_losses': traces['train']['losses'],
                        'train_mse_losses': traces['train']['mse'],
                        'train_kl_divs': traces['train']['kl_divs'],
                        'train_recons': traces['train']['recons'],
                        'valid_losses': traces['valid']['losses'],
                        'valid_mse_losses': traces['valid']['mse'],
                        'valid_kl_divs': traces['valid']['kl_divs'],
                        'valid_recons': traces['valid']['recons'],
                    },
                    f"saved_models/vae_{loss_function}_{ftype}_{optim_type}_nflows{nflows}_lr{lr}_prune{pruning_ratio}_wd{wd}_l1{l1_reg}.pth")

        elif (EARLY_STOP == True):

            early_step += 1
            model.zs = torch.empty(size=(0, z_dim))
            if (early_step >= EARLY_STOPPING_STEPS):
                break
        model.zs = torch.empty(size=(0, z_dim))

        del train_loss, train_recon, train_kl, valid_loss, valid_preds, valid_recon, valid_kl, train_mse, valid_mse
    del train_dataset, valid_dataset, model, traces, optimizer, scheduler, loss_fn, valid_df, target_cols, \
        train_df, train, train_targets_scored, train_features, folds, mskf, x_valid, x_train, y_train, y_valid, target, \
        trainloader, validloader, best_loss
    return best_train_loss, best_train_mse_loss, best_train_kld_loss, best_train_recon_loss, \
           best_valid_loss1, best_valid_mse_loss, best_valid_kld_loss, best_valid_recon_loss


if __name__ == "__main__":
    first = True
    best_valid_loss = np.inf
    best_ftype = ''
    best_optim_type = ''
    best_lr = np.inf
    best_nflows = 0
    best_pruning_ratio = 0
    best_wd = np.inf
    best_l1_reg = np.inf
    best_loss_fn = ''

    z_dim = 3
    for loss_fn in ['mse']:
        for ftype in ['o-sylvester']:
            for optim_type in ['adam']:
                for nflows in [3, 10, 100]:
                    for lr in [1e-3]:
                        for pruning_ratio in [0.2]:
                            for wd in [1e-3]:
                                for l1 in [1e-3]:
                                    print(ftype, optim_type, lr, wd, l1)
                                    writer = SummaryWriter(
                                        f"runs/"
                                        f"zdim{z_dim}/"
                                        f"{loss_fn}/"
                                        f"{ftype}/"
                                        f"{optim_type}/"
                                        f"nflows{nflows}/"
                                        f"lr{lr}/"
                                        f"prune{pruning_ratio}/"
                                        f"wd{wd}/"
                                        f"l1{l1}"
                                    )
                                    train_loss, train_mse, train_kld, train_recon_loss, valid_loss, valid_mse, valid_kld, valid_recon_loss = \
                                        main(writer, first, loss_fn, ftype, optim_type, nflows, lr, pruning_ratio, wd,
                                             l1, z_dim=z_dim)
                                    if train_loss is None:
                                        break
                                    writer.add_hparams({
                                        'Loss_fn': loss_fn,
                                        'Ftype': ftype,
                                        'Optim_type': optim_type,
                                        'Nflows': nflows,
                                        'z_dim': z_dim,
                                        'LR': lr,
                                        'Pruning Ratio (L1 Unstructured)': pruning_ratio,
                                        'L1_reg': l1,
                                        'Weight Decay': wd
                                    },
                                        {
                                            'Train Loss': train_loss,
                                            'Train MSE Loss': train_mse,
                                            'Train KLD Loss': train_kld,
                                            'Train Recon Loss': train_recon_loss,
                                            'Valid Loss': valid_loss,
                                            'Valid MSE Loss': valid_mse,
                                            'Valid KLD Loss': valid_kld,
                                            'Valid Recon Loss': valid_recon_loss,
                                        })
                                    if valid_loss < best_valid_loss:
                                        best_loss_fn = loss_fn
                                        best_valid_loss = valid_loss
                                        best_ftype = ftype
                                        best_optim_type = optim_type
                                        best_nflows = nflows
                                        best_lr = lr
                                        best_wd = wd
                                        best_l1_reg = l1
                                        best_pruning_ratio = pruning_ratio
                                    first = False
                                    writer.close()
                                    del valid_kld, valid_recon_loss, valid_loss, train_kld, train_recon_loss, \
                                        train_loss

    """
    
    EPOCHS = 10000
    writer = SummaryWriter(f"runs/zdim{z_dim}/{ftype}/{optim_type}/lr{lr}/wd{wd}/l1{l1}")
    train_loss, train_kld, train_recon_loss, valid_loss, valid_kld, valid_recon_loss = main(writer, best_ftype,
                                                                                            best_optim_type,
                                                                                            best_lr, best_wd,
                                                                                            best_l1_reg)
    writer.add_hparams({
        'ftype': best_ftype,
        'optim_type': best_optim_type,
        'LR': best_lr,
        'L1_reg': best_l1_reg,
        'Weight Decay': best_wd
    },
        {
            'Train Loss': train_loss,
            'Train KLD Loss': train_kld,
            'Train Recon Loss': train_recon_loss,
            'Valid Loss': valid_loss,
            'Valid KLD Loss': valid_kld,
            'Valid Recon Loss': valid_recon_loss,
        })

    """
