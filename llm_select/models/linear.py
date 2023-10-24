import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import random

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)
from netcal.metrics import ECE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
torch.set_default_dtype(torch.float32)

from ray import air
from ray.air import session

DEFAULT_SEED = 42

class LinearModel(torch.nn.Module):
    '''Logistic/linear regression model.'''

    def __init__(self, d_in, d_out=1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.linear(x)

def train(
    config, 
    datasets,
    epochs=10, 
    tune=False,
    return_metrics=False,
    return_model=False,
    show_summary=False,
    seed=DEFAULT_SEED
):
    '''Training function for MLP, compatible with Ray Tune.'''

    # Seed the search algorithms and schedulers
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Unpack datasets and set up loaders
    train_dataset, val_dataset, test_dataset = datasets
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Data config
    d_in = train_dataset.X.shape[-1]
    pred_type = train_dataset.get_pred_type()

    model = LinearModel(
        d_in=d_in,
        d_out=1,
    ).to(device)

    if show_summary:
        print(summary(model))

    if pred_type == 'classification':
        # Compute the positive weight for dataset
        Y_train = torch.Tensor(train_dataset.Y).to('cpu') # TODO: Should use something like torch.ToTensor()
        class_weights = compute_class_weight(
            class_weight='balanced', classes=np.unique(Y_train), y=np.array(Y_train)
        )
        pos_weight = torch.Tensor([class_weights[1]/class_weights[0]]).to(device)

        # Loss function
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif pred_type == 'regression':
        # Loss function
        criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['l2'])

    # Train and evaluate
    all_metrics_list = []
    with tqdm(range(epochs), unit='epochs', disable=tune) as tepoch: 
        for epoch in tepoch:
            if not tune:
                tepoch.set_description(f'Epoch {epoch+1}')

            model.train()
            for batch in train_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                elif len(batch) == 3:
                    x, y, g = batch
                    x, y, g = x.to(device), y.to(device), g.to(device)

                optimizer.zero_grad()
                outputs = model(x.float()).view(-1)
                loss = criterion(outputs, torch.squeeze(y.float()))
                loss.backward()
                optimizer.step()

            # Compute and track evaluation metrics
            train_metrics = eval(model, train_loader, criterion, prefix='train')
            val_metrics = eval(model, val_loader, criterion, prefix='val')
            test_metrics = eval(model, test_loader, criterion, prefix='test')

            if not tune:
                tepoch.set_postfix(**train_metrics, **val_metrics)

            all_metrics = train_metrics | val_metrics | test_metrics
            all_metrics_list.append(all_metrics)

            if tune:
                os.makedirs(f'linear_{epoch}', exist_ok=True)
                torch.save((model.state_dict(), optimizer.state_dict()), f'linear_{epoch}/checkpoint.pt')
                checkpoint = air.checkpoint.Checkpoint.from_directory(f'linear_{epoch}')
                
                # Add group-wise metrics
                session.report(all_metrics, checkpoint=checkpoint)

    print('Finished training.')

    if return_metrics:
        # Combine the metrics across all epochs
        all_metrics_combined = {}

        for d in all_metrics_list:
            for k, v in d.items():
                if k in all_metrics_combined:
                    all_metrics_combined[k].append(v)
                else:
                    all_metrics_combined[k] = [v]

        if return_model:
            return model, all_metrics_combined
        else:
            return all_metrics_combined

    elif return_model:
        return model

def eval(model, dataloader, criterion, prefix=''):
    '''Evaluation function for logistic/linear regression, compatible with Ray Tune.'''
    
    total_loss = 0.
    steps = 0
    out_list = []
    y_list = []

    # Check prediction type
    if isinstance(criterion, nn.MSELoss):
        pred_type = 'regression'
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        pred_type = 'classification'
    else:
        raise RuntimeError(f'[Error] Unhandled loss function. Got: {type(criterion)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(device), y.to(device)
            elif len(batch) == 3:
                x, y, g = batch
                x, y, g = x.to(device), y.to(device), g.to(device)

            out = model(x.float()).view(-1) # [N,]
            
            if pred_type == 'classification':
                out = torch.sigmoid(out.data) # [N,]
            
            # Validation loss
            loss = criterion(out, torch.squeeze(y.float()))
            total_loss += loss.item()
            steps += 1

            out_list.append(out.tolist())
            y_list.append(y.tolist())
    
    outs = np.concatenate(out_list, axis=0)
    ys = np.concatenate(y_list, axis=0)
    
    eval_metrics = {}
    eval_metrics[f'{prefix}_loss'] = total_loss / steps # Average loss
    
    # Regression metrics: MSE, MAE
    if pred_type == 'regression':
        eval_metrics[f'{prefix}_mse'] = mean_squared_error(outs, ys)
        eval_metrics[f'{prefix}_mae'] = mean_absolute_error(outs, ys)

    # Classification metrics: AUROC, AUPRC, Balanced Accuracy, ECE
    elif pred_type == 'classification':
        preds = (outs >= 0.5).astype(int)
        eval_metrics[f'{prefix}_auroc'] = roc_auc_score(ys, outs)
        eval_metrics[f'{prefix}_auprc'] = average_precision_score(ys, outs)
        eval_metrics[f'{prefix}_b_acc'] = balanced_accuracy_score(ys, preds)
        eval_metrics[f'{prefix}_ece'] = ECE(bins=10).measure(outs, ys)

    return eval_metrics