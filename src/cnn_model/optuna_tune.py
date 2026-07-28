import os
import sys
import json
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import optuna

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))

from model import InceptionTime1D
from train import DynamicJitterDataset, MultiClassFocalLoss, set_seed
from config import CNN_OUT_DIR
from data_utils import get_stratified_group_split

DATA_DIR = CNN_OUT_DIR
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "best_optuna_params.json")

def evaluate_loader(model, loader, device, classes, criterion):
    model.eval()
    probas, trues, total_loss = [], [], 0.0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            total_loss += loss.item() * bx.size(0)
            probs = torch.softmax(out, dim=1)
            probas.extend(probs.cpu().numpy())
            trues.extend(by.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    y_true = np.array(trues)
    y_pred = np.argmax(probas, axis=1)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    
    return avg_loss, macro_f1

def make_loader(X_np, y_np, batch_size, is_train=False):
    # Data is now pre-scaled globally before being passed here
    if is_train:
        ds = DynamicJitterDataset(
            torch.tensor(X_np, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.long),
            max_jitter=15, noise_std=0.02, scale_range=(0.85, 1.15),
            time_warp_prob=0.8, channel_drop_prob=0.1, is_train=True
        )
    else:
        ds = DynamicJitterDataset(
            torch.tensor(X_np, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.long),
            max_jitter=0, noise_std=0, scale_range=None,
            time_warp_prob=0, channel_drop_prob=0, is_train=False
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=is_train)

def objective(trial):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Search Space
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    channels = trial.suggest_categorical("channels", [16, 32, 48, 64])
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    epochs = 50
    n_splits = 4
    
    # Load Data (Only Dev set)
    X_all = np.load(os.path.join(DATA_DIR, "cnn_1d_X.npy"))
    y_all = np.load(os.path.join(DATA_DIR, "cnn_1d_y.npy"))
    groups_all = np.load(os.path.join(DATA_DIR, "cnn_1d_groups.npy"))
    
    dev_g, test_g = get_stratified_group_split(groups_all, y_all, train_ratio=0.8)
    dev_m = np.isin(groups_all, dev_g)
    
    X_dev, y_raw_dev, groups_dev = X_all[dev_m], y_all[dev_m], groups_all[dev_m]
    
    le = LabelEncoder()
    y_dev = le.fit_transform(y_raw_dev)
    classes = le.classes_
    in_channels = X_dev.shape[1]
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(sgkf.split(X_dev, y_dev, groups_dev))
    
    models = []
    optimizers = []
    schedulers = []
    criterions = []
    tr_loaders = []
    vl_loaders = []
    
    # Initialize components for each fold
    for tr_idx, vl_idx in splits:
        X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
        X_vl, y_vl = X_dev[vl_idx], y_dev[vl_idx]
        
        # Calculate global mean and std from X_tr
        global_means = np.mean(X_tr, axis=(0, 2), keepdims=True)
        global_stds = np.std(X_tr, axis=(0, 2), keepdims=True)
        global_stds = np.where(global_stds < 1e-6, 1.0, global_stds)
        
        X_tr_scaled = (X_tr - global_means) / global_stds
        X_vl_scaled = (X_vl - global_means) / global_stds
        
        tr_ld = make_loader(X_tr_scaled, y_tr, batch_size, is_train=True)
        vl_ld = make_loader(X_vl_scaled, y_vl, batch_size, is_train=False)
        
        model = InceptionTime1D(in_channels=in_channels, num_classes=len(classes), channels=channels, dropout_rate=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw = cw / cw.sum() * len(cw)
        criterion = MultiClassFocalLoss(weight=None, gamma=gamma)
        
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        criterions.append(criterion)
        tr_loaders.append(tr_ld)
        vl_loaders.append(vl_ld)
    
    fold_best_macro = [0.0] * n_splits
    
    for epoch in range(epochs):
        epoch_val_macros = []
        
        for fold in range(n_splits):
            models[fold].train()
            for bx, by in tr_loaders[fold]:
                bx, by = bx.to(device), by.to(device)
                optimizers[fold].zero_grad()
                loss = criterions[fold](models[fold](bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(models[fold].parameters(), max_norm=1.0)
                optimizers[fold].step()
            
            schedulers[fold].step()
            
            vl_loss, vl_macro = evaluate_loader(models[fold], vl_loaders[fold], device, classes, criterions[fold])
            epoch_val_macros.append(vl_macro)
            
            if vl_macro > fold_best_macro[fold]:
                fold_best_macro[fold] = vl_macro
        
        current_mean_macro = np.mean(epoch_val_macros)
        trial.report(current_mean_macro, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    final_mean = np.mean(fold_best_macro)
    final_std = np.std(fold_best_macro)
    robust_score = final_mean - final_std
    
    logger.info(f"Trial {trial.number} finished. Mean F1: {final_mean:.4f}, Std F1: {final_std:.4f}, Score: {robust_score:.4f}")
    
    return robust_score

if __name__ == "__main__":
    logger.info("Starting Optuna Multivariate Search for 1D-CNN (10 Trials)...")
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=20)
    )
    study.optimize(objective, n_trials=10)
    
    logger.info("Study finished!")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best robust score: {study.best_value:.4f}")
    logger.info("Best Params:")
    for key, value in study.best_trial.params.items():
        logger.info(f"  {key}: {value}")
        
    with open(RESULTS_PATH, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    logger.info(f"Saved best params to {RESULTS_PATH}")
