import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "05_pipeline_experiment"))

from model import InceptionTime1D
from train import DynamicJitterDataset, MultiLabelFocalLoss, set_seed, get_stratified_group_split, scale_instance_level

from config import CNN_OUT_DIR
DATA_DIR = CNN_OUT_DIR

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "lr_tune_results.json")

LRS        = [0.1, 0.01, 0.001, 0.0001, 0.00001]
EPOCHS     = 20    # same as train_cnn.py
BATCH_SIZE = 32    # same as train_cnn.py

def evaluate_dataset(model, loader, device, p_idx, sb_idx, classes,
                     best_thresh_p, best_thresh_sb, criterion):
    model.eval()
    probas, trues, total_loss = [], [], 0.0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            total_loss += criterion(out, by).item() * bx.size(0)
            probs = torch.softmax(out, dim=1)
            probas.extend(probs.cpu().numpy())
            trues.extend(by.cpu().numpy())
    probas   = np.array(probas)
    trues    = np.array(trues)
    avg_loss = total_loss / len(loader.dataset)

    preds  = np.zeros_like(trues)
    ne_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0
    for i, proba in enumerate(probas):
        pp  = proba[p_idx]
        sbp = proba[sb_idx] if sb_idx != -1 else 0.0
        pt  = pp  >= best_thresh_p
        sbt = sb_idx != -1 and sbp >= best_thresh_sb
        if pt and sbt:
            preds[i] = p_idx if pp >= sbp else sb_idx
        elif pt:
            preds[i] = p_idx
        elif sbt:
            preds[i] = sb_idx
        else:
            preds[i] = ne_idx
    macro_f1   = f1_score(trues, preds, average="macro",  zero_division=0)
    pothole_f1 = f1_score(trues, preds, labels=[p_idx], average="macro", zero_division=0)
    return avg_loss, macro_f1, pothole_f1

def train_fold_model(X_tr, y_tr, X_vl, y_vl, lr, epochs, batch_size,
                     p_idx, sb_idx, classes, device):
    """Identical to train_cnn.py: CosineAnnealingLR + Max-minority-PR-AUC selection."""
    def to_ds(X, y, train=False):
        Xs = scale_instance_level(X)
        return DynamicJitterDataset(
            torch.tensor(Xs, dtype=torch.float32),
            torch.tensor(y,  dtype=torch.long),
            max_jitter=15 if train else 0,
            noise_std=0.02 if train else 0,
            scale_range=(0.85, 1.15) if train else None,
            is_train=train)

    tr_ld = DataLoader(to_ds(X_tr, y_tr, True),  batch_size=batch_size, shuffle=True)
    vl_ld = DataLoader(to_ds(X_vl, y_vl, False), batch_size=batch_size, shuffle=False)

    cw   = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw   = cw / cw.sum() * len(cw)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

    model = InceptionTime1D(in_channels=X_tr.shape[1], num_classes=len(classes), num_blocks=3, channels=128, bottleneck_channels=32, dropout_rate=0.5).to(device)
    crit = MultiLabelFocalLoss(weight=cw_t, gamma=2.0)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_prauc, best_vloss, best_state = -1.0, float("inf"), None

    for _ in range(epochs):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sch.step()

        model.eval()
        vl_ep, vp, vt = 0.0, [], []
        with torch.no_grad():
            for bx, by in vl_ld:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                vl_ep += crit(out, by).item() * bx.size(0)
                vp.extend(torch.softmax(out, dim=1).cpu().numpy())
                vt.extend(by.cpu().numpy())
        vl_ep /= len(vl_ld.dataset)
        vp, vt = np.array(vp), np.array(vt)

        pp, rp, _ = precision_recall_curve((vt == p_idx).astype(int), vp[:, p_idx])
        pa = auc(rp, pp)
        if sb_idx != -1:
            psb, rsb, _ = precision_recall_curve((vt == sb_idx).astype(int), vp[:, sb_idx])
            mn_prauc = (pa + auc(rsb, psb)) / 2.0
        else:
            mn_prauc = pa

        if mn_prauc > best_prauc or (np.isclose(mn_prauc, best_prauc) and vl_ep < best_vloss):
            best_prauc, best_vloss, best_state = mn_prauc, vl_ep, model.state_dict()

    model.load_state_dict(best_state)

    # Threshold optimisation on val
    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for bx, by in vl_ld:
            vp.extend(torch.softmax(model(bx.to(device)), dim=1).cpu().numpy())
            vt.extend(by.numpy())
    vp, vt = np.array(vp), np.array(vt)

    t_p = 0.5
    t_sb = 0.5

    return model, None, None, t_p, t_sb


def main():
    set_seed(42)
    X_all      = np.load(os.path.join(DATA_DIR, "cnn_1d_X.npy"))
    y_all      = np.load(os.path.join(DATA_DIR, "cnn_1d_y.npy"))
    groups_all = np.load(os.path.join(DATA_DIR, "cnn_1d_groups.npy"))

    dev_g, test_g = get_stratified_group_split(groups_all, y_all, train_ratio=0.7)
    dev_m, test_m = np.isin(groups_all, dev_g), np.isin(groups_all, test_g)

    X_dev, y_raw_dev, groups_dev = X_all[dev_m], y_all[dev_m], groups_all[dev_m]
    X_test_raw, y_test_raw       = X_all[test_m], y_all[test_m]

    le      = LabelEncoder()
    y_dev   = le.fit_transform(y_raw_dev)
    y_test  = le.transform(y_test_raw)
    classes = le.classes_
    p_idx   = list(classes).index("Pothole")
    sb_idx  = list(classes).index("Speed Bump") if "Speed Bump" in classes else -1
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | LR search: {LRS}")
    print("Single-model per fold | Max-PR-AUC selection | CosineAnnealingLR")
    results    = []
    param_key  = "lr"
    param_label = "LR"

    for lr in LRS:
        epochs     = EPOCHS
        batch_size = BATCH_SIZE
        print(f"\n{'='*55}\nLR = {lr}")
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        f_tr_loss, f_tr_mac, f_tr_ph = [], [], []
        f_vl_loss, f_vl_mac, f_vl_ph = [], [], []
        f_te_loss, f_te_mac, f_te_ph = [], [], []

        for fold, (tr_idx, vl_idx) in enumerate(sgkf.split(X_dev, y_dev, groups_dev)):
            X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
            X_vl, y_vl = X_dev[vl_idx], y_dev[vl_idx]

            model, ch_means, ch_stds, t_p, t_sb = train_fold_model(
                X_tr, y_tr, X_vl, y_vl,
                lr=lr, epochs=epochs, batch_size=batch_size,
                p_idx=p_idx, sb_idx=sb_idx, classes=classes, device=device)

            def make_ld(X_np, y_np, bs=batch_size):
                Xs = scale_instance_level(X_np)
                ds  = DynamicJitterDataset(
                    torch.tensor(Xs, dtype=torch.float32),
                    torch.tensor(y_np, dtype=torch.long),
                    max_jitter=0, noise_std=0, scale_range=None, is_train=False)
                return DataLoader(ds, batch_size=bs, shuffle=False)

            cw_e   = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
            cw_e   = cw_e / cw_e.sum() * len(cw_e)
            crit_e = MultiLabelFocalLoss(weight=torch.tensor(cw_e, dtype=torch.float32).to(device), gamma=2.0)

            trl, trm, trp = evaluate_dataset(model, make_ld(X_tr, y_tr),        device, p_idx, sb_idx, classes, t_p, t_sb, crit_e)
            vll, vlm, vlp = evaluate_dataset(model, make_ld(X_vl, y_vl),        device, p_idx, sb_idx, classes, t_p, t_sb, crit_e)
            tel, tem, tep = evaluate_dataset(model, make_ld(X_test_raw, y_test), device, p_idx, sb_idx, classes, t_p, t_sb, crit_e)

            f_tr_loss.append(trl); f_tr_mac.append(trm); f_tr_ph.append(trp)
            f_vl_loss.append(vll); f_vl_mac.append(vlm); f_vl_ph.append(vlp)
            f_te_loss.append(tel); f_te_mac.append(tem); f_te_ph.append(tep)
            print(f"  Fold {fold+1} | P_thr={t_p:.4f} SB_thr={t_sb:.4f} "
                  f"| Val Pothole F1={vlp:.4f} | Test Pothole F1={tep:.4f}")
        run_res = {
            "lr": lr,
            "train_loss": float(np.mean(f_tr_loss)), "train_macro_f1": float(np.mean(f_tr_mac)),
            "train_pothole_f1": float(np.mean(f_tr_ph)),
            "val_loss":   float(np.mean(f_vl_loss)), "val_macro_f1":   float(np.mean(f_vl_mac)),
            "val_pothole_f1":   float(np.mean(f_vl_ph)),
            "test_loss":  float(np.mean(f_te_loss)), "test_macro_f1":  float(np.mean(f_te_mac)),
            "test_pothole_f1":  float(np.mean(f_te_ph)),
        }
        results.append(run_res)
        print(f"-> LR={lr} | Train F1={run_res['train_pothole_f1']:.4f} "
              f"| Val F1={run_res['val_pothole_f1']:.4f} | Test F1={run_res['test_pothole_f1']:.4f}")
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=4)
    results = sorted(results, key=lambda x: x["val_pothole_f1"], reverse=True)
    print("\n" + "="*85)
    print(f"{'Rank':<6} {param_label:<10} {'TrainLoss':<11} {'TrainF1':<9} {'ValLoss':<10} {'ValF1':<9} {'TestLoss':<10} {'TestF1':<9}")
    print("-"*85)
    for i, r in enumerate(results):
        print(f"{i+1:<6} {str(r[param_key]):<10} {r['train_loss']:<11.4f} {r['train_pothole_f1']:<9.4f} "
              f"{r['val_loss']:<10.4f} {r['val_pothole_f1']:<9.4f} {r['test_loss']:<10.4f} {r['test_pothole_f1']:<9.4f}")
    print("="*85)
    print(f"Best {param_label} (by Val Pothole F1): {results[0][param_key]}")

if __name__ == "__main__":
    main()
