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

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "batch_size_tune_results.json")
IMG_DIR      = os.path.join(os.path.dirname(__file__), "img")
os.makedirs(IMG_DIR, exist_ok=True)

BATCH_SIZES = [16, 32, 64, 128]
LR     = 0.001  # updated LR from lr_testing.py
EPOCHS = 20     # updated epochs from epoch_testing.py

def evaluate_dataset(model, loader, device, p_idx, sb_idx, classes,
                     best_thresh_p, best_thresh_sb, criterion):
    model.eval()
    probas, trues, total_loss = [], [], 0.0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            by_oh = torch.nn.functional.one_hot(by, num_classes=len(classes)).float()
            total_loss += criterion(out, by_oh).item() * bx.size(0)
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
            by_oh = torch.nn.functional.one_hot(by, num_classes=len(classes)).float()
            loss = crit(model(bx), by_oh)
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
                by_oh = torch.nn.functional.one_hot(by, num_classes=len(classes)).float()
                vl_ep += crit(out, by_oh).item() * bx.size(0)
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

def predict_with_thresholds(probas, trues, p_idx, sb_idx, classes, t_p, t_sb):
    preds  = np.zeros_like(trues)
    ne_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0
    for i, proba in enumerate(probas):
        pp  = proba[p_idx]
        sbp = proba[sb_idx] if sb_idx != -1 else 0.0
        pt  = pp  >= t_p
        sbt = sb_idx != -1 and sbp >= t_sb
        if pt and sbt:
            preds[i] = p_idx if pp >= sbp else sb_idx
        elif pt:
            preds[i] = p_idx
        elif sbt:
            preds[i] = sb_idx
        else:
            preds[i] = ne_idx
    return preds

def save_confusion_matrix(cm, classes, batch_size):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Purples')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label', xlabel='Predicted Label',
           title=f'Confusion Matrix (Test - Batch {batch_size}, Single-Model)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tot = cm[i].sum()
            pct = cm[i, j] / tot * 100 if tot > 0 else 0
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight='bold',
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    path = os.path.join(IMG_DIR, f"confusion_matrix_batch_{batch_size}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

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
    print(f"Device: {device} | Batch size search: {BATCH_SIZES}")
    print("Single-model per fold | Max-PR-AUC selection | CosineAnnealingLR")
    results    = []
    param_key  = "batch_size"
    param_label = "Batch"

    for batch_size in BATCH_SIZES:
        lr     = LR
        epochs = EPOCHS
        print(f"\n{'='*55}\nBatch Size = {batch_size}")
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        f_tr_loss, f_tr_mac, f_tr_ph = [], [], []
        f_vl_loss, f_vl_mac, f_vl_ph = [], [], []
        f_te_loss, f_te_mac, f_te_ph = [], [], []
        best_fold = {"val_ph": -1}

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

            # Full test probas for confusion matrix
            te_ld = make_ld(X_test_raw, y_test)
            model.eval()
            te_probas, te_trues_list, te_loss_sum = [], [], 0.0
            with torch.no_grad():
                for bx, by in te_ld:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    by_oh = torch.nn.functional.one_hot(by, num_classes=len(classes)).float()
                    te_loss_sum += crit_e(out, by_oh).item() * bx.size(0)
                    te_probas.extend(torch.softmax(out, dim=1).cpu().numpy())
                    te_trues_list.extend(by.cpu().numpy())
            te_probas = np.array(te_probas)
            te_trues_np = np.array(te_trues_list)
            tel = te_loss_sum / len(te_ld.dataset)

            te_preds = predict_with_thresholds(te_probas, te_trues_np, p_idx, sb_idx, classes, t_p, t_sb)
            tem = f1_score(te_trues_np, te_preds, average="macro",  zero_division=0)
            tep = f1_score(te_trues_np, te_preds, labels=[p_idx], average="macro", zero_division=0)

            f_tr_loss.append(trl); f_tr_mac.append(trm); f_tr_ph.append(trp)
            f_vl_loss.append(vll); f_vl_mac.append(vlm); f_vl_ph.append(vlp)
            f_te_loss.append(tel); f_te_mac.append(tem); f_te_ph.append(tep)

            if vlp > best_fold["val_ph"]:
                best_fold = {"val_ph": vlp, "te_preds": te_preds, "te_trues": te_trues_np}

            print(f"  Fold {fold+1} | P_thr={t_p:.4f} SB_thr={t_sb:.4f} "
                  f"| Val Pothole F1={vlp:.4f} | Test Pothole F1={tep:.4f}")

        cm = confusion_matrix(best_fold["te_trues"], best_fold["te_preds"])
        save_confusion_matrix(cm, classes, batch_size)

        prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
            best_fold["te_trues"], best_fold["te_preds"],
            labels=range(len(classes)), zero_division=0)
        class_metrics = {
            cn: {"precision": float(prec_c[ci]), "recall": float(rec_c[ci]), "f1_score": float(f1_c[ci])}
            for ci, cn in enumerate(classes)
        }

        run_res = {
            "batch_size": batch_size,
            "train_loss": float(np.mean(f_tr_loss)), "train_macro_f1": float(np.mean(f_tr_mac)),
            "train_pothole_f1": float(np.mean(f_tr_ph)),
            "val_loss":   float(np.mean(f_vl_loss)), "val_macro_f1":   float(np.mean(f_vl_mac)),
            "val_pothole_f1":   float(np.mean(f_vl_ph)),
            "test_loss":  float(np.mean(f_te_loss)), "test_macro_f1":  float(np.mean(f_te_mac)),
            "test_pothole_f1":  float(np.mean(f_te_ph)),
            "class_metrics": class_metrics,
        }
        results.append(run_res)
        print(f"-> Batch={batch_size} | Train F1={run_res['train_pothole_f1']:.4f} "
              f"| Val F1={run_res['val_pothole_f1']:.4f} | Test F1={run_res['test_pothole_f1']:.4f}")
        for cn, m in class_metrics.items():
            print(f"   {cn:<15} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1_score']:.3f}")
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
    print("\nPer-class metrics (Test Set - Best Single-Model Fold):")
    for r in results:
        print(f"\n  Batch Size = {r['batch_size']}")
        for cn, m in r["class_metrics"].items():
            print(f"    {cn:<15} Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1_score']:.4f}")

if __name__ == "__main__":
    main()
