# Standard Library Imports  
from itertools import combinations
import warnings

# Suppress Specific Warnings  
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Data Handling  
import pandas as pd
import numpy as np

# PyTorch  
import torch
import torch.nn.functional as F

# Evaluation Metrics  
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# Visualization  
import matplotlib.pyplot as plt

# Device Configuration  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Color Mapping for Model Names  
colors = {
    "Contrastive": "#63AAC0",    # Blue tone
    "MS-Loss":     "#F99B45",    # Orange tone
    "Circle-Loss": "#284E60"     # Dark teal tone
}


def compute_matrix(embeddings, loss_type):
    """
    Compute a similarity or distance matrix based on the loss type.
    """
    if loss_type == "Contrastive":
        dists = torch.cdist(embeddings, embeddings, p=2)
        dists.fill_diagonal_(0.0)
        norm_dists = dists / dists.max()
        return norm_dists.cpu().numpy(), "Euclidean Distance (normalized)", True
    else:
        emb = F.normalize(embeddings, p=2, dim=1)
        sims = emb @ emb.T
        sims.fill_diagonal_(1.0)
        return sims.cpu().numpy(), "Cosine Similarity", False


def add_thumbnails(ax, images, side="left", pad=0.003):
    """
    Add small image thumbnails to a heatmap axis.
    """
    fig, n = ax.figure, len(images)
    axpos = ax.get_position()
    cell_size = (axpos.height if side == "left" else axpos.width) / n

    for i, img in enumerate(images):
        if side == "left":
            x0 = axpos.x0 - pad - cell_size
            y0 = axpos.y0 + axpos.height - (i + 1) * cell_size
        else:
            x0 = axpos.x0 + i * cell_size
            y0 = axpos.y1 + pad
        thumb = fig.add_axes([x0, y0, cell_size, cell_size])
        thumb.imshow(np.asarray(img))
        thumb.axis("off")


def collect_scores(model, dataloader, device, sim='cosine'):
    """
    Collect similarity/distance scores and labels from the model using a dataloader.
    """
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for img1, img2, lbl, *_ in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            e1, e2 = model(img1, img2)
            if sim == 'cosine':
                s = -F.cosine_similarity(e1, e2)  # Negated so that higher = more dissimilar
            else:
                s = F.pairwise_distance(e1, e2)
            scores.extend(s.cpu().numpy())
            labels.extend(lbl.view(-1).cpu().numpy())
    return np.asarray(scores), np.asarray(labels)


def collect_scores_fixed(model, pairs, sim='cosine'):
    """
    Collect similarity/distance scores and labels from a fixed list of image pairs.
    """
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for img1, img2, lbl in pairs:
            e1, e2 = model(img1, img2)
            if sim == 'cosine':
                s = -F.cosine_similarity(e1, e2)
            else:
                s = F.pairwise_distance(e1, e2)
            scores.extend(s.cpu().numpy())
            labels.extend(lbl.view(-1).cpu().numpy())
    return np.asarray(scores), np.asarray(labels)


def metrics_at_threshold(scores, labels, thr):
    """
    Compute accuracy, precision, recall, and F1-score at a given threshold.
    """
    preds = scores > thr  # Predict 1 if dissimilar
    return (
        accuracy_score(labels, preds),
        precision_score(labels, preds, zero_division=0),
        recall_score(labels, preds, zero_division=0),
        f1_score(labels, preds, zero_division=0)
    )


def evaluate_models(model_paths, dataloader, device, show_plot=True, show_cm=True):
    """
    Evaluate all models in model_paths using ROC, optimal threshold, and classification metrics.
    """
    rows = []
    if show_plot:
        plt.figure(figsize=(7, 7))

    for name, model in model_paths.items():
        sim = 'euclid' if name == "Contrastive" else 'cosine'
        scores, labels = collect_scores(model, dataloader, device, sim=sim)
        fpr, tpr, thr = roc_curve(labels, scores)

        if show_plot:
            plt.plot(fpr, tpr,
                     label=f"{name} (AUROC={auc(fpr, tpr):.3f})",
                     color=colors[name])

        # Find best threshold by F1-score
        f1s = [metrics_at_threshold(scores, labels, t)[-1] for t in thr]
        best_thr = thr[int(np.nanargmax(f1s))]
        acc, prec, rec, f1 = metrics_at_threshold(scores, labels, best_thr)
        auprc = average_precision_score(labels, scores)

        rows.append({
            'model': name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'F1': f1,
            'AUROC': auc(fpr, tpr),
            'AUPRC': auprc,
            'opt_threshold': float(best_thr)
        })

        if show_cm:
            cm = confusion_matrix(labels, scores > best_thr)
            print(f"\nConfusion Matrix – {name} (thr={best_thr:.3f})\n", cm)

    if show_plot:
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC comparison – all models")
        plt.legend()
        plt.grid(True)
        plt.show()

    return pd.DataFrame(rows).set_index('model').round(3)


def evaluate_models_fixed(model_paths, pairs, show_plot=True, show_cm=True):
    """
    Same as evaluate_models but for a fixed list of image pairs (no dataloader).
    """
    rows = []
    if show_plot:
        plt.figure(figsize=(7, 7))

    for name, model in model_paths.items():
        sim = 'euclid' if name == "Contrastive" else 'cosine'
        scores, labels = collect_scores_fixed(model, pairs, sim=sim)
        fpr, tpr, thr = roc_curve(labels, scores)

        if show_plot:
            plt.plot(fpr, tpr,
                     label=f"{name} (AUROC={auc(fpr, tpr):.3f})",
                     color=colors[name])

        f1s = [metrics_at_threshold(scores, labels, t)[-1] for t in thr]
        best_thr = thr[int(np.nanargmax(f1s))]
        acc, prec, rec, f1 = metrics_at_threshold(scores, labels, best_thr)
        auprc = average_precision_score(labels, scores)

        rows.append({
            'model': name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'F1': f1,
            'AUROC': auc(fpr, tpr),
            'AUPRC': auprc,
            'opt_threshold': float(best_thr)
        })

        if show_cm:
            cm = confusion_matrix(labels, scores > best_thr)
            print(f"\nConfusion Matrix – {name} (thr={best_thr:.3f})\n", cm)

    if show_plot:
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.show()

    return pd.DataFrame(rows).set_index('model').round(3)


def plot_grouped_metrics(summary_df):
    """
    Grouped bar plot comparing multiple models on accuracy, precision, recall, and F1.
    """
    metrics = ['accuracy', 'precision', 'recall', 'F1']
    models = summary_df.index.tolist()

    n_metrics = len(metrics)
    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(n_metrics)

    plt.figure(figsize=(9, 5))
    for i, model in enumerate(models):
        values = summary_df.loc[model, metrics].values
        bars = plt.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            label=model,
            color=colors[model]
        )
        # Annotate each bar with its value
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}",
                ha='center', va='bottom', fontsize=9
            )

    plt.xticks(
        x + bar_width * (n_models - 1) / 2,
        ['Accuracy', 'Precision', 'Recall', 'F1'],
        rotation=0
    )
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.legend(title='Loss Function', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def compute_intra_inter(dataloader, model, loss_key):
    """
    Computes intra-class and inter-class distances based on embeddings from the model.
    """
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for img0, img1, _, l0, l1 in dataloader:
            img0, img1 = img0.to(DEVICE), img1.to(DEVICE)
            out0, out1 = model(img0, img1)
            all_emb.append(torch.cat([out0, out1]))
            all_lbl.append(torch.cat([l0.to(DEVICE), l1.to(DEVICE)]))

    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)

    # Group embeddings by label
    cls2emb = {}
    for e, v in zip(emb, lbl):
        cls2emb.setdefault(int(v), []).append(e)
    for k in cls2emb:
        cls2emb[k] = torch.stack(cls2emb[k])

    # Choose distance metric
    if loss_key == "Contrastive":
        metric = F.pairwise_distance
        xlabel = "Euclidean Distance"
    else:
        metric = lambda x, y: 1 - F.cosine_similarity(x, y)
        xlabel = "1 - Cosine Similarity"

    intra, inter = [], []
    keys = list(cls2emb.keys())

    # Compute intra-class distances
    for k in keys:
        em = cls2emb[k]
        for i, j in combinations(range(len(em)), 2):
            d = metric(em[i].unsqueeze(0), em[j].unsqueeze(0))
            intra.append(d.item())

    # Compute inter-class distances
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for e1 in cls2emb[keys[i]]:
                for e2 in cls2emb[keys[j]]:
                    d = metric(e1.unsqueeze(0), e2.unsqueeze(0))
                    inter.append(d.item())

    intra = np.array(intra)
    inter = np.array(inter)

    # Balance both distributions to same length
    min_len = min(len(intra), len(inter))
    if len(intra) != len(inter):
        idx_intra = np.random.choice(len(intra), min_len, replace=False)
        idx_inter = np.random.choice(len(inter), min_len, replace=False)
        intra = intra[idx_intra]
        inter = inter[idx_inter]

    return intra, inter, xlabel


def separation_metrics(intra, inter):
    """
    Compute separation metrics: means, stds, Fisher ratio, Bhattacharyya coefficient.
    """
    mu_in, mu_it = intra.mean(), inter.mean()
    sd_in, sd_it = intra.std(), inter.std()
    fisher = (mu_it - mu_in)**2 / (sd_it**2 + sd_in**2 + 1e-8)

    # Bhattacharyya coefficient calculation
    term1 = 0.25 * np.log(0.25 * (sd_in**2 / sd_it**2 + sd_it**2 / sd_in**2 + 2))
    term2 = 0.25 * ((mu_in - mu_it)**2 / (sd_in**2 + sd_it**2 + 1e-8))
    bc = np.exp(-(term1 + term2))

    return {
        "mu_intra": mu_in,
        "mu_inter": mu_it,
        "sd_intra": sd_in,
        "sd_inter": sd_it,
        "fisher": fisher,
        "bhattacharyya": bc
    }