import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, \
    confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_history(history, metrics_names, **kwargs):
    """
    Plots metrics monitored during training.
    N.B. This function does not perform any check on the parameters!

    Parameters
    ----------
    history : dict
        Dictionary with the metrics, lr and loss.
    metrics_name : list
        Name of metrics to plot. 
        Validation metrics must have same name with "val_" in front.
    **kwargs
        Addition kwargs passed to plt.subplots(). 
    """
    fig, axs = plt.subplots(**kwargs)

    for m,ax in zip(["lr", "loss"] + metrics_names, axs.flatten()):
        ax.plot(history[m], c="C0") # train
        if m!="lr":
            ax.plot(history[f"val_{m}"], color="C1") # validation
        ax.set_title(m)

    # fake common legend
    axs[0,0].plot([], c="C0", label="Train")
    axs[0,0].plot([], c="C1", label="Validation")
    fig.legend(ncol=2, loc="lower center")
    #fig.tight_layout()
    return

def print_metrics(y_pred, y_true, p=.5, cm_labels="auto"):
    """
    Print the following metrics: accuracy, balanced accuracy, precision, recall, f1.
    If the model is able to predict probabilities, also auc is calculated.
    Moreover, the confusion matrix is plotted.

    Parameters
    ----------
    y_pred : array like
        Model predictions (probabilities).
    y_true : array like
        Target values.
    cm_labels : list [default="auto"]
        Optional labels to be used in the confusion matrix.

    Returns
    -------
    metrics : dict
        Dictionary with the calculated metrics.

    """
    y_pred = np.array(y_pred)
    y_class = (y_pred > p).astype(int)

    metrics = {
        "Accuracy"    : accuracy_score(y_true, y_class),
        "Bal Accuracy": balanced_accuracy_score(y_true, y_class),
        "Precision"   : precision_score(y_true, y_class),
        "Recall"      : recall_score(y_true, y_class),
        "f1"          : f1_score(y_true, y_class),
        "AUC"         : roc_auc_score(y_true, y_pred)
    }

    for k,v in metrics.items():
        print("{:12s} = {}".format(k,v))  
    print("\n")
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_class, normalize="true")
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, ax=ax, square=True, vmin=0, vmax=1, annot=True, 
                linewidths=.05, fmt=".2f", cbar_kws={"shrink":.8}, 
                xticklabels=cm_labels, yticklabels=cm_labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    metrics["cm"] = cm

    return metrics