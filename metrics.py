import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_conf_mat(y_true, y_pred, name, time):

    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=unique_labels,
        yticklabels=unique_labels
    )
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.savefig(f'metrics/{name}_{time}_confusion_matrix.png')
    plt.clf()

def save_classification_report(y_true, y_pred, name, time):

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report_dict).T
    df = df.round(2)
    
    # 2) Set overall style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold"
    })
    
    # 3) Create figure & hide axes
    n_rows = df.shape[0]
    fig, ax = plt.subplots(figsize=(8, n_rows * 0.5 + 1))
    ax.axis('off')
    
    # 4) Build table
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # 5) Tweak font & scaling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # 6) Style every cell
    for (row, col), cell in table.get_celld().items():
        # all cell borders
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
        
        # header row
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        
        # row‑label column
        elif col == -1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f1f1f2')
        
        # zebra striping for data rows
        elif row % 2 == 0:
            cell.set_facecolor('#f9f9f9')
    
    plt.savefig(f'metrics/{name}_{time}_classification_report.png', bbox_inches='tight')
    plt.close(fig)