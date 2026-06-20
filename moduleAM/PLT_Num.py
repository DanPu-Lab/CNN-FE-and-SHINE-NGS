import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import interpolate, stats
from scipy.interpolate import interp1d
from scipy.stats import chi2_contingency, ttest_ind, gaussian_kde
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    jaccard_score, matthews_corrcoef, auc, average_precision_score, precision_recall_fscore_support, \
    precision_recall_curve, roc_curve
import seaborn as sns
from matplotlib import font_manager as fm

def plot_pie_in_density(scores, t):
    # Define threshold for pathogenic vs. non-pathogenic variants
    threshold = t  # Adjust based on the model

    # Create binary labels based on the threshold
    pathogenic = scores >= threshold
    non_pathogenic = scores < threshold

    # Count pathogenic and non-pathogenic variants
    pathogenic_count = np.sum(pathogenic)
    non_pathogenic_count = np.sum(non_pathogenic)

    # Create the main figure for score distribution
    fig, ax = plt.subplots(figsize=(2.2, 1.65), dpi=600)

    # Plot density (score distribution) using seaborn #FF7F0E #75C8AE
    sns.kdeplot(scores, fill=True, color='#FF7F0E', ax=ax, alpha=0.6)
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=1)  # Vertical line at threshold
    ax.set_xlim(0, 1)
    ax.set_xlabel('', fontsize=8)
    ax.set_ylabel('', fontsize=8)
    # Add inset pie chart inside the density plot
    ax_inset = inset_axes(ax, width="46%", height="65%", loc='upper left', borderpad=1)
    custom_colors = ['#FF7F0E', '#98A9D0']
    ax_inset.pie([pathogenic_count, non_pathogenic_count], startangle=90, colors=custom_colors)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    # plt.savefig("../img/img6/RP1L1_ClinPred.pdf")
    plt.close()

file = "Resnet34_ION_INDEL2.csv"
df = pd.read_csv(file)
score = df["Positive_Probability"]
t = df["F1_Optimal_Threshold"][1]
plot_pie_in_density(score, t)