# transcript/plotting_utils.py

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from typing import Union, List


# Suppress warnings globally
warnings.filterwarnings("ignore")


def plot_categorical_distributions(df, categorical_cols, save_dir="../results/metadata", title="Categorical Distributions"):
    """
    Plot bar charts showing the distribution of categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame DataFrame containing the categorical columns to be plotted.

    categorical_cols : list of str List of column names in `df` that are categorical.

    title : str, optional Overall title for the figure. Default is "Categorical Variable Distributions".

    Returns Displays the plots directly using matplotlib.
    """
    os.makedirs(save_dir, exist_ok=True)
    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, palette="Set2", order=df[col].dropna().value_counts().index)
        plt.title(f"{col.replace('_', ' ').title()} Distribution")
        plt.ylabel("Count")
        plt.xlabel(col.replace('_', ' ').title())
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{col}_distribution.png"))
        plt.show()


def plot_continuous_distributions(df, continuous_cols, save_dir="../results/metadata", bins=30, color="lightcoral"):
    """
    Plot histograms with KDE overlays for continuous survival-related variables.

    Parameters
    df : pandas.DataFrame DataFrame containing the continuous columns to be plotted.

    continuous_cols : list of str List of column names in `df` that are continuous variables.

    bins : int, optional Number of bins to use in the histograms. Default is 30.

    color : str, optional Color used for histogram bars. Default is 'lightcoral'.

    Returns Displays the plots directly using matplotlib.
    """
    os.makedirs(save_dir, exist_ok=True)
    for col in continuous_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=bins, color=color)
        plt.title(f"{col.replace('_', ' ').upper()} Distribution")
        plt.xlabel(col.replace('_', ' ').upper())
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{col}_distribution.png"))
        plt.show()


def plot_reductions(adata, save_dir="../results/dimensionality_reduction", color_by: Union[str, List[str]]):
    """
    Plot PCA, t-SNE, and UMAP projections of an AnnData object colored by one or more metadata variables.

    Parameters
    adata : AnnData Annotated data matrix (e.g., cancer_adata) with PCA, t-SNE, and UMAP already computed.
        
    color_by : str or list of str Column name(s) in `adata.obs` to use for coloring the plots. Can be a single variable or a list of variables.

    Returns Displays PCA, t-SNE, and UMAP plots.
    """
    os.makedirs("results/dimensionality_reduction", exist_ok=True)

    if isinstance(color_by, str):
        color_by = [color_by]

    for var in color_by:
        for method in ['pca', 'tsne', 'umap']:
            filename = f"{save_dir}/{method}_{var}.png"
            plot_func = getattr(sc.pl, method)
            plot_func(adata, color=var, title=f"{method.upper()} - colored by {var}", show=True, save=False)
            fig = plt.gcf()
            fig.savefig(filename, bbox_inches='tight')