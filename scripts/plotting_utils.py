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


def plot_reductions(adata, color_by: Union[str, List[str]], save_dir="../results/dimensionality_reduction"):
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
            #save figure
            plot_func = getattr(sc.pl, method)
            plot_func(
                adata,
                color=var,
                title=f"{method.upper()} - colored by {var}",
                show=False
            )
            filename = f"../results/dimensionality_reduction/{method}_{var}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            # show figure
            plot_func(
                adata,
                color=var,
                title=f"{method.upper()} - colored by {var}",
                show=True
            )


def plot_time_distribution_by_outcome(metadata_df: pd.DataFrame, 
                                      time_col: str = 'os_time', 
                                      status_col: str = 'os_status',
                                      figsize: tuple = (10, 5),
                                      palette_name: str = 'Set2') -> None:
    """
    Plot a Kernel Density Estimate (KDE) of time-to-event data grouped by outcome status.
    
    This function visualizes the distribution of survival/censoring times for different 
    outcome groups using KDE plots and a rug plot for individual data points.

    Parameters:
    -----------
    metadata_df : pd.DataFrame
        DataFrame containing survival metadata (e.g., time and event status).
        
    time_col : str, default='os_time'
        Column name representing time-to-event or follow-up time.
        
    status_col : str, default='os_status'
        Column name representing outcome status (e.g., 'Dead', 'Alive', 'Censored').

    figsize : tuple, default=(10, 5)
        Size of the matplotlib figure.
        
    palette_name : str, default='Set2'
        Seaborn color palette name for distinguishing outcome groups.

    Returns:
    --------
    None
        Displays a KDE plot with density estimates and a rug plot overlay.
    """

    plt.figure(figsize=figsize)

    # Generate color palette
    unique_statuses = metadata_df[status_col].unique()
    palette = sns.color_palette(palette_name, len(unique_statuses))

    # KDE plots for each outcome
    for i, status in enumerate(unique_statuses):
        subset = metadata_df[metadata_df[status_col] == status][time_col]
        sns.kdeplot(subset, fill=True, alpha=0.5, linewidth=2, label=status, color=palette[i])

    # Rug plot for individual time points
    sns.rugplot(data=metadata_df, x=time_col, hue=status_col, palette=palette, height=0.03, alpha=0.7)

    # Plot styling
    plt.title('Time Distribution by Outcome', fontsize=18, weight='bold')
    plt.xlabel('Time to Event or Censoring', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(title='Outcome', title_fontsize=13, fontsize=11, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_volcano(results: pd.DataFrame, output_path: str):
    """
    Create and save a volcano plot of DE results.

    Args:
        results (pd.DataFrame): DE results.
        output_path (str): File path to save plot image.
    """
    results = results.copy()
    results['neg_log10_padj'] = -np.log10(results['padj'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=results,
        x='log2FoldChange',
        y='neg_log10_padj',
        hue=results['padj'] < 0.05,
        palette={True: 'red', False: 'grey'},
        alpha=0.6
    )
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('Volcano Plot of Differential Expression')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 Adjusted p-value')
    plt.legend(title='Significant (padj < 0.05)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_ma(results: pd.DataFrame, output_path: str):
    """
    Create and save an MA plot of DE results.

    Args:
        results (pd.DataFrame): DE results.
        output_path (str): File path to save plot image.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=results,
        x='baseMean',
        y='log2FoldChange',
        hue=results['padj'] < 0.05,
        palette={True: 'red', False: 'grey'},
        alpha=0.6
    )
    plt.xscale('log')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('MA Plot of Differential Expression')
    plt.xlabel('Mean of Normalized Counts (log scale)')
    plt.ylabel('Log2 Fold Change')
    plt.legend(title='Significant (padj < 0.05)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_heatmap(results: pd.DataFrame, scaled_expression: pd.DataFrame, metadata_df: pd.DataFrame,
                 output_path: str, top_n: int = 20, cluster_cols: bool = False):
    """
    Generate and save a heatmap of top N DE genes.

    Args:
        results (pd.DataFrame): DE results.
        scaled_expression (pd.DataFrame): Scaled expression matrix (samples x genes).
        metadata_df (pd.DataFrame): Sample metadata (index matches samples in scaled_expression).
        output_path (str): File path to save heatmap image.
        top_n (int): Number of top genes to include.
        cluster_cols (bool): Whether to cluster columns in heatmap.
    """
    top_genes = results.sort_values("padj").head(top_n).index
    heatmap_data = scaled_expression[top_genes]

    # Annotate samples by sample_type if available
    if 'sample_type' in metadata_df.columns:
        row_colors = metadata_df.loc[heatmap_data.index, 'sample_type'].map({
            v: sns.color_palette("Set2")[i] for i, v in enumerate(metadata_df['sample_type'].unique())
        })
    else:
        row_colors = None

    sns.clustermap(
        heatmap_data,
        cmap="vlag",
        figsize=(10, 8),
        col_cluster=cluster_cols,
        row_colors=row_colors
    )
    plt.savefig(output_path, dpi=300)
    plt.close()