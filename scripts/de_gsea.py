"""
run_deseq_pipeline.py

A modular pipeline for differential expression analysis using PyDESeq2 and GSEA using GSEApy.

This script:
1. Loads count, metadata, and scaled expression data
2. Runs DESeq2 differential expression analysis
3. Generates and saves:
   - Ranked gene list
   - Volcano plot
   - MA plot
   - Heatmap of top genes
4. Performs pre-ranked GSEA and plots the top enriched pathway
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import gseapy as gp

# -------------------------
# Configuration
# -------------------------
DATA_DIR = "../data"
RESULTS_DIR = "../results"
GSEA_DIR = os.path.join(RESULTS_DIR, "gsea_results")

# Create output directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GSEA_DIR, exist_ok=True)

# -------------------------
# Data Loading
# -------------------------
def load_data():
    """
    Load input data for DE analysis.

    Returns:
        count_df (DataFrame): Raw gene counts
        metadata_df (DataFrame): Sample metadata with design variables
        scaled_log_tmm (DataFrame): Log-scaled normalized expression matrix for plotting
    """
    count_df = pd.read_csv(f'{DATA_DIR}/filtered_raw_expression.csv', index_col=0)
    metadata_df = pd.read_csv(f'{DATA_DIR}/metadata.csv', index_col=0)
    scaled_log_tmm = pd.read_csv(f'{DATA_DIR}/scaled_log_tmm.csv', index_col=0)
    return count_df, metadata_df, scaled_log_tmm

# -------------------------
# DESeq2 Analysis
# -------------------------
def run_deseq2(count_df, metadata_df):
    """
    Run DESeq2 pipeline using PyDESeq2.

    Args:
        count_df (DataFrame): Raw counts (genes x samples)
        metadata_df (DataFrame): Metadata with design factors

    Returns:
        dds (DeseqDataSet): Fitted DESeq2 object
    """
    dds = DeseqDataSet(
        counts=count_df.T,
        metadata=metadata_df,
        design_factors=["sample_type", "age", "gender"]
    )
    dds.deseq2()
    return dds

def run_stats(dds):
    """
    Run DE statistics for contrast (Tumor vs Healthy).

    Args:
        dds (DeseqDataSet): Fitted DESeq2 object

    Returns:
        results (DataFrame): DE results with log2FC, padj, etc.
    """
    ds = DeseqStats(dds, contrast=("sample_type", "Healthy sample", "Primary cancer"))
    ds.summary()
    results = ds.results_df.dropna(subset=["log2FoldChange", "pvalue", "padj"])
    return results

# -------------------------
# Ranking Genes
# -------------------------
def save_ranked_genes(results):
    """
    Save ranked gene list by statistical significance (Wald statistic).

    Args:
        results (DataFrame): DE results

    Returns:
        ranked_series (Series): Ranked gene statistics
    """
    ranked = results.sort_values(by="stat", ascending=False)
    ranked_series = pd.Series(ranked["stat"].values, index=ranked.index)
    ranked_series.to_csv(f"{RESULTS_DIR}/ranked_genes.csv", header=False)
    return ranked_series

# -------------------------
# Plotting Functions
# -------------------------
def plot_volcano(results):
    """
    Create and save a volcano plot of DE results.
    """
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
    plt.savefig(f"{RESULTS_DIR}/volcano_plot.png", dpi=300)
    plt.close()

def plot_ma(results):
    """
    Create and save an MA plot of DE results.
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
    plt.savefig(f"{RESULTS_DIR}/ma_plot.png", dpi=300)
    plt.close()

def plot_heatmap(results, scaled_log_tmm, metadata_df):
    """
    Generate a heatmap of the top 20 most significant DE genes.

    Args:
        results (DataFrame): DE results
        scaled_log_tmm (DataFrame): Scaled log expression matrix
        metadata_df (DataFrame): Sample metadata
    """
    top_genes = results.sort_values("padj").head(20).index
    heatmap_data = scaled_log_tmm[top_genes]
    heatmap_data.index = metadata_df.loc[heatmap_data.index, 'sample_type']
    sns.clustermap(heatmap_data, cmap="vlag", figsize=(10, 8), col_cluster=False)
    plt.savefig(f"{RESULTS_DIR}/heatmap_top20_genes.png", dpi=300)
    plt.close()

# -------------------------
# GSEA with GSEApy
# -------------------------
def run_gsea(ranked_genes):
    """
    Run GSEA on a ranked gene list and save plots/results.

    Args:
        ranked_genes (Series): Ranked gene statistics

    Returns:
        res2d (DataFrame): GSEA summary table
    """
    pre_res = gp.prerank(
        rnk=ranked_genes,
        gene_sets='GO_Biological_Process_2021',
        permutation_num=100,
        outdir=GSEA_DIR,
        no_plot=False,
        seed=42
    )

    # Plot the top enriched term
    top_term = pre_res.res2d['Term'].iloc[0]
    gsea_result = pre_res.results[top_term]

    gp.plot.gseaplot(
        hits=gsea_result['hits'],
        RES=gsea_result['RES'],
        term=top_term,
        rank_metric=pre_res.ranking,
        nes=gsea_result['nes'],
        pval=gsea_result['pval'],
        fdr=gsea_result['fdr']
    )

    plt.savefig(f"{GSEA_DIR}/top_term_enrichment.png", dpi=300)
    plt.close()

    # Save the summary table
    pre_res.res2d.to_csv(f"{GSEA_DIR}/gsea_summary.csv")
    return pre_res.res2d

# -------------------------
# Main Pipeline Function
# -------------------------
def main():
    """
    Main function to run the DE + GSEA pipeline.
    """
    print("🔹 Loading data...")
    count_df, metadata_df, scaled_log_tmm = load_data()

    print("🔹 Running DESeq2 analysis...")
    dds = run_deseq2(count_df, metadata_df)
    results = run_stats(dds)

    print("🔹 Saving ranked genes...")
    ranked_genes = save_ranked_genes(results)

    print("🔹 Generating plots...")
    plot_volcano(results)
    plot_ma(results)
    plot_heatmap(results, scaled_log_tmm, metadata_df)

    print("🔹 Running GSEA...")
    gsea_results = run_gsea(ranked_genes)

    print("✅ Pipeline completed successfully!")
    print("🔬 Top enriched pathways:\n", gsea_results.head())

# # -------------------------
# # Run Script
# # -------------------------
if __name__ == "__main__":
    main()
