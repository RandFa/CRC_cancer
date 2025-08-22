"""
run_deseq_pipeline.py

A modular pipeline for differential expression analysis using PyDESeq2 and GSEA using GSEApy.

This script contains reusable functions that:
- Run DESeq2 differential expression analysis
- Generate and save ranked gene lists and common plots (volcano, MA, heatmap)
- Perform preranked GSEA and plot the top enriched pathway

All input/output paths and data are provided via function arguments for flexibility.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import gseapy as gp
import plotting_utils as mplt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -------------------------
# DESeq2 Analysis Functions
# -------------------------
def run_deseq2(count_df: pd.DataFrame, metadata_df: pd.DataFrame, design_factors: list) -> DeseqDataSet:
    """
    Run DESeq2 pipeline using PyDESeq2.

    Args:
        count_df (pd.DataFrame): Raw counts matrix (genes x samples).
        metadata_df (pd.DataFrame): Metadata dataframe with design factors.
        design_factors (list): List of column names in metadata to use as design factors.

    Returns:
        DeseqDataSet: Fitted DESeq2 dataset object.
    """
    count_df = count_df.loc[:, count_df.sum(axis=0) > 0]
    dds = DeseqDataSet(
        counts=count_df,
        metadata=metadata_df,
        design_factors=design_factors
    )
    dds.deseq2()
    return dds

def run_deseq_stats(dds: DeseqDataSet, contrast: tuple) -> pd.DataFrame:
    """
    Compute DE statistics for a specific contrast.

    Args:
        dds (DeseqDataSet): Fitted DESeq2 dataset.
        contrast (tuple): Contrast tuple in form (factor, condition1, condition2).

    Returns:
        pd.DataFrame: Differential expression results with log2FC, p-values, etc.
    """
    ds = DeseqStats(dds, contrast=contrast)
    ds.summary()
    results = ds.results_df.dropna(subset=["log2FoldChange", "pvalue", "padj"])
    return results

# -------------------------
# Ranked Genes Handling
# -------------------------
def save_ranked_genes(results: pd.DataFrame, output_path: str) -> pd.Series:
    """
    Save genes ranked by DESeq2 Wald statistic for GSEA.

    Args:
        results (pd.DataFrame): DE results.
        output_path (str): Path to save ranked gene list (csv).

    Returns:
        pd.Series: Ranked gene statistics (statistic as values, gene names as index).
    """
    ranked = results.sort_values(by="stat", ascending=False)
    ranked_series = pd.Series(ranked["stat"].values, index=ranked.index)
    ranked_series.to_csv(output_path, header=False)
    return ranked_series



# -------------------------
# GSEA Functions
# -------------------------
def run_gsea_prerank(ranked_genes: pd.Series, outdir: str, gene_sets: str = 'GO_Biological_Process_2021',
                     permutation_num: int = 100, seed: int = 42, plot_top_term: bool = True) -> pd.DataFrame:
    """
    Run preranked GSEA using gseapy.

    Args:
        ranked_genes (pd.Series): Ranked gene list (statistic values with gene names as index).
        outdir (str): Directory to save GSEA output.
        gene_sets (str): Gene set database to use (e.g., 'Hallmark', 'KEGG_2016').
        permutation_num (int): Number of permutations for significance testing.
        seed (int): Random seed for reproducibility.
        plot_top_term (bool): Whether to plot the top enriched term.

    Returns:
        pd.DataFrame: GSEA results summary table.
    """
    os.makedirs(outdir, exist_ok=True)

    pre_res = gp.prerank(
        rnk=ranked_genes,
        gene_sets=gene_sets,
        permutation_num=permutation_num,
        outdir=outdir,
        no_plot=not plot_top_term,
        seed=seed
    )

    if plot_top_term:
        top_term = pre_res.res2d['Term'][0]
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
        plt.savefig(os.path.join(outdir, "top_term_enrichment.png"), dpi=300)
        plt.close()

    pre_res.res2d.to_csv(os.path.join(outdir, "gsea_summary.csv"))
    return pre_res.res2d

# -------------------------
# Example Pipeline Runner
# -------------------------
def run_pipeline(count_df: pd.DataFrame, metadata_df: pd.DataFrame, scaled_expression: pd.DataFrame,
                 design_factors: list, contrast: tuple, results_dir: str, gene_sets: str = 'GO_Biological_Process_2021'):
    """
    Run the full DESeq2 + GSEA pipeline given input data.

    Args:
        count_df (pd.DataFrame): Raw counts (genes x samples).
        metadata_df (pd.DataFrame): Sample metadata.
        scaled_expression (pd.DataFrame): Normalized & scaled expression for plotting.
        design_factors (list): Design factors for DESeq2.
        contrast (tuple): Contrast for DESeq2 stats (factor, cond1, cond2).
        results_dir (str): Directory to save all outputs.
        gene_sets (str): Gene set database to use (e.g., 'Hallmark', 'KEGG_2016').

    Returns:
        None
    """
    os.makedirs(results_dir, exist_ok=True)
    gsea_dir = os.path.join(results_dir, "gsea_results")
    os.makedirs(gsea_dir, exist_ok=True)

    print("🔹 Running DESeq2...")
    dds = run_deseq2(count_df, metadata_df, design_factors)
    results = run_deseq_stats(dds, contrast)

    print("🔹 Saving ranked genes...")
    ranked_path = os.path.join(results_dir, "ranked_genes.csv")
    ranked_genes = save_ranked_genes(results, ranked_path)

    print("🔹 Plotting volcano plot...")
    mplt.plot_volcano(results, os.path.join(results_dir, "volcano_plot.png"))

    print("🔹 Plotting MA plot...")
    mplt.plot_ma(results, os.path.join(results_dir, "ma_plot.png"))

    print("🔹 Plotting heatmap of top genes...")
    mplt.plot_heatmap(results, scaled_expression, metadata_df, os.path.join(results_dir, "heatmap_top20_genes.png"))

    print("🔹 Running GSEA prerank...")
    gsea_results = run_gsea_prerank(ranked_genes, gsea_dir, gene_sets, permutation_num=100)

    print("✅ Pipeline complete.")
    print("Top enriched pathways:")
    print(gsea_results.head())

