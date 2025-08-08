# %% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score
)

# %% Constants and Output Paths
RESULTS_DIR = "../results/cancer_survival"
os.makedirs(RESULTS_DIR, exist_ok=True)

# %% Data Loading
def load_data(metadata_path, expression_path):
    metadata_df = pd.read_csv(metadata_path, index_col=0).drop(['age'], axis=1)
    expression_df = pd.read_csv(expression_path, index_col=0)
    return metadata_df.merge(expression_df, left_index=True, right_index=True)

# %% Target Preparation
def prepare_survival_target(df):
    y_structured = np.array(
        list(zip(df['os_status'].astype(bool), df['os_time'])),
        dtype=[('os_status', '?'), ('os_time', '<f8')]
    )
    X = df.drop(columns=['os_status', 'os_time'])
    return X, y_structured

# %% Train/Test Split
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y['os_status']
    )

# %% Custom Scorer for Survival Models
def c_index_scorer(estimator, X, y):
    surv_pred = estimator.predict(X)
    return concordance_index_censored(y['os_status'], y['os_time'], surv_pred)[0]

# %% Model Training with Hyperparameter Tuning
def tune_model(model_class, param_dist, X_train, y_train, scoring_func, model_name):
    search = RandomizedSearchCV(
        model_class(random_state=42),
        param_distributions=param_dist,
        cv=5,
        scoring=scoring_func,
        random_state=42,
    )
    search.fit(X_train, y_train)
    
    print(f"{model_name} - Best Parameters:", search.best_params_)
    print(f"{model_name} - Best C-index on training data:", search.best_score_)

    # Save best parameters
    param_df = pd.DataFrame([search.best_params_])
    param_df["best_score"] = search.best_score_
    param_df.to_csv(f"{RESULTS_DIR}/{model_name}_best_params.csv", index=False)

    return search.best_estimator_

# %% AUC Plotting
def plot_auc(model_name, y_train, y_test, preds, intervals):
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, preds, intervals)
    plt.figure()
    plt.plot(intervals, auc, marker="o", label=f"{model_name} (mean AUC = {mean_auc:.3f})")
    plt.axhline(mean_auc, linestyle="--", color='gray')
    plt.xlabel("Days from enrollment")
    plt.ylabel("Time-dependent AUC")
    plt.title(f"{model_name} - Time-dependent AUC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_auc_plot.png")
    plt.close()
    return auc, mean_auc

# %% Final AUC Comparison Plot
def compare_auc_plots(rsf_data, gbs_data, intervals):
    rsf_auc, rsf_mean = rsf_data
    gbs_auc, gbs_mean = gbs_data
    plt.figure()
    plt.plot(intervals, rsf_auc, "o-", label=f"RSF (mean AUC = {rsf_mean:.3f})")
    plt.plot(intervals, gbs_auc, "o-", label=f"GBS (mean AUC = {gbs_mean:.3f})")
    plt.xlabel("Days from enrollment")
    plt.ylabel("Time-dependent AUC")
    plt.legend(loc="lower center")
    plt.grid(True)
    plt.title("Model Comparison: RSF vs GBS")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/rsf_vs_gbs_auc_comparison.png")
    plt.close()

# %% Model Evaluation (C-index & Brier Score)
def evaluate_models(rsf, gbs, X_test, y_test, y_train, intervals):
    rsf_surv_prob = np.vstack([fn(intervals) for fn in rsf.predict_survival_function(X_test)])
    gbs_surv_prob = np.vstack([fn(intervals) for fn in gbs.predict_survival_function(X_test)])

    score_cindex = pd.Series(
        [rsf.score(X_test, y_test), gbs.score(X_test, y_test), 0.5],
        index=["RSF", "GBS", "Random"],
        name="C-index",
    )

    score_brier = pd.Series(
        [integrated_brier_score(y_train, y_test, prob, intervals)
         for prob in (rsf_surv_prob, gbs_surv_prob)],
        index=["RSF", "GBS"],
        name="IBS",
    )

    result = pd.concat((score_cindex, score_brier), axis=1).round(3)
    print(result)

    result.to_csv(f"{RESULTS_DIR}/model_evaluation_scores.csv")
    return result

# %% Permutation Importance Plot
def plot_permutation_importance(model, X_test, y_test, model_name):
    result = permutation_importance(model, X_test, y_test, n_repeats=100, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    
    fig, ax = plt.subplots(figsize=(10, len(sorted_idx) // 2))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title(f'Permutation Importance - {model_name}')
    fig.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_permutation_importance.png")
    plt.close()

    # Save raw importances
    importance_df = pd.DataFrame({
        "feature": X_test.columns[sorted_idx],
        "importance_mean": result.importances_mean[sorted_idx],
        "importance_std": result.importances_std[sorted_idx]
    })
    importance_df.to_csv(f"{RESULTS_DIR}/{model_name}_feature_importance.csv", index=False)

# %% Main execution flow
def main():
    # Load and prepare data
    df = load_data('../data/cancer_metadata.csv', '../data/cancer_only_hvg.csv')
    X, y = prepare_survival_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # RSF model training
    rsf_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 3, 5, 7],
        "min_samples_leaf": [2, 3, 5],
        "max_features": ["sqrt", "log2", None]
    }
    rsf_model = tune_model(RandomSurvivalForest, rsf_params, X_train, y_train, c_index_scorer, "RSF")
    rsf_preds = rsf_model.predict(X_test)

    # GBS model training
    gbs_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 3, 5, 7],
        "min_samples_leaf": [2, 3, 5],
        "learning_rate": [0.01, 0.1]
    }
    gbs_model = tune_model(GradientBoostingSurvivalAnalysis, gbs_params, X_train, y_train, c_index_scorer, "GBS")
    gbs_preds = gbs_model.predict(X_test)

    # AUC plotting
    y_min, y_max = np.min(y['os_time']), np.max(y['os_time'])
    intervals = np.arange(y_min, y_max, 100)
    rsf_auc = plot_auc("RSF", y_train, y_test, rsf_preds, intervals)
    gbs_auc = plot_auc("GBS", y_train, y_test, gbs_preds, intervals)
    compare_auc_plots(rsf_auc, gbs_auc, intervals)

    # Model evaluation
    evaluate_models(rsf_model, gbs_model, X_test, y_test, y_train, intervals)

    # Permutation importance
    plot_permutation_importance(rsf_model, X_test, y_test, "RSF")
    plot_permutation_importance(gbs_model, X_test, y_test, "GBS")

# Run the pipeline
if __name__ == "__main__":
    main()
