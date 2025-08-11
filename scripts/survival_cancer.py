# cancer_survival_pipeline.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score

# -----------------------------------------------------------
# 1. Data Loading and Filtering
# -----------------------------------------------------------
def load_and_filter_data(metadata_path: str, expression_path: str, min_time: int = 180, max_time = 4000):
    """
    Load and filter metadata, expression, and GSEA results.
    
    Args:
        metadata_path (str): Path to metadata CSV.
        expression_path (str): Path to expression CSV.
        min_time (int): Minimum survival time threshold in days.
        
    Returns:
        tuple: (metadata_df, expression_df)
    """
    metadata_df = pd.read_csv(metadata_path, index_col=0)
    expression_df = pd.read_csv(expression_path, index_col=0)
    metadata_df = metadata_df[(metadata_df['os_time'] >= min_time)&(metadata_df['os_time'] <= max_time)]
    expression_df = expression_df.loc[metadata_df.index]

    return metadata_df, expression_df


# -----------------------------------------------------------
# 2. Data Preparation
# -----------------------------------------------------------
def prepare_survival_data(metadata_df: pd.DataFrame, expression_df: pd.DataFrame, k_features: int = 55):
    """
    Prepare train/test datasets for survival analysis.
    
    Args:
        metadata_df (pd.DataFrame): Clinical metadata with os_time, os_status.
        expression_df (pd.DataFrame): Gene expression data.
        k_features (int): Number of top features to select.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    metadata_df["os_status"] = metadata_df["os_status"].map({'Death': True, 'Censored': False})
    metadata_df["os_time"] = metadata_df["os_time"].astype(float)

    y = np.array(list(zip(metadata_df["os_status"], metadata_df["os_time"])),
                 dtype=[('event', '?'), ('time', '<f8')])
    
    clinical_df = metadata_df.drop(columns=['os_status', 'os_time'])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        expression_df, y, test_size=0.2, random_state=42, stratify=y['event']
    )

    bestk = SelectKBest(score_func=f_classif, k=k_features)
    bestk.fit(X_train_raw, y_train)
    selected_features = X_train_raw.columns[bestk.get_support()]

    X_train = pd.DataFrame(bestk.transform(X_train_raw), columns=selected_features, index=X_train_raw.index)
    X_test = pd.DataFrame(bestk.transform(X_test_raw), columns=selected_features, index=X_test_raw.index)

    X_train = X_train.merge(clinical_df, left_index=True, right_index=True)
    X_test = X_test.merge(clinical_df, left_index=True, right_index=True)

    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X_train.select_dtypes(include=['number']).columns

    # Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        mask_train = X_train[col].notnull()
        X_train.loc[mask_train, col] = le.fit_transform(X_train.loc[mask_train, col].astype(str))
        mask_test = X_test[col].notnull()
        X_test.loc[mask_test, col] = le.transform(X_test.loc[mask_test, col].astype(str))
        label_encoders[col] = le

    # Scaling
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Imputation
    imputer = IterativeImputer()
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------
# 3. Model Training
# -----------------------------------------------------------
def c_index_scorer(estimator, X, y):
    """Custom scorer for RandomizedSearchCV."""
    surv_pred = estimator.predict(X)
    return concordance_index_censored(y['event'], y['time'], surv_pred)[0]

def train_model(X_train, y_train, model_type: str):
    """
    Train a survival model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Survival target array.
        model_type (str): 'RSF' or 'GBS'.
        
    Returns:
        tuple: (best_model, best_params)
    """
    if model_type == 'RSF':
        model = RandomSurvivalForest(random_state=42)
        param_dist = {
            "n_estimators": [50, 75, 100, 200, 300],
            "max_depth": [2, 3, 5, 10, None],
            "min_samples_split": [2, 3, 5, 7],
            "min_samples_leaf": [2, 3, 5],
            "max_features": ["sqrt", "log2"]
        }
    else:
        model = GradientBoostingSurvivalAnalysis(random_state=42)
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 3, 5, 7],
            "min_samples_leaf": [2, 3, 5],
            "learning_rate": [0.01, 0.1]
        }

    search = RandomizedSearchCV(model, param_distributions=param_dist, cv=3,
                                 scoring=c_index_scorer, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# -----------------------------------------------------------
# 4. Plotting Functions
# -----------------------------------------------------------
def plot_auc(model_name, y_train, y_test, preds, intervals, output_dir):
    """
    Plot and save time-dependent AUC curve.
    """
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, preds, intervals)
    plt.figure()
    plt.plot(intervals, auc, marker="o", label=f"{model_name} (mean AUC = {mean_auc:.3f})")
    plt.axhline(mean_auc, linestyle="--", color='gray')
    plt.xlabel("Years from enrollment")
    plt.ylabel("Time-dependent AUC")
    plt.title(f"{model_name} - Time-dependent AUC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_auc_plot.png"))
    plt.close()
    return mean_auc

def plot_permutation_importance(model, X_test, y_test, model_name, output_dir, n_repeats=100):
    """
    Compute, show, and save permutation importance plot for the top 10 features
    by absolute mean importance, using the original signed values for display.
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42)
    
    # Select top 10 by absolute importance
    abs_importances = np.abs(result.importances_mean)
    top10_idx = np.argsort(abs_importances)[-10:]
    
    # Sort top 10 by signed mean importance for plotting
    top10_idx = top10_idx[np.argsort(result.importances_mean[top10_idx])]
    
    # Plot
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[top10_idx].T,
               vert=False,
               labels=X_test.columns[top10_idx])
    ax.set_title(f'Permutation Importance - Top 10 Absolute ({model_name})')
    fig.tight_layout()
    
    # Show before saving
    plt.show()
    
    # Save
    save_path = os.path.join(output_dir, f"{model_name}_perm_importance.png")
    fig.savefig(save_path)
    plt.close(fig)


# -----------------------------------------------------------
# 5. Main Pipeline
# -----------------------------------------------------------
def run_pipeline(metadata_path="../data/cancer_metadata.csv", 
         expression_path="../data/cancer_tmm_log.csv", 
         output_dir="../results/cancer_survival",k_features = 50):

    os.makedirs(output_dir, exist_ok=True)

    # Load & prepare data
    metadata_df, expression_df = load_and_filter_data(metadata_path, expression_path)
    X_train, X_test, y_train, y_test = prepare_survival_data(metadata_df, expression_df,k_features)

    # Train models
    models = {}
    for model_type in ['RSF', 'GBS']:
        model, params = train_model(X_train, y_train, model_type)
        models[model_type] = (model, params)

    # Evaluate models
    results = []
    y_min, y_max = np.min(y_test['time']), np.max(y_test['time'])
    intervals = np.arange(y_min+90, y_max, 365)

    for name, (model, _) in models.items():
        preds = model.predict(X_test)
        mean_auc = plot_auc(name, y_train, y_test, preds, intervals, output_dir)

        surv_prob = np.vstack([fn(intervals) for fn in model.predict_survival_function(X_test)])
        c_index = model.score(X_test, y_test)
        ibs = integrated_brier_score(y_train, y_test, surv_prob, intervals)

        results.append({"Model": name, "C-index": c_index, "IBS": ibs, "Mean AUC": mean_auc})
        plot_permutation_importance(model, X_test, y_test, name, output_dir)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "model_scores.csv"), index=False)
    print(results_df)

