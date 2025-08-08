import pandas as pd
import numpy as np
import json
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, mean_squared_error


def align_expression_and_metadata(expression_df, metadata):
    """
    Aligns the expression matrix and metadata DataFrame on shared sample indices.

    Parameters:
        hvg_matrix (pd.DataFrame): Gene expression matrix (samples x genes).
        cancer_df (pd.DataFrame): Metadata DataFrame (samples x variables).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Aligned metadata and expression matrix.
    """
    common_idx = metadata.index.intersection(expression_df.index)
    return metadata.loc[common_idx].copy(), expression_df.loc[common_idx].copy()


def encode_categorical_columns(metadata_df, extra_cats=['gender']):
    """
    Encodes categorical variables using ordinal encoding, including optional fixed columns.

    Parameters:
        metadata (pd.DataFrame): Metadata DataFrame.
        extra_cats (List[str]): Additional categorical columns to encode (default = ['gender']).

    Returns:
        Tuple:
            - pd.DataFrame: Updated DataFrame with encoded columns.
            - List[str]: Names of numeric columns with missing values.
            - List[str]: Names of categorical columns with missing values.
            - List[str]: Names of new encoded categorical columns.
            - List[str]: All columns to impute (numeric + encoded categorical).
            - Dict: Fitted encoders for each categorical column.
    """
    missing_cols = metadata_df.columns[metadata_df.isnull().any()].tolist()
    cat_cols = metadata_df[missing_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = metadata_df[missing_cols].select_dtypes(include=['number']).columns.tolist()

    encoders = {}
    all_cats = list(set(cat_cols + extra_cats))
    for col in all_cats:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        reshaped = metadata_df[[col]].astype(str)
        metadata_df[col + '_enc'] = enc.fit_transform(reshaped)
        encoders[col] = enc

    encoded_cat_cols = [col + '_enc' for col in cat_cols]
    all_impute_cols = num_cols + encoded_cat_cols + ['gender_enc']
    return metadata_df, num_cols, cat_cols, encoded_cat_cols, all_impute_cols, encoders


def mask_for_validation(combined_data, all_impute_cols, fraction=0.1, seed=42):
    """
    Randomly masks a fraction of observed values for evaluation of imputation error.

    Parameters:
        combined_data (pd.DataFrame): Combined expression + target matrix.
        all_impute_cols (List[str]): Columns to mask.
        fraction (float): Fraction of observed values to mask.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Masked data and true targets before masking.
    """
    np.random.seed(seed)
    masked_data = combined_data.copy()
    targets = combined_data[all_impute_cols]
    impute_mask = targets.notna()

    for col in all_impute_cols:
        observed_indices = impute_mask[col][impute_mask[col]].index
        n_mask = int(len(observed_indices) * fraction)
        if n_mask == 0:
            continue
        mask_indices = np.random.choice(observed_indices, n_mask, replace=False)
        masked_data.loc[mask_indices, col] = np.nan

    return masked_data, targets.copy()


def optimize_k(masked_data, combined_data, true_targets, all_impute_cols, encoded_cat_cols, k_values=[3, 5]):
    """
    Finds the best number of neighbors (k) for KNN imputation based on reconstruction error.

    Parameters:
        masked_data (pd.DataFrame): Input data with masked values.
        combined_data (pd.DataFrame): Full combined dataset (expression + targets).
        true_targets (pd.DataFrame): Ground truth (unmasked) target values.
        all_impute_cols (List[str]): Columns to impute.
        encoded_cat_cols (List[str]): Encoded categorical columns.
        k_values (List[int]): List of k values to try.

    Returns:
        int: Optimal k value with minimum error.
    """
    errors = []
    for k in k_values:
        imputer = KNNImputer(n_neighbors=k)
        imputed_array = imputer.fit_transform(masked_data)
        imputed_df = pd.DataFrame(imputed_array, columns=combined_data.columns, index=combined_data.index)
        imputed_targets = imputed_df[all_impute_cols]

        total_error = 0
        for col in all_impute_cols:
            mask_col = true_targets[col].notna()
            true_vals = true_targets.loc[mask_col, col]
            imputed_vals = imputed_targets.loc[mask_col, col]

            if col in encoded_cat_cols or col == 'gender_enc':
                acc = accuracy_score(true_vals.round().astype(int), imputed_vals.round().astype(int))
                error = 1 - acc
            else:
                error = np.sqrt(mean_squared_error(true_vals, imputed_vals))

            total_error += error

        errors.append((k, total_error))
        print(f"k={k}: total error = {total_error:.4f}")

    return min(errors, key=lambda x: x[1])[0]


def impute_final(complete_rows, combined_data, best_k):
    """
    Performs final KNN imputation using the best k value.

    Parameters:
        complete_rows (pd.DataFrame): Subset of combined data with no missing values.
        combined_data (pd.DataFrame): Full dataset to impute.
        best_k (int): Optimal number of neighbors.

    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    imputer = KNNImputer(n_neighbors=best_k)
    imputer.fit(complete_rows)
    imputed_array = imputer.transform(combined_data)
    return pd.DataFrame(imputed_array, columns=combined_data.columns, index=combined_data.index)


def restore_metadata(cancer_df, final_targets, num_cols, cat_cols, encoded_cat_cols):
    """
    Inserts imputed values back into the metadata DataFrame.

    Parameters:
        cancer_df (pd.DataFrame): Original metadata.
        final_targets (pd.DataFrame): Imputed values.
        num_cols (List[str]): Numeric columns.
        cat_cols (List[str]): Categorical columns.
        encoded_cat_cols (List[str]): Encoded categorical columns.

    Returns:
        pd.DataFrame: Metadata with imputed values restored.
    """
    for col in num_cols:
        cancer_df[col] = final_targets[col]
    for col in cat_cols + ['gender']:
        col_enc = col + '_enc'
        cancer_df[col] = final_targets[col_enc]
    cancer_df.drop(columns=encoded_cat_cols + ['gender_enc'], inplace=True)
    return cancer_df


def save_files(encoders, metadata_df, directory="../data"):
    """
    Saves the mapping from encoded integers back to categorical labels and mputed metadta.

    Parameters:
        encoders (Dict): Fitted OrdinalEncoders.
        metadta_df: imputed metadata
        filename (str): Output JSON file.
    """
    decoding = {}
    for col, encoder in encoders.items():
        values = encoder.categories_[0]
        decoding[col] = {int(i): str(v) for i, v in enumerate(values) if str(v)!='nan'}
    with open(f"{directory}/category_decoding.json", "w") as f:
        json.dump(decoding, f, indent=4)
    metadata_df.to_csv(f"{directory}/imputed_cancer_metadata.csv")

def knn_expression_impute_from_matrix(hvg_matrix, cancer_df, k_values=[3, 5], mask_fraction=0.1):
    """
    Main pipeline to impute missing metadata using KNN based on expression matrix.

    Parameters:
        hvg_matrix (pd.DataFrame): Gene expression matrix (samples x genes).
        cancer_df (pd.DataFrame): Metadata DataFrame with missing values.
        k_values (List[int]): List of candidate k values for optimization.
        mask_fraction (float): Fraction of known data to mask for validation.

    Returns:
        pd.DataFrame: Metadata DataFrame with imputed values.
    """
    print("🔁 Aligning expression and metadata ...")
    cancer_df, hvg_matrix = align_expression_and_metadata(hvg_matrix, cancer_df)

    print("🧬 Encoding categorical columns ...")
    cancer_df, num_cols, cat_cols, encoded_cat_cols, all_impute_cols, encoders = encode_categorical_columns(cancer_df)

    targets = cancer_df[all_impute_cols].copy()
    combined_data = pd.concat([hvg_matrix, targets], axis=1)
    complete_rows = combined_data.dropna()

    print("🎭 Creating mask for validation ...")
    masked_data, true_targets = mask_for_validation(combined_data, all_impute_cols, fraction=mask_fraction)

    print("🔍 Optimizing K value ...")
    best_k = optimize_k(masked_data, combined_data, true_targets, all_impute_cols, encoded_cat_cols, k_values)
    print(f"\n✅ Selected best k = {best_k}")

    print("🧩 Performing final KNN imputation ...")
    final_imputed_df = impute_final(complete_rows, combined_data, best_k)
    final_targets = final_imputed_df[all_impute_cols]

    print("📝 Writing imputed data back to metadata ...")
    cancer_df = restore_metadata(cancer_df, final_targets, num_cols, cat_cols, encoded_cat_cols)

    print("💾 Saving encoding mappings and new metaadta ...")
    save_files(encoders,cancer_df)

    print("\n✅ KNN imputation using expression matrix completed.")
    return cancer_df

