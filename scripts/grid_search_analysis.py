#!/usr/bin/env python3
# %%
"""Grid search analysis script for processing experiment results."""

import json
import pandas as pd
from pathlib import Path
from typing import Any
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_experiment_data(output_dir: Path) -> pd.DataFrame:
    """Load all experiment_info.json files and create a dataframe with flattened parameters."""
    experiments = []
    
    for exp_dir in sorted(output_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith('exp'):
            continue
            
        experiment_info_path = exp_dir / "experiment_info.json"
        if not experiment_info_path.exists():
            print(f"Warning: experiment_info.json not found in {exp_dir}")
            continue
            
        with open(experiment_info_path) as f:
            exp_data = json.load(f)
        
        # Flatten the parameters
        flattened_data = {}
        for key, value in exp_data.items():
            if key == "parameters":
                # Flatten parameters with prefix
                for param_key, param_value in value.items():
                    flattened_data[f"param_{param_key}"] = param_value
            else:
                flattened_data[key] = value
        
        experiments.append(flattened_data)


    
    df = pd.DataFrame(experiments)
    df.rename(columns={"param_model_name": "model_name",
                       "param_model_kwargs.reasoning.effort": "effort"}, inplace=True)
    
    return df


def load_batch_log_data(output_dir: Path, experiments_df: pd.DataFrame) -> pd.DataFrame:
    """Load batch log data and merge with experiments dataframe."""
    batch_data = []
    
    for _, exp_row in experiments_df.iterrows():
        exp_id = exp_row['experiment_id']
        exp_dir = output_dir / f"exp{exp_id:03d}"
        
        # Find fattura_check_v1_* folder
        fattura_dirs = sorted(list(exp_dir.glob("fattura_check_v1_*")))
        if not fattura_dirs:
            print(f"Warning: No fattura_check_v1_* folder found in {exp_dir}")
            continue
            
        fattura_dir = fattura_dirs[0]  # Take the first one
        
        # Find batch_log*.json file
        batch_log_files = sorted(list(fattura_dir.glob("batch_log*.json")))
        if not batch_log_files:
            print(f"Warning: No batch_log*.json file found in {fattura_dir}")
            continue
            
        batch_log_path = batch_log_files[0]  # Take the first one
        
        with open(batch_log_path) as f:
            batch_data_raw = json.load(f)
        
        # Extract required fields
        batch_info = {
            "experiment_id": exp_id,
            "total_input_tokens": batch_data_raw.get("total_input_tokens", 0),
            "total_output_tokens": batch_data_raw.get("total_output_tokens", 0),
            "total_retry_count": batch_data_raw.get("total_retry_count", 0),
            "total_retry_input_tokens": batch_data_raw.get("total_retry_input_tokens", 0),
            "total_retry_output_tokens": batch_data_raw.get("total_retry_output_tokens", 0),
            "processing_time": batch_data_raw.get("processing_time", 0),
            "max_retries": batch_data_raw.get("max_retries", 0),
            "actual_cost": batch_data_raw.get("actual_cost", 0),
        }
        
        batch_data.append(batch_info)
    
    batch_df = pd.DataFrame(batch_data)
    
    # Merge with experiments dataframe
    if not batch_df.empty:
        merged_df = experiments_df.merge(batch_df, on="experiment_id", how="left")
    else:
        merged_df = experiments_df
        
    return merged_df


def load_ground_truth(ground_truth_path: Path) -> pd.DataFrame:
    """Load ground truth data."""
    if not ground_truth_path.exists():
        print(f"Warning: Ground truth file not found: {ground_truth_path}")
        return pd.DataFrame()
    
    try:
        # Try semicolon delimiter first (common in European CSV formats)
        return pd.read_csv(ground_truth_path, sep=';')
    except Exception:
        try:
            # Fallback to comma delimiter
            return pd.read_csv(ground_truth_path, sep=',')
        except Exception as e:
            print(f"Warning: Error reading ground truth file: {e}")
            print("Continuing analysis without ground truth comparison...")
            return pd.DataFrame()


def compare_with_ground_truth(main_table_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict[str, float]:
    """Compare main table data with ground truth and return accuracy metrics."""
    if main_table_df.empty or ground_truth_df.empty:
        return {
            "ddt_number_accuracy": 0.0,
            "ddt_date_accuracy": 0.0,
            "all_items_in_ddt_accuracy": 0.0,
            "invoice_ddt_match_accuracy": 0.0
        }
    
    results = {}
    
    # Merge on file_id for comparison
    if 'file_id' in main_table_df.columns and 'file_id' in ground_truth_df.columns:
        merged = main_table_df.merge(ground_truth_df, on='file_id', suffixes=('_pred', '_truth'))
    else:
        # If no file_id, compare row by row (assuming same order)
        min_len = min(len(main_table_df), len(ground_truth_df))
        merged = pd.concat([
            main_table_df.head(min_len).reset_index(drop=True),
            ground_truth_df.head(min_len).reset_index(drop=True)
        ], axis=1)
        
        # Add suffixes manually
        main_cols = main_table_df.columns.tolist()
        truth_cols = ground_truth_df.columns.tolist()
        new_cols = [f"{col}_pred" for col in main_cols] + [f"{col}_truth" for col in truth_cols]
        merged.columns = new_cols
    
    # Calculate accuracies
    if len(merged) > 0:
        # DDT number accuracy
        if 'ddt_number_pred' in merged.columns and 'ddt_number_truth' in merged.columns:
            ddt_number_matches = (merged['ddt_number_pred'] == merged['ddt_number_truth']).sum()
            results['ddt_number_accuracy'] = ddt_number_matches / len(merged)
        else:
            results['ddt_number_accuracy'] = 0.0
        
        # DDT date accuracy  
        if 'ddt_date_pred' in merged.columns and 'ddt_date_truth' in merged.columns:
            ddt_date_matches = (merged['ddt_date_pred'] == merged['ddt_date_truth']).sum()
            results['ddt_date_accuracy'] = ddt_date_matches / len(merged)
        else:
            results['ddt_date_accuracy'] = 0.0
        
        # All items in DDT accuracy (fix missing values to FALSE)
        if 'all_items_in_ddt_pred' in merged.columns and 'all_items_in_ddt_truth' in merged.columns:
            merged['all_items_in_ddt_pred'] = merged['all_items_in_ddt_pred'].infer_objects(copy=False)
            merged['all_items_in_ddt_truth'] = merged['all_items_in_ddt_truth'].infer_objects(copy=False)
            all_items_matches = (merged['all_items_in_ddt_pred'] == merged['all_items_in_ddt_truth']).sum()
            results['all_items_in_ddt_accuracy'] = all_items_matches / len(merged)
        else:
            results['all_items_in_ddt_accuracy'] = 0.0
        
        # Invoice DDT match accuracy (fix missing values to FALSE)
        if 'invoice_ddt_match_pred' in merged.columns and 'invoice_ddt_match_truth' in merged.columns:
            merged['invoice_ddt_match_pred'] = merged['invoice_ddt_match_pred'].infer_objects(copy=False)
            merged['invoice_ddt_match_truth'] = merged['invoice_ddt_match_truth'].infer_objects(copy=False)
            invoice_matches = (merged['invoice_ddt_match_pred'] == merged['invoice_ddt_match_truth']).sum()
            results['invoice_ddt_match_accuracy'] = invoice_matches / len(merged)
        else:
            results['invoice_ddt_match_accuracy'] = 0.0
    else:
        results = {
            "ddt_number_accuracy": 0.0,
            "ddt_date_accuracy": 0.0,
            "all_items_in_ddt_accuracy": 0.0,
            "invoice_ddt_match_accuracy": 0.0
        }
    
    return results


def load_all_results_tables(output_dir: Path, experiments_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all main_table.csv and items_table.csv files and create consolidated dataframes."""
    all_main_tables = []
    all_items_tables = []
    
    for _, exp_row in experiments_df.iterrows():
        exp_id = exp_row['experiment_id']
        exp_dir = output_dir / f"exp{exp_id:03d}"
        
        # Find fattura_check_v1_* folder
        fattura_dirs = list(exp_dir.glob("fattura_check_v1_*"))
        if not fattura_dirs:
            continue
            
        fattura_dir = fattura_dirs[0]
        
        # Load main_table.csv
        main_table_path = fattura_dir / "main_table.csv"
        if main_table_path.exists():
            main_df = pd.read_csv(main_table_path)
            main_df['experiment_id'] = exp_id
            all_main_tables.append(main_df)
        
        # Load items_table.csv
        items_table_path = fattura_dir / "items_table.csv"
        if items_table_path.exists():
            items_df = pd.read_csv(items_table_path)
            items_df['experiment_id'] = exp_id
            all_items_tables.append(items_df)
    
    main_tables_consolidated = pd.concat(all_main_tables, ignore_index=True) if all_main_tables else pd.DataFrame()
    items_tables_consolidated = pd.concat(all_items_tables, ignore_index=True) if all_items_tables else pd.DataFrame()
    
    return main_tables_consolidated, items_tables_consolidated



# %%
output_dir = Path("/Users/vigji/Desktop/pages_sample-data/concrete/grid_search/output_large_grid_final6")
ground_truth_path = Path("/Users/vigji/Desktop/pages_sample-data/concrete/grid_search/data/ground_truth/main_table_fixed.csv")

print("Starting grid search analysis...")
print(f"Output directory: {output_dir}")
print(f"Ground truth path: {ground_truth_path}")

# Load experiment data
print("\n1. Loading experiment data...")
experiments_df = load_experiment_data(output_dir)
print(f"Loaded {len(experiments_df)} experiments")

# Load batch log data and merge
print("\n2. Loading batch log data...")
experiments_df = load_batch_log_data(output_dir, experiments_df)
print(f"Merged batch log data for experiments")

# Load ground truth
print("\n3. Loading ground truth...")
ground_truth_df = load_ground_truth(ground_truth_path)
print(f"Loaded ground truth with {len(ground_truth_df)} records")

# Load all results tables
print("\n4. Loading all results tables...")
main_tables_df, items_tables_df = load_all_results_tables(output_dir, experiments_df)
with pd.option_context("future.no_silent_downcasting", True):
    main_tables_df = main_tables_df.fillna(False).infer_objects(copy=False)
    items_tables_df = items_tables_df.fillna(False).infer_objects(copy=False)

print(f"Loaded {len(main_tables_df)} main table records across all experiments")
print(f"Loaded {len(items_tables_df)} items table records across all experiments")

experiments_df["cost_per_page"] = experiments_df["actual_cost"] / 5

true_file_ids = ["test_pages_002", "test_pages_003", "test_pages_004"]
invalid_file_ids = ["test_pages_001", "test_pages_005"]
all_items_correct_pages = ["test_pages_002", "test_pages_003"]
all_items_incorrect_pages = ["test_pages_004"]
metrics = []
for exp_id, exp_main_table in main_tables_df.groupby("experiment_id"):
    valid_id_count = np.mean(exp_main_table[exp_main_table["file_id"].isin(true_file_ids)]["ddt_number"].values == ground_truth_df[ground_truth_df["file_id"].isin(true_file_ids)]["ddt_number"].values)
    valid_match_assessment = np.mean(exp_main_table[exp_main_table["file_id"].isin(true_file_ids)]["invoice_ddt_match"].values == ground_truth_df[ground_truth_df["file_id"].isin(true_file_ids)]["invoice_ddt_match"].values)
    invalid_match_assessment = np.mean(exp_main_table[exp_main_table["file_id"].isin(invalid_file_ids)]["invoice_ddt_match"].values == ground_truth_df[ground_truth_df["file_id"].isin(invalid_file_ids)]["invoice_ddt_match"].values)
    all_items_in_ddt_accuracy = np.mean(exp_main_table[exp_main_table["file_id"].isin(all_items_correct_pages)]["all_items_in_ddt"].values == ground_truth_df[ground_truth_df["file_id"].isin(all_items_correct_pages)]["all_items_in_ddt"].values)
    all_items_in_ddt_incorrect_accuracy = np.mean(exp_main_table[exp_main_table["file_id"].isin(all_items_incorrect_pages)]["all_items_in_ddt"].values == ground_truth_df[ground_truth_df["file_id"].isin(all_items_incorrect_pages)]["all_items_in_ddt"].values)
    metrics.append(dict(experiment_id=exp_id, 
                        valid_id_count=valid_id_count, 
                        valid_match_assessment=valid_match_assessment, 
                        invalid_match_assessment=invalid_match_assessment,
                        all_items_in_ddt_accuracy=all_items_in_ddt_accuracy,
                        all_items_in_ddt_incorrect_accuracy=all_items_in_ddt_incorrect_accuracy))

metrics_df = pd.DataFrame(metrics)
metrics_df.set_index("experiment_id", inplace=True)
experiments_df = experiments_df.merge(metrics_df, on="experiment_id", how="left")

wrong_ids = experiments_df.loc[~(experiments_df["all_items_in_ddt_incorrect_accuracy"].astype(bool)), "experiment_id"].values

dfs = main_tables_df.loc[main_tables_df["experiment_id"].isin(wrong_ids) & (main_tables_df["file_id"] == "test_pages_004"), "all_items_in_ddt"]
for df in dfs:
    print(df)
    print("-"*100)

# %%
sns.lineplot(data=experiments_df, x="effort", y="cost_per_page", hue="model_name", marker="o")
plt.title("Cost per Page vs Effort by Model")
plt.show()

# %%
sns.lineplot(data=experiments_df, x="effort", y="valid_id_count", hue="model_name", marker="o")
plt.title("Valid ID Count vs Effort by Model")
plt.show()
# %%
sns.lineplot(data=experiments_df, x="effort", y="valid_match_assessment", hue="model_name", marker="o")
plt.title("Valid Match Assessment vs Effort by Model")
plt.show()
# %%
sns.lineplot(data=experiments_df, x="effort", y="invalid_match_assessment", hue="model_name", marker="o")
plt.title("Invalid Match Assessment vs Effort by Model")
plt.show()
# %%
# %%
sns.lineplot(data=experiments_df, x="effort", y="all_items_in_ddt_accuracy", hue="model_name", marker="o")
plt.title("All Items in DDT Accuracy vs Effort by Model")
plt.show()
# %%
sns.lineplot(data=experiments_df, x="effort", y="all_items_in_ddt_incorrect_accuracy", hue="model_name", marker="o")
plt.title("All Items in DDT Incorrect Accuracy vs Effort by Model")
plt.show()
# %%
ground_truth_df
# %%
wrong_ids = experiments_df.loc[~(experiments_df["all_items_in_ddt_incorrect_accuracy"].astype(bool)), "experiment_id"].values

dfs = main_tables_df.loc[main_tables_df["experiment_id"].isin(wrong_ids) & (main_tables_df["file_id"] == "test_pages_004"), "all_items_in_ddt"]
for df in dfs:
    print(df)
    print("-"*100)
# %%
# %%
wrong_ids
# %%
main_tables_df["experiment_id"]
# %%
main_tables_df
# %%
