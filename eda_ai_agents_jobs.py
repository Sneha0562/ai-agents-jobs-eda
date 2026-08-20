"""
Exploratory Data Analysis (EDA) for AI Agents Global Industry Jobs and Security Dataset
This script performs comprehensive EDA on the Kaggle AI agents jobs dataset.
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. DOWNLOAD AND LOAD DATASET
# ============================================================================

print("=" * 80)
print("Downloading dataset from Kaggle...")
print("=" * 80)

path = kagglehub.dataset_download("zkskhurram/ai-agents-global-industry-jobs-and-security")
print(f"Path to dataset files: {path}\n")

# Load the dataset (adjust filename if needed)
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"CSV files found: {csv_files}")

if csv_files:
    df = pd.read_csv(os.path.join(path, csv_files[0]))
else:
    print("No CSV files found!")
    exit()

# ============================================================================
# 2. DATASET OVERVIEW
# ============================================================================

print("\n" + "=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print(f"\nDataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\n" + "-" * 80)
print("COLUMN INFORMATION")
print("-" * 80)
print(df.info())

print("\n" + "-" * 80)
print("FIRST FEW ROWS")
print("-" * 80)
print(df.head(10))

print("\n" + "-" * 80)
print("BASIC STATISTICS")
print("-" * 80)
print(df.describe())

# ============================================================================
# 3. MISSING VALUES ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_data) > 0:
    print("\nMissing Values:")
    print(missing_data.to_string(index=False))
else:
    print("\nNo missing values found!")

# Visualize missing values
if len(missing_data) > 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    missing_data.set_index('Column')['Missing_Percentage'].plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_title('Missing Values Distribution')
    plt.tight_layout()
    plt.savefig('01_missing_values.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 01_missing_values.png")
    plt.close()

# ============================================================================
# 4. DATA TYPES ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("DATA TYPES ANALYSIS")
print("=" * 80)

print("\nData Types Distribution:")
dtype_counts = df.dtypes.value_counts()
print(dtype_counts)

fig, ax = plt.subplots(figsize=(10, 5))
dtype_counts.plot(kind='bar', ax=ax, color='skyblue')
ax.set_title('Data Types Distribution')
ax.set_xlabel('Data Type')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('02_data_types.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 02_data_types.png")
plt.close()

# ============================================================================
# 5. NUMERICAL COLUMNS ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("NUMERICAL COLUMNS ANALYSIS")
print("=" * 80)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumerical Columns ({len(numerical_cols)}): {numerical_cols}")

if numerical_cols:
    print("\nDescriptive Statistics:")
    print(df[numerical_cols].describe().round(2))
    
    # Distribution plots for numerical columns
    n_cols = len(numerical_cols)
    fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(15, 4 * ((n_cols + 1) // 2)))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Remove extra subplots
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('03_numerical_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 03_numerical_distributions.png")
    plt.close()

# ============================================================================
# 6. CATEGORICAL COLUMNS ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CATEGORICAL COLUMNS ANALYSIS")
print("=" * 80)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical Columns ({len(categorical_cols)}): {categorical_cols}")

for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"\n{col}:")
    print(f"  - Unique Values: {unique_count}")
    if unique_count <= 20:
        print(f"  - Value Counts:\n{df[col].value_counts()}")
    else:
        print(f"  - Top 10 Value Counts:\n{df[col].value_counts().head(10)}")

# Visualize top categorical features
categorical_to_plot = [col for col in categorical_cols if df[col].nunique() <= 15]
n_cols = len(categorical_to_plot)

if n_cols > 0:
    fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(15, 4 * ((n_cols + 1) // 2)))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_to_plot):
        top_cats = df[col].value_counts().head(10)
        axes[idx].barh(range(len(top_cats)), top_cats.values, color='teal', alpha=0.7)
        axes[idx].set_yticks(range(len(top_cats)))
        axes[idx].set_yticklabels(top_cats.index)
        axes[idx].set_title(f'Top Values in {col}')
        axes[idx].set_xlabel('Count')
    
    # Remove extra subplots
    for idx in range(len(categorical_to_plot), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('04_categorical_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 04_categorical_distributions.png")
    plt.close()

# ============================================================================
# 7. CORRELATION ANALYSIS (if numerical columns exist)
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

if len(numerical_cols) > 1:
    corr_matrix = df[numerical_cols].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(2))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax, fmt='.2f', cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: 05_correlation_heatmap.png")
    plt.close()
else:
    print("\nInsufficient numerical columns for correlation analysis.")

# ============================================================================
# 8. DUPLICATE RECORDS ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("DUPLICATE RECORDS ANALYSIS")
print("=" * 80)

total_duplicates = df.duplicated().sum()
total_duplicate_rows = df.duplicated(keep=False).sum()

print(f"\nTotal Duplicate Rows: {total_duplicates}")
print(f"Total Rows Involved in Duplicates: {total_duplicate_rows}")

if total_duplicates > 0:
    print("\nSample Duplicate Rows:")
    print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head(10))

# ============================================================================
# 9. OUTLIER DETECTION
# ============================================================================

print("\n" + "=" * 80)
print("OUTLIER DETECTION (Numerical Columns)")
print("=" * 80)

def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[col] < lower_bound) | (data[col] > upper_bound)

for col in numerical_cols:
    outliers = detect_outliers_iqr(df, col)
    outlier_count = outliers.sum()
    outlier_percentage = (outlier_count / len(df)) * 100
    if outlier_count > 0:
        print(f"\n{col}:")
        print(f"  - Outliers: {outlier_count} ({outlier_percentage:.2f}%)")

# ============================================================================
# 10. SUMMARY STATISTICS TABLE
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary_data = {
    'Metric': [
        'Total Records',
        'Total Columns',
        'Numerical Columns',
        'Categorical Columns',
        'Missing Values',
        'Duplicate Records',
        'Memory Usage (MB)'
    ],
    'Value': [
        df.shape[0],
        df.shape[1],
        len(numerical_cols),
        len(categorical_cols),
        df.isnull().sum().sum(),
        total_duplicates,
        round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
print("\nGenerated Visualizations:")
print("  1. 01_missing_values.png - Missing values distribution")
print("  2. 02_data_types.png - Data types breakdown")
print("  3. 03_numerical_distributions.png - Distributions of numerical features")
print("  4. 04_categorical_distributions.png - Top values in categorical features")
print("  5. 05_correlation_heatmap.png - Correlation matrix (if applicable)")
