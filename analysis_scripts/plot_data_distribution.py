import os
import pandas as pd

import matplotlib.pyplot as plt

# Path to the CSV file
csv_path = '.tmp/val_dataset.csv'
# Output directory for plots
output_dir = '.tmp/data_distribution_plots_val_set'
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_path)

# Plot distribution for each column
for col in df.columns:
    plt.figure()
    # Choose plot type based on data type
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].plot.hist(bins=30, alpha=0.7)
        plt.xlabel(col)
        plt.title(f'Distribution of {col}')
    else:
        df[col].value_counts().plot.bar()
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(f'Value Counts of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
    plt.close()