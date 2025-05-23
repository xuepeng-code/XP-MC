import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_preprocess(file1, file2):

    df1 = pd.read_csv(file1, index_col=0)
    df2 = pd.read_csv(file2, index_col=0)

    common_samples = df1.index.intersection(df2.index)
    df1 = df1.loc[common_samples]
    df2 = df2.loc[common_samples]

    df1 = df1.apply(pd.to_numeric, errors='coerce')
    df2 = df2.apply(pd.to_numeric, errors='coerce')

    df1_scaled = (df1 - df1.mean()) / df1.std()
    df2_scaled = (df2 - df2.mean()) / df2.std()
    
    return df1_scaled, df2_scaled, common_samples

def calculate_correlations(df1, df2, name1, name2):

    results = []
    for col1 in df1.columns:
        for col2 in df2.columns:

            valid_idx = df1[col1].notna() & df2[col2].notna()
            if sum(valid_idx) < 3:  
                continue
            
            corr, p_value = pearsonr(df1[col1][valid_idx], df2[col2][valid_idx])
            results.append({
                f"{name1}": col1,
                f"{name2}": col2,
                "Correlation": corr,
                "P_Value": p_value
            })
    return pd.DataFrame(results)
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 26
def process_pair(file_pair, output_dir="ANA_results"):

    os.makedirs(output_dir, exist_ok=True)
    

    (file1, name1), (file2, name2) = file_pair
    print(f"\nProcessing: {name1} vs {name2}")

    df1, df2, samples = load_and_preprocess(file1, file2)
    print(f"Common samples: {len(samples)}")
    

    results_df = calculate_correlations(df1, df2, name1, name2)

    p_vals = results_df["P_Value"].values
    if len(p_vals) > 0:
        reject, p_adjusted, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
        results_df["P_Adjusted"] = p_adjusted
        results_df["Significant"] = reject
    else:
        print("No valid correlations found!")
        return

    pair_name = f"{name1}_vs_{name2}"
    results_file = os.path.join(output_dir, f"{pair_name}_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Saved results to: {results_file}")

    cor_matrix = results_df.pivot(index=name1, columns=name2, values="Correlation")
    signif_matrix = results_df.pivot(index=name1, columns=name2, values="Significant").fillna(False)

    annot_matrix = signif_matrix.applymap(lambda x: "*" if x else "")
    

    plt.figure(figsize=(20, 15))
    sns.heatmap(
        cor_matrix, 
        cmap="coolwarm", 
        center=0,
        annot=annot_matrix,  
        fmt="",              
        # mask=mask,           
        annot_kws={
            "size": 15,      
            "color": "black",
            "weight": "bold" 
        },
        cbar_kws={"label": "Pearson Correlation"},
        linewidths=0.8,
        linecolor='grey'
    )
    
    plt.title(f"Correlations: {name1} vs {name2}")
    plt.tight_layout()
    

    plot_path = os.path.join(output_dir, f"{name1}_vs_{name2}_significant_heatmap.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved annotated heatmap to: {plot_path}")


data_config = [
    ("explaindata/ALL/PROTEIN_DATA.csv", "Proteomics"),
    ("explaindata/ALL/ROI1.csv", "Brain regions"),
    ("explaindata/ALL/GENE_DATAF.csv", "SNPs")  
]


from itertools import combinations
file_pairs = list(combinations(data_config, 2))


for pair in file_pairs:
    process_pair(pair)

print("finished")