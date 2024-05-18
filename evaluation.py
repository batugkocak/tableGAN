import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div

def ks_test(data1, data2):
    statistic, p_value = ks_2samp(data1, data2)
    return statistic, p_value

def js_divergence(data1, data2, num_bins=100):
    # Create histograms
    hist1, bin_edges1 = np.histogram(data1, bins=num_bins, density=True)
    hist2, bin_edges2 = np.histogram(data2, bins=num_bins, density=True)
    
    # Normalize the histograms to get probability distributions
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Calculate the mid-point distribution
    m = 0.5 * (hist1 + hist2)
    
    # Calculate JS divergence
    js_div = 0.5 * (rel_entr(hist1, m).sum() + rel_entr(hist2, m).sum())
    return js_div


import numpy as np
from scipy.special import rel_entr

def kl_divergence(data1, data2, epsilon=1e-9, num_bins=100):
    # Create histograms
    hist1, bin_edges1 = np.histogram(data1, bins=num_bins, density=True)
    hist2, bin_edges2 = np.histogram(data2, bins=num_bins, density=True)
    
    # Normalize the histograms to get probability distributions
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Add epsilon to avoid division by zero or log(0)
    hist1 = hist1 + epsilon
    hist2 = hist2 + epsilon
    
    # Calculate KL divergence
    kl_div1 = rel_entr(hist1, hist2).sum()
    kl_div2 = rel_entr(hist2, hist1).sum()
    
    return kl_div1, kl_div2


df1 = pd.read_csv('./samples/Adult/Adult_OI_11_00_fake.csv')
df2 = pd.read_csv('./data/Adult/Adult.csv')

# Column names (based on your sample data)
columns = ["age", "workclass", "fnlwgt","education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", 
           "capital-loss", "hours-per-week", "native-country"]

df1.columns = columns
df2.columns = columns
# Remove rows with missing values.
df1.dropna(inplace=True)
df2.dropna(inplace=True)

# Create dictionaries to store results
ks_results = {}
js_results = {}
kl_results = {}

# Explicitly choose numeric columns for comparison
common_columns = columns 

# Iterate through columns and calculate metrics
for col in common_columns:
    data1 = df1[col].astype(float).values
    data2 = df2[col].astype(float).values
    # Kolmogorov-Smirnov Test
    ks_statistic, ks_p_value = ks_test(data1, data2)
    ks_results[col] = {'statistic': ks_statistic, 'p_value': ks_p_value}
    
    # Jensen-Shannon Divergence
    js_div = js_divergence(data1, data2)
    js_results[col] = js_div

    # KL Divergence
    kl_div_1to2, kl_div_2to1 = kl_divergence(data1, data2)
    kl_results[col] = {'Data1 to Data2': kl_div_1to2, 'Data2 to Data1': kl_div_2to1}

# Display results
print("Kolmogorov-Smirnov Test Results:")
for col, result in ks_results.items():
    print(f"Column: {col}, Statistic: {result['statistic']:.4f}, P-Value: {result['p_value']:.4f}")

print("\nJensen-Shannon Divergence Results:")
for col, div in js_results.items():
    print(f"Column: {col}, Divergence: {div:.4f}")

print("\nKL Divergence Results:")
for col, divs in kl_results.items():
    print(f"Column: {col}, Data1 to Data2: {divs['Data1 to Data2']:.4f}, Data2 to Data1: {divs['Data2 to Data1']:.4f}")
