import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the CSV data
df = pd.read_csv('beam_search_results.csv')

# Set up the plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.ioff()  # Turn off interactive mode to prevent display issues

def calculate_mode(series):
    """Calculate mode, handling cases where mode might not exist or be non-unique"""
    try:
        mode_result = stats.mode(series, keepdims=True)
        if len(mode_result.mode) > 0:
            return mode_result.mode[0]
        else:
            return np.nan
    except:
        return np.nan

def calculate_stats_by_parameter(df, param_col, metrics):
    """Calculate median and mode statistics for metrics grouped by parameter"""
    stats_dict = {'median': {}, 'mode': {}}
    
    grouped = df.groupby(param_col)
    
    for metric in metrics:
        # Calculate median
        median_values = grouped[metric].median()
        stats_dict['median'][metric] = median_values.to_dict()
        
        # Calculate mode
        mode_values = grouped[metric].apply(calculate_mode)
        stats_dict['mode'][metric] = mode_values.to_dict()
    
    return stats_dict

def plot_metrics_comparison(df, param_name, metrics, title_suffix):
    """Plot metrics with both median and mode for comparison"""
    # Calculate statistics
    stats = calculate_stats_by_parameter(df, param_name, metrics)
    
    # Create DataFrames for plotting
    median_data = []
    mode_data = []
    
    for param_val in sorted(df[param_name].unique()):
        row_median = {'parameter': param_val, 'stat_type': 'Median'}
        row_mode = {'parameter': param_val, 'stat_type': 'Mode'}
        
        for metric in metrics:
            row_median[metric] = stats['median'][metric].get(param_val, np.nan)
            row_mode[metric] = stats['mode'][metric].get(param_val, np.nan)
        
        median_data.append(row_median)
        mode_data.append(row_mode)
    
    combined_data = median_data + mode_data
    plot_df = pd.DataFrame(combined_data)
    
    # Create subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        # Filter out infinite values for plotting
        plot_data = plot_df.replace([float('inf'), -float('inf')], np.nan)
        
        # Create line plot with different markers for median and mode
        for stat_type in ['Median', 'Mode']:
            subset = plot_data[plot_data['stat_type'] == stat_type]
            marker = 'o' if stat_type == 'median' else 's'
            linestyle = '-' if stat_type == 'median' else '--'
            
            axes[i].plot(subset['parameter'], subset[metric], 
                        marker=marker, linestyle=linestyle, 
                        label=f'{stat_type}', linewidth=2, markersize=8)
        
        axes[i].set_title(f'{metric.replace("_", " ").title()} vs. {param_name.replace("_", " ").title()}')
        axes[i].set_xlabel(param_name.replace("_", " ").title())
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'csv_analysis_{param_name}_comparison.png', dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()

def plot_distribution_comparison(df, param_name, metric, title_suffix):
    """Create box plots to show distribution of metrics by parameter"""
    plt.figure(figsize=(12, 6))
    
    # Convert infinite values to NaN for plotting
    df_plot = df.replace([float('inf'), -float('inf')], np.nan)
    
    sns.boxplot(data=df_plot, x=param_name, y=metric)
    plt.title(f'Distribution of {metric.replace("_", " ").title()} by {param_name.replace("_", " ").title()}')
    plt.xlabel(param_name.replace("_", " ").title())
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'csv_analysis_{param_name}_{metric}_distribution.png', dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()

# Define metrics to analyze
numerical_metrics = ['mse', 'inference_time']
parameters = ['beam_size', 'length_penalty', 'max_len']

print("Generating comparison plots (Median vs Mode) from CSV data...")
print("=" * 60)

# Generate comparison plots for each parameter
for param in parameters:
    print(f"\nAnalyzing parameter: {param}")
    print("-" * 40)
    
    # Plot median vs mode comparison
    plot_metrics_comparison(df, param, numerical_metrics, f'(CSV Analysis)')
    
    # Generate distribution plots for key metrics
    for metric in ['mse', 'inference_time']:
        plot_distribution_comparison(df, param, metric, f'(CSV Analysis)')

# Generate summary statistics table
print("\nSummary Statistics by Parameter:")
print("=" * 60)

for param in parameters:
    print(f"\n{param.replace('_', ' ').title()}:")
    print("-" * 30)
    
    stats = calculate_stats_by_parameter(df, param, ['mse', 'inference_time'])
    
    for metric in ['mse', 'inference_time']:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for param_val in sorted(df[param].unique()):
            median_val = stats['median'][metric].get(param_val, np.nan)
            mode_val = stats['mode'][metric].get(param_val, np.nan)
            print(f"  {param_val}: Median={median_val:.6f}, Mode={mode_val:.6f}")

# Additional analysis: Find best configurations based on different criteria
print("\n\nBest Configurations from CSV Data:")
print("=" * 60)

# Best (lowest) MSE
best_mse = df.loc[df['mse'].idxmin()]
print(f"\nBest MSE ({best_mse['mse']:.2e}):")
print(f"  Beam Size: {best_mse['beam_size']}")
print(f"  Length Penalty: {best_mse['length_penalty']}")
print(f"  Max Length: {best_mse['max_len']}")
print(f"  Equation: {best_mse['equation_name']}")

# Fastest inference time
fastest = df.loc[df['inference_time'].idxmin()]
print(f"\nFastest Inference ({fastest['inference_time']:.3f}s):")
print(f"  Beam Size: {fastest['beam_size']}")
print(f"  Length Penalty: {fastest['length_penalty']}")
print(f"  Max Length: {fastest['max_len']}")
print(f"  Equation: {fastest['equation_name']}")

print("\nPlot generation complete! Check the generated PNG files for visualizations.")
