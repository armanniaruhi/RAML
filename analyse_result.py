import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create output directories if they don't exist
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [
        'Liberation Serif',
        'DejaVu Serif',
        'Times New Roman',
        'serif'
    ],
    'font.size': 5,
    'axes.labelsize': 4,
    'axes.titlesize': 4,
    'xtick.labelsize': 3,
    'ytick.labelsize': 3,
    'legend.fontsize': 3,
    'figure.titlesize': 2,
    'figure.dpi': 300,
    'lines.linewidth': 0.2,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'axes.titlepad': 3,
    'axes.labelpad': 2,
})

# Load data
df = pd.read_csv("results/metric_results.csv")
df = df[~df['key'].isin(['batch_train_loss', 'val_roc_auc'])]
pivot_df = df.pivot_table(index=['run_id', 'step'], columns='key', values='value').reset_index()

# Get metrics and runs
all_metrics = [col for col in pivot_df.columns if col not in ['run_id', 'step']]
run_ids = pivot_df['run_id'].unique()
colors = ["#062f4d", "#1993b8", '#ff7f0e']

# =============================================
# ANALYSIS FUNCTIONS
# =============================================

def create_comparison_tables(pivot_df, output_dir):
    """Create and save comparison tables to Excel"""
    writer = pd.ExcelWriter(os.path.join(output_dir, 'metric_comparisons.xlsx'), engine='xlsxwriter')
    
    # 1. Final values comparison
    final_values = pivot_df.sort_values('step').groupby('run_id').last().reset_index()
    final_values.to_excel(writer, sheet_name='Final Values', index=False)
    
    # 2. Summary statistics
    stats_df = final_values.drop(columns=['run_id', 'step']).agg(['mean', 'std', 'min', 'max', 'median'])
    stats_df.to_excel(writer, sheet_name='Summary Statistics')
    
    # 3. Training dynamics (metrics at specific steps)
    steps = np.linspace(0, pivot_df['step'].max(), 5, dtype=int)
    dynamics_data = []
    for step in steps:
        step_data = pivot_df[pivot_df['step'] <= step].groupby('run_id').last().reset_index()
        step_data['analysis_step'] = step
        dynamics_data.append(step_data)
    dynamics_df = pd.concat(dynamics_data)
    dynamics_df.to_excel(writer, sheet_name='Training Dynamics', index=False)
    
    # 4. Relative improvement
    initial = pivot_df.groupby('run_id').first().add_suffix('_initial')
    final = pivot_df.groupby('run_id').last().add_suffix('_final')
    improvement = (final.values - initial.values) / initial.values
    improvement_df = pd.DataFrame(
        improvement,
        columns=[col.replace('_final', '_improvement') for col in final.columns],
        index=final.index
    )
    improvement_df.reset_index(inplace=True)
    improvement_df.to_excel(writer, sheet_name='Relative Improvement', index=False)
    
    writer.close()

# =============================================
# PLOTTING FUNCTIONS (your existing code)
# =============================================

def plot_loss_comparison(pivot_df, output_dir):
    """Plot individual loss curves for each run (no subplots)"""
    colors = ['#1f77b4', '#ff7f0e']
    for run_id in run_ids:
        run_data = pivot_df[pivot_df['run_id'] == run_id].sort_values('step')
        
        fig, ax = plt.subplots(figsize=(1.5, 2))  # Small, readable size
        
        ax.plot(run_data['step'], run_data['training loss'], 
                color=colors[0], linewidth=0.6, label='Train loss')
        ax.plot(run_data['step'], run_data['validation loss'], 
                color=colors[1], linewidth=0.6, label='Validation loss')
        
        # Highlight final points
        last = run_data.iloc[-1]
        ax.plot(last['step'], last['training loss'], 'o', color=colors[0], markersize=2)
        ax.plot(last['step'], last['validation loss'], 'o', color=colors[1], markersize=2)
        
        y_pad = 0.05 * (run_data[['training loss', 'validation loss']].max().max() -
                        run_data[['training loss', 'validation loss']].min().min())
        ax.set_ylim([
            run_data[['training loss', 'validation loss']].min().min() - y_pad,
            run_data[['training loss', 'validation loss']].max().max() + y_pad
        ])
    
        ax.set_xlabel('Step', labelpad=1)
        ax.set_ylabel('Loss', labelpad=1)
        ax.legend(loc='upper right', fontsize=4, handlelength=0.5, handletextpad=0.2)
        
        plt.tight_layout(pad=0.4)
        filename = f"loss_{run_id.replace('/', '_')}.png"
        plt.savefig(os.path.join(output_dir, 'plots', filename), bbox_inches='tight')
        plt.close()


def plot_metric_comparisons(pivot_df, output_dir):
    """Plot individual metric comparisons"""
    for metric in [m for m in all_metrics if m not in ['training loss', 'validation loss']]:
        fig, ax = plt.subplots(figsize=(1.5, 1.8))
        
        for run_idx, run_id in enumerate(run_ids):
            run_data = pivot_df[pivot_df['run_id']==run_id].sort_values('step')
            ax.plot(run_data['step'], run_data[metric], 
                   color=colors[run_idx%len(colors)],
                   linewidth=0.5,
                   label=run_id)
            
        ax.set_xlabel('Step', labelpad=1)
        
        
        ax.legend(loc='lower right', fontsize=4, handlelength=0.5, handletextpad=0.2)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(output_dir, 'plots', f'{metric}.png'), bbox_inches='tight')
        plt.close()

# =============================================
# EXECUTE ANALYSIS
# =============================================

# Generate all plots
plot_loss_comparison(pivot_df, "results")
plot_metric_comparisons(pivot_df, "results")

# Generate comparison tables
create_comparison_tables(pivot_df, "results/tables")

print("Analysis complete!")
print(f"Plots saved to: results/plots")
print(f"Tables saved to: results/tables/metric_comparisons.xlsx")