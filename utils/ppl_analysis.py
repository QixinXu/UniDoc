import json
import numpy as np
import matplotlib.pyplot as plt

# --- Configurations ---
PLOT_STYLE_CONFIG = {
    'title_fontsize': 28, 'label_fontsize': 26,
    'legend_fontsize': 24, 'tick_fontsize': 22,
}
PLOT_CONFIG = {
    'MIN_LEN_DIFF_THRESHOLD': 20
}
plt.style.use('seaborn-v0_8-whitegrid')

# --- Helper Functions ---
def remove_outliers(data):
    """Removes outliers from a NumPy array using the IQR method."""
    if len(data) == 0: return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    if iqr == 0: return data
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data[(data >= lower) & (data <= upper)]

def get_optimal_bin_width(data):
    """Computes optimal bin width using the Freedman-Diaconis rule."""
    if len(data) < 2: return 1
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    n = len(data)
    bin_width = 2 * iqr * (n ** (-1/3))
    if bin_width < 1e-8:
        return (data.max() - data.min()) / np.sqrt(n)
    return bin_width

# --- Data Loading and Processing ---
file_path = "/media/cheliu/cl_code/Entropy/synth_writing/synth_collection/collection_0810_20K.jsonl"
try:
    with open(file_path, 'r') as f: data = [json.loads(line) for line in f]

    ppl_before, ppl_after, token_len_before, token_len_after = [], [], [], []
    for item in data:
        if not item.get('refinements'): continue
        ref = item['refinements'][0]
        ppl_diffs = [float(x) for x in ref['ppl_diff'][-1].split('->')]
        ppl_before.append(ppl_diffs[0])
        ppl_after.append(ppl_diffs[1])
        token_len_before.append(ref['initial_tokens'])
        token_len_after.append(ref['final_tokens'])

    # Convert, calculate differences, and filter outliers
    ppl_before, ppl_after = np.array(ppl_before), np.array(ppl_after)
    token_len_before, token_len_after = np.array(token_len_before), np.array(token_len_after)
    ppl_difference = ppl_before - ppl_after
    token_len_difference = token_len_after - token_len_before

    ppl_before_filtered = remove_outliers(ppl_before)
    ppl_after_filtered = remove_outliers(ppl_after)
    ppl_difference_filtered = remove_outliers(ppl_difference)
    token_len_before_filtered = remove_outliers(token_len_before)
    token_len_after_filtered = remove_outliers(token_len_after)
    token_len_difference_filtered = remove_outliers(token_len_difference)
    
    token_len_diff_thresholded = token_len_difference_filtered[
        token_len_difference_filtered > PLOT_CONFIG['MIN_LEN_DIFF_THRESHOLD']
    ]

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(32, 7))

    # Plot 1: PPL Distribution (Auto-binned)
    ax1 = axes[0]
    combined_ppl = np.concatenate((ppl_before_filtered, ppl_after_filtered))
    bin_width_ppl = get_optimal_bin_width(combined_ppl)
    bins_ppl = np.arange(np.floor(combined_ppl.min()), np.ceil(combined_ppl.max()) + bin_width_ppl, bin_width_ppl)
    ax1.hist(ppl_before_filtered, bins=bins_ppl, alpha=0.7, label='Before', color='#4A4488', edgecolor='none')
    ax1.hist(ppl_after_filtered, bins=bins_ppl, alpha=0.7, label='After', color='#4E9A91', edgecolor='none')
    # CORRECTED LINES: Using specific keyword arguments
    ax1.set_title('PPL Distribution', fontsize=PLOT_STYLE_CONFIG['title_fontsize'], fontweight='bold')
    ax1.set_xlabel('Perplexity (PPL)', fontsize=PLOT_STYLE_CONFIG['label_fontsize'])
    ax1.set_ylabel('Frequency', fontsize=PLOT_STYLE_CONFIG['label_fontsize'])
    ax1.legend(fontsize=PLOT_STYLE_CONFIG['legend_fontsize'])

    # Plot 2: PPL Improvement (Auto-binned)
    ax2 = axes[1]
    bin_width_ppl_diff = get_optimal_bin_width(ppl_difference_filtered)
    bins_ppl_diff = np.arange(np.floor(ppl_difference_filtered.min()), np.ceil(ppl_difference_filtered.max()) + bin_width_ppl_diff, bin_width_ppl_diff)
    ax2.hist(ppl_difference_filtered, bins=bins_ppl_diff, color='#9E453F', edgecolor='none')
    # CORRECTED LINES
    ax2.set_title('PPL Improvement', fontsize=PLOT_STYLE_CONFIG['title_fontsize'], fontweight='bold')
    ax2.set_xlabel('PPL Difference (Before - After)', fontsize=PLOT_STYLE_CONFIG['label_fontsize'])

    # Plot 3: Token Length Distribution (Auto-binned)
    ax3 = axes[2]
    combined_lengths = np.concatenate((token_len_before_filtered, token_len_after_filtered))
    bin_width_token = get_optimal_bin_width(combined_lengths)
    bin_width_token = max(1, round(bin_width_token))
    bins_len = np.arange(np.floor(combined_lengths.min()), np.ceil(combined_lengths.max()) + bin_width_token, bin_width_token)
    ax3.hist(token_len_before_filtered, bins=bins_len, alpha=0.7, label='Before', color='#4A4488', edgecolor='none')
    ax3.hist(token_len_after_filtered, bins=bins_len, alpha=0.7, label='After', color='#4E9A91', edgecolor='none')
    # CORRECTED LINES
    ax3.set_title('Token Length Distribution', fontsize=PLOT_STYLE_CONFIG['title_fontsize'], fontweight='bold')
    ax3.set_xlabel('Token Length', fontsize=PLOT_STYLE_CONFIG['label_fontsize'])
    ax3.legend(fontsize=PLOT_STYLE_CONFIG['legend_fontsize'])

    # Plot 4: Length Difference Distribution (Auto-binned)
    ax4 = axes[3]
    bin_width_token_diff = get_optimal_bin_width(token_len_diff_thresholded)
    bin_width_token_diff = max(1, round(bin_width_token_diff))
    bins_diff = np.arange(np.floor(token_len_diff_thresholded.min()), np.ceil(token_len_diff_thresholded.max()) + bin_width_token_diff, bin_width_token_diff)
    ax4.hist(token_len_diff_thresholded, bins=bins_diff, color='#8C8C8C', edgecolor='none')
    # CORRECTED LINES
    ax4.set_title('Token Length Change', fontsize=PLOT_STYLE_CONFIG['title_fontsize'], fontweight='bold')
    ax4.set_xlabel(f'Length Diff > {PLOT_CONFIG["MIN_LEN_DIFF_THRESHOLD"]}', fontsize=PLOT_STYLE_CONFIG['label_fontsize'])

    # Final Figure Adjustments
    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=PLOT_STYLE_CONFIG['tick_fontsize'])

    plt.tight_layout(pad=2.0)
    plt.savefig('/media/cheliu/cl_code/Entropy/writing_analysis_related/ppl_distr.pdf', dpi=300, bbox_inches='tight')
    print("Plots saved as distributions_auto_binned_v2.png")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")