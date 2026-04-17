# =============================================================================
# visualize.py — Stage 3.1
# Seven plots saved as PNG files in PLOTS_DIR
# Each plot is wrapped in try/except — one failure won't stop the rest
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats

# config.py already called matplotlib.use('Agg') before pyplot was imported
from config import FEATURE_COLS, FAULT_NAMES, PLOTS_DIR


def save_plot(filename):
    """Save current figure to PLOTS_DIR and close it cleanly."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Saved → {path}")


def plot_class_distribution(df):
    """Plot 1 — Bar chart of sample counts per fault_type."""
    try:
        counts = df['fault_type'].value_counts().sort_index()
        labels = [FAULT_NAMES[i] for i in counts.index]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, counts.values, color='steelblue', edgecolor='black')
        plt.title('Class Distribution — Fault Types')
        plt.xlabel('Fault Type')
        plt.ylabel('Sample Count')
        plt.xticks(rotation=15)
        plt.tight_layout()
        save_plot('01_class_distribution.png')

    except Exception as e:
        print(f"    [ERROR] Plot 1 failed: {e}")


def plot_boxplots(df):
    """Plot 2 — Box plot for each feature grouped by fault_type."""
    try:
        fig, axes = plt.subplots(6, 3, figsize=(18, 24))
        axes = axes.flatten()

        for idx, feature in enumerate(FEATURE_COLS):
            ax = axes[idx]
            groups = [df[df['fault_type'] == ft][feature].values
                      for ft in sorted(FAULT_NAMES.keys())]
            ax.boxplot(groups, labels=list(FAULT_NAMES.values()), patch_artist=True)
            ax.set_title(feature, fontsize=9)
            ax.tick_params(axis='x', rotation=45, labelsize=7)

        # Hide the unused 18th subplot slot
        for idx in range(len(FEATURE_COLS), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Box Plots — Features Grouped by Fault Type', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_plot('02_boxplots_per_feature.png')

    except Exception as e:
        print(f"    [ERROR] Plot 2 failed: {e}")


def plot_correlation_heatmap(df):
    """Plot 3 — Correlation heatmap of all 17 features."""
    try:
        corr = df[FEATURE_COLS].corr()

        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    linewidths=0.5, annot_kws={'size': 7})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        save_plot('03_correlation_heatmap.png')

    except Exception as e:
        print(f"    [ERROR] Plot 3 failed: {e}")


def plot_histograms(df):
    """Plot 4 — Overlapping histograms for each feature, coloured by fault_type."""
    try:
        cmap = matplotlib.colormaps.get_cmap('tab10')
        fig, axes = plt.subplots(6, 3, figsize=(18, 24))
        axes = axes.flatten()

        for idx, feature in enumerate(FEATURE_COLS):
            ax = axes[idx]
            for ft in sorted(FAULT_NAMES.keys()):
                subset = df[df['fault_type'] == ft][feature]
                ax.hist(subset, bins=30, alpha=0.5,
                        label=FAULT_NAMES[ft], color=cmap(ft / 9))
            ax.set_title(feature, fontsize=9)
            ax.legend(fontsize=6)

        for idx in range(len(FEATURE_COLS), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Feature Distributions by Fault Type', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_plot('04_feature_histograms.png')

    except Exception as e:
        print(f"    [ERROR] Plot 4 failed: {e}")


def plot_outlier_counts(df):
    """Plot 5 — Bar chart of outlier counts per feature using Z-score method."""
    try:
        outlier_counts = []
        for feature in FEATURE_COLS:
            z = np.abs(stats.zscore(df[feature].dropna()))
            outlier_counts.append((z > 3).sum())

        plt.figure(figsize=(14, 5))
        plt.bar(FEATURE_COLS, outlier_counts, color='tomato', edgecolor='black')
        plt.title('Outlier Count per Feature (Z-Score |z| > 3)')
        plt.xlabel('Feature')
        plt.ylabel('Number of Outliers')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_plot('05_outlier_counts_zscore.png')

    except Exception as e:
        print(f"    [ERROR] Plot 5 failed: {e}")


def plot_pairplot(df):
    """Plot 6 — Seaborn pairplot of the 5 most correlated features."""
    try:
        # Pick 5 features with highest average absolute correlation
        avg_corr = df[FEATURE_COLS].corr().abs().mean().sort_values(ascending=False)
        top5 = avg_corr.head(5).index.tolist()
        print(f"    Top 5 features for pairplot: {top5}")

        plot_df = df[top5 + ['fault_type']].copy()
        plot_df['fault_type'] = plot_df['fault_type'].astype(str)

        pair_fig = sns.pairplot(plot_df, hue='fault_type', diag_kind='kde',
                                palette='tab10', plot_kws={'alpha': 0.4})
        pair_fig.fig.suptitle('Pairplot — Top 5 Features', y=1.02, fontsize=13)

        import os
        path = os.path.join(PLOTS_DIR, '06_pairplot_top5_features.png')
        pair_fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close('all')
        print(f"    Saved → {path}")

    except Exception as e:
        print(f"    [ERROR] Plot 6 failed: {e}")


def plot_classwise_means(df):
    """Plot 7 — Grouped bar chart of per-class feature means."""
    try:
        class_means = df.groupby('fault_type')[FEATURE_COLS].mean()
        cmap = matplotlib.colormaps.get_cmap('tab10')
        x = np.arange(len(FEATURE_COLS))
        bar_width = 0.11

        plt.figure(figsize=(20, 7))
        for i, ft in enumerate(sorted(FAULT_NAMES.keys())):
            offset = (i - 3) * bar_width
            plt.bar(x + offset, class_means.loc[ft], width=bar_width,
                    label=FAULT_NAMES[ft], color=cmap(i / 9), alpha=0.85)

        plt.xticks(x, FEATURE_COLS, rotation=45, ha='right', fontsize=8)
        plt.title('Class-wise Feature Means')
        plt.xlabel('Feature')
        plt.ylabel('Mean Value')
        plt.legend(title='Fault Type', fontsize=8)
        plt.tight_layout()
        save_plot('07_classwise_feature_means.png')

    except Exception as e:
        print(f"    [ERROR] Plot 7 failed: {e}")


def run_all_plots(df):
    """Run all 7 plots in sequence. One failure does not stop the others."""
    print("\n[Plot 1] Class distribution bar chart...")
    plot_class_distribution(df)

    print("[Plot 2] Box plots per feature...")
    plot_boxplots(df)

    print("[Plot 3] Correlation heatmap...")
    plot_correlation_heatmap(df)

    print("[Plot 4] Feature histograms...")
    plot_histograms(df)

    print("[Plot 5] Outlier counts (Z-score)...")
    plot_outlier_counts(df)

    print("[Plot 6] Pairplot of top 5 features...")
    plot_pairplot(df)

    print("[Plot 7] Class-wise feature means...")
    plot_classwise_means(df)

    print("[OK] All plots complete.")
