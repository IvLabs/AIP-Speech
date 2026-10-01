#!/usr/bin/env python3
"""
09_make_figures.py — Stage 9: the paper's figures and appendix tables.

Generates the figures in results/figures/ and the tables in results/tables/:
  - Reads the acoustic descriptors of every background recording (Stage 4).
  - Averages each task's degradation per background over 10, 5, 0 dB SNR and
    the five models (ASR: ΔWER; KWS: accuracy drop).
  - Computes Spearman correlations of each descriptor with that degradation,
    pooled over the models and for each model on its own.
  - Selects the top 3 descriptors driven by data (largest mean |r_s|).
  - Figure 2: regression lines for the top 3 descriptors (ASR + KWS, raw degradation).
  - Figure 3: Spearman correlation heatmap (tasks x descriptors).
  - Figure 4: per-background degradation ranking.
  - Figure 5: ΔWER across the accent groups of the Common Voice utterances.
  - Figure 6: prompt-steering Residual Effect Ratio per model.
  - Figure 7: WER and KWS accuracy across SNRs, side by side.
  - Saves the correlation, descriptor, accent and RER tables of the Appendix.

Output files → paper:
  results/figures/figure2_descriptor_regressions.png    Figure 2
  results/figures/figure3_descriptor_correlations.png   Figure 3
  results/figures/figure4_background_ranking.png        Figure 4
  results/figures/figure5_accent_degradation.png        Figure 5
  results/figures/figure6_prompt_steering_rer.png       Figure 6
  results/figures/figure7_snr_performance.png           Figure 7
  results/tables/table2_descriptor_correlations.csv     Table 2
  results/tables/table3_descriptor_values.csv           Table 3
  results/tables/table4_accent_dwer.csv                 Table 4
  results/tables/table5_prompt_steering_rer.csv         Table 5
Figure 1 (pipeline diagram) is assets/figure1.png; Table 1 (the 20 background
recordings) is BATTERY_SPEC in 03_curate_backgrounds.py.

Inputs: results/scores/<task>.csv from 08_score_outputs.py and
descriptors/acoustic_descriptors.csv from 04_extract_descriptors.py.

Usage:
  python scripts/09_make_figures.py
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

DESCRIPTORS_CSV = ROOT / "descriptors" / "acoustic_descriptors.csv"
SCORES_DIR = ROOT / "results" / "scores"

MODELS = ['gemma3n_e4b', 'phi4_multimodal', 'qwen25_omni_3b', 'qwen25_omni_7b', 'qwen2_audio_7b']

# model names in the paper's tables
TABLE_NAMES = {
    'gemma3n_e4b': 'Gemma 3n',
    'phi4_multimodal': 'Phi-4-MM',
    'qwen25_omni_3b': 'Qwen2.5-Omni-3B',
    'qwen25_omni_7b': 'Qwen2.5-Omni-7B',
    'qwen2_audio_7b': 'Qwen2-Audio',
}

# Table 3 columns: Stage 4 descriptor → paper header, in the paper's order
TABLE3_COLUMNS = {
    'Zero Crossing Rate (ZCR)': 'ZCR',
    'Spectral Centroid (Hz)': 'Centroid',
    'Spectral Bandwidth (Hz)': 'Bandwidth',
    'Spectral Rolloff (Hz)': 'Rolloff',
    'Silero VAD Speech Fraction': 'Speech',
    'Spectral Flux': 'Flux',
    'Spectral Flatness': 'Flatness',
    'Spectral Contrast (dB)': 'Contrast',
    'RMS Energy Variance': 'RMS Var.',
}

# Stage 2 labels every LibriSpeech utterance with this accent; the accent
# analysis uses the accent-labelled Common Voice utterances only
LIBRISPEECH_ACCENT = 'american_english'

TASK_COLORS = {
    'ASR': '#4C72B0',
    'KWS': '#DD8452',
}

def set_paper_style():
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'legend.title_fontsize': 17,
        'figure.titlesize': 22,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'font.family': 'sans-serif',
        'figure.dpi': 300
    })
    sns.set_style("whitegrid")


def task_degradations(asr, kws):
    """Per-background degradation averaged over the noisy rows (10, 5, 0 dB SNR)."""
    asr_noisy = asr[asr['condition'] == 'noisy']
    kws_noisy = kws[kws['condition'] == 'noisy']

    asr_deg = asr_noisy.groupby('bg_id')['dwer'].mean()

    # flip dacc so that, as with ΔWER, a larger number means more damage
    kws_deg = -kws_noisy.groupby('bg_id')['dacc'].mean()

    deg_df = pd.DataFrame({'ASR': asr_deg, 'KWS': kws_deg})
    return deg_df


def compute_correlations(desc_df, deg_df):
    merged = desc_df.join(deg_df).dropna()
    corrs = []
    for col in desc_df.columns:
        r_asr, p_asr = stats.spearmanr(merged[col], merged['ASR'])
        r_kws, p_kws = stats.spearmanr(merged[col], merged['KWS'])
        mean_r = np.mean([r_asr, r_kws])
        mean_abs_r = np.mean([abs(r_asr), abs(r_kws)])
        corrs.append({
            'Descriptor': col,
            'ASR (r_s)': round(r_asr, 3), 'ASR (p)': round(p_asr, 3),
            'KWS (r_s)': round(r_kws, 3), 'KWS (p)': round(p_kws, 3),
            'Mean r_s': round(mean_r, 3),
            'Mean |r_s|': round(mean_abs_r, 3)
        })
    corr_df = pd.DataFrame(corrs).sort_values('Mean |r_s|', ascending=False)
    return merged, corr_df


def background_ranking(deg_df):
    """Backgrounds by mean degradation over ASR and KWS, least harmful first."""
    deg = deg_df[['ASR', 'KWS']].copy()
    deg['Mean_Degradation'] = deg.mean(axis=1)
    return deg.sort_values('Mean_Degradation', ascending=True).reset_index()


def save_table2_descriptor_correlations(desc_df, corr_df, asr, kws):
    """Table 2: r_s pooled over the models (the Figure 3 values) and per model."""
    columns = {'Pooled (All 5)': corr_df.set_index('Descriptor')}
    for m in reversed(MODELS):  # the paper's column order
        deg_df = task_degradations(asr[asr['model'] == m], kws[kws['model'] == m])
        columns[TABLE_NAMES[m]] = compute_correlations(desc_df, deg_df)[1].set_index('Descriptor')

    rows = []
    for desc in desc_df.columns:
        for task in ['ASR', 'KWS']:
            # the paper's row labels leave out the units
            row = {'Descriptor': desc.split(' (')[0], 'Task': task}
            for name, corr in columns.items():
                row[name] = corr.at[desc, f'{task} (r_s)']
            rows.append(row)
    out = TABLE_DIR / "table2_descriptor_correlations.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format='%.3f')
    print(f"Saved {out.name}")


def save_table3_descriptor_values(desc_df, ranking):
    """Table 3: descriptor values per background, in the Figure 4 order."""
    table = desc_df.loc[ranking['bg_id'], list(TABLE3_COLUMNS)].rename(columns=TABLE3_COLUMNS)
    out = TABLE_DIR / "table3_descriptor_values.csv"
    table.rename_axis('Background').to_csv(out)
    print(f"Saved {out.name}")


def generate_figure2_descriptor_regressions(merged, top_descriptors):
    print("Generating Figure 2: 1x3 Scatter + Regression plots for top 3 descriptors...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for idx, desc in enumerate(top_descriptors):
        ax = axes[idx]
        for task in ['ASR', 'KWS']:
            sns.regplot(
                data=merged,
                x=desc,
                y=task,
                label=task,
                color=TASK_COLORS[task],
                ax=ax,
                scatter_kws={'s': 65, 'alpha': 0.85},
                line_kws={'linewidth': 3.0},
                seed=0  # fixed bootstrap for the confidence bands
            )
        ax.set_title(f"({chr(97+idx)}) {desc} vs Degradation", pad=15, fontsize=18, fontweight='bold')
        ax.set_xlabel(desc, fontsize=17, labelpad=10)
        ax.set_ylabel("Raw Degradation Value", fontsize=17, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(title="Task", loc='upper left', fontsize=15, title_fontsize=16, frameon=True, framealpha=0.95)

    plt.tight_layout()
    out = FIG_DIR / "figure2_descriptor_regressions.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


def generate_figure3_descriptor_correlations(corr_df):
    print("Generating Figure 3: Spearman correlation heatmap...")
    heatmap_df = corr_df.set_index('Descriptor')[['ASR (r_s)', 'KWS (r_s)']].copy()
    heatmap_df.columns = ['ASR', 'KWS']
    heatmap_df = heatmap_df.T

    # drawn in matplotlib's default style, not the seaborn one of the other figures
    with plt.style.context('default'):
        plt.figure(figsize=(15.4, 6))
        ax = sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 17},
            cmap="vlag",
            center=0,
            # a narrower colorbar slot than the default 0.15, which would
            # leave a blank strip at the right edge
            cbar_kws={'label': 'Spearman Correlation', 'fraction': 0.12},
            linewidths=1.2,
            linecolor='white'
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label('Spearman Correlation', fontsize=17, labelpad=12)

        # bold the strongest correlation of each task
        max_col_per_row = heatmap_df.abs().idxmax(axis=1)
        for r_idx, row_name in enumerate(heatmap_df.index):
            c_idx = heatmap_df.columns.get_loc(max_col_per_row[row_name])
            ax.texts[r_idx * len(heatmap_df.columns) + c_idx].set_weight('bold')

        plt.title("Spearman Correlation between Acoustic Descriptors and Task Degradations", pad=18, fontsize=19, fontweight='bold')
        plt.ylabel("Evaluation Task", labelpad=12, fontsize=17, fontweight='bold')
        plt.xlabel("Acoustic Descriptor", labelpad=12, fontsize=17, fontweight='bold')
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=17, labelrotation=0)
        plt.xticks(rotation=40, ha='right')
        plt.tight_layout()
        out = FIG_DIR / "figure3_descriptor_correlations.png"
        plt.savefig(out, dpi=300)
        plt.close()
    print(f"Saved {out.name}")


def generate_figure4_background_ranking(ranking):
    print("Generating Figure 4: Per-background mean degradation ranking horizontal bar plot...")
    fig = plt.figure(figsize=(9.5, 9.5))
    ax = sns.barplot(
        data=ranking,
        y='bg_id',
        x='Mean_Degradation',
        color='#4C72B0'
    )
    fig.suptitle("Per-Background Mean Degradation Ranking\n(Averaged across ASR and KWS)", y=0.96, fontsize=18, fontweight='bold')
    plt.xlabel("Mean Raw Degradation (Higher = Worse)", fontsize=17, labelpad=12)
    plt.ylabel("Background Sound Class", fontsize=17, labelpad=12)
    ax.tick_params(axis='both', labelsize=15)
    plt.subplots_adjust(top=0.86, bottom=0.10, left=0.28, right=0.94)
    out = FIG_DIR / "figure4_background_ranking.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


def generate_figure5_accent_degradation():
    print("Generating Figure 5: Demographic robustness across accents line plot...")
    asr_path = SCORES_DIR / "asr.csv"
    if not asr_path.exists():
        print(f"ASR scores not found at {asr_path}, skipping Figure 5.")
        return

    asr_df = pd.read_csv(asr_path)
    df = asr_df[(asr_df["condition"] == "noisy") & asr_df["accent"].notnull()
                & (asr_df["accent"] != LIBRISPEECH_ACCENT)].copy()
    if df.empty:
        print("No noisy Common Voice accent data found in ASR scores, skipping Figure 5.")
        return

    accent_clean_map = {
        'United States English': 'US English',
        'India and South Asia (India, Pakistan, Sri Lanka)': 'India & S. Asia',
        'England English': 'England English',
        'Canadian English': 'Canadian English',
        'West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)': 'West Indies & Bermuda',
        'Scottish English': 'Scottish English',
        'Irish English': 'Irish English',
        'Australian English': 'Australian English',
        'Southern African (South Africa, Zimbabwe, Namibia)': 'Southern African',
        'Hong Kong English': 'Hong Kong English',
        'New Zealand English': 'New Zealand English'
    }
    df['accent_clean'] = df['accent'].map(accent_clean_map).fillna(df['accent'])

    group_df = df.groupby(['model', 'accent_clean'])['dwer'].mean().reset_index()

    acc_csv = TABLE_DIR / "table4_accent_dwer.csv"
    table = group_df.pivot(index='accent_clean', columns='model', values='dwer').rename(columns=TABLE_NAMES)
    table.rename_axis(index='Accent Group', columns=None).round(4).to_csv(acc_csv, float_format='%.4f')
    print(f"Saved {acc_csv.name}")

    # line and legend order of the paper's figure
    model_order = ['gemma3n_e4b', 'qwen25_omni_3b', 'qwen25_omni_7b', 'qwen2_audio_7b', 'phi4_multimodal']
    fig, ax = plt.subplots(figsize=(11, 7.2))

    sns.lineplot(
        data=group_df,
        x='accent_clean',
        y='dwer',
        hue='model',
        style='model',
        hue_order=model_order,
        style_order=model_order,
        markers=True,
        dashes=False,
        linewidth=3.5,
        markersize=12,
        ax=ax
    )

    fig.suptitle("ASR: Mean Performance Degradation (ΔWER)\nacross Accent Subgroups", fontsize=20, fontweight='bold', y=0.97)
    plt.ylabel("Mean ΔWER (Noisy - Clean)", fontsize=19, labelpad=12)
    plt.xlabel("Accent Subgroup", fontsize=19, labelpad=12)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=17)
    plt.xticks(rotation=40, ha='right')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=14, frameon=True, framealpha=0.95)
    plt.subplots_adjust(top=0.78, bottom=0.30, left=0.11, right=0.96)
    out = FIG_DIR / "figure5_accent_degradation.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


def generate_figure6_prompt_steering_rer():
    print("Generating Figure 6: Prompt steering Residual Effect Ratio (RER) bar plot...")

    def load(task):
        path = SCORES_DIR / f"{task}.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def model_rows(df, m):
        return df[df['model'] == m] if not df.empty else df

    asr_base, asr_steer = load("asr"), load("asr_steer_p5")
    kws_base, kws_steer = load("kws"), load("kws_steer_p5")

    records = []

    for m in MODELS:
        # ASR: mean paired ΔWER under the steering prompt over the baseline one
        base, steer = model_rows(asr_base, m), model_rows(asr_steer, m)
        if base.empty or steer.empty:
            print(f"Skipping ASR RER for {m}: baseline or steering scores missing.")
        else:
            dwer_base = base[base['condition'] == 'noisy']['dwer']
            dwer_steer = steer[steer['condition'] == 'noisy']['dwer']
            rer_asr = dwer_steer.mean() / (dwer_base.mean() + 1e-9)
            records.append({'Model': m, 'Task': 'ASR', 'RER': rer_asr})

        # KWS: clean-to-noisy accuracy drop under the steering prompt over the baseline one
        base, steer = model_rows(kws_base, m), model_rows(kws_steer, m)
        if base.empty or steer.empty:
            print(f"Skipping KWS RER for {m}: baseline or steering scores missing.")
        else:
            dacc_base = (base[base['condition'] == 'clean']['correct'].mean()
                         - base[base['condition'] == 'noisy']['correct'].mean())
            dacc_steer = (steer[steer['condition'] == 'clean']['correct'].mean()
                          - steer[steer['condition'] == 'noisy']['correct'].mean())
            rer_kws = dacc_steer / (dacc_base + 1e-9)
            # the ratio is meaningless when neither prompt moves accuracy by a
            # point; such cases are reported as 1.0
            if abs(dacc_base) < 0.01 and abs(dacc_steer) < 0.01:
                rer_kws = 1.0
            records.append({'Model': m, 'Task': 'KWS', 'RER': rer_kws})

    df_rer = pd.DataFrame(records)
    if df_rer.empty:
        return

    rer_csv = TABLE_DIR / "table5_prompt_steering_rer.csv"
    table = df_rer.pivot(index='Model', columns='Task', values='RER').rename(index=TABLE_NAMES)
    table.rename_axis(columns=None).round(3).to_csv(rer_csv, float_format='%.3f')
    print(f"Saved {rer_csv.name}")

    # drawn in matplotlib's default style, not the seaborn one of the other figures
    with plt.style.context('default'):
        plt.figure(figsize=(11, 6.5))
        ax = sns.barplot(
            data=df_rer,
            x='Model',
            y='RER',
            hue='Task',
            palette=TASK_COLORS
        )
        plt.axhline(1.0, color='red', linestyle='--', linewidth=1.8, label='Baseline (RER = 1.0)')
        plt.title("Prompt Effectiveness: Residual Effect Ratio (RER) by Model", pad=18, fontsize=18, fontweight='bold')
        plt.ylabel("Residual Effect Ratio (RER)", fontsize=16, labelpad=12)
        plt.xlabel("Model", fontsize=16, labelpad=12)
        ax.tick_params(axis='both', labelsize=15)
        plt.legend(loc='upper left', fontsize=14, frameon=True, framealpha=0.95)
        # headroom so the label of the tallest bar stays inside the axes; the
        # bottom is left to autoscale so that negative RERs are drawn too
        ax.set_ylim(top=max(df_rer['RER'].max(), 1.0) * 1.135)

        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f'{height:.2f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            xytext=(0, 4), textcoords='offset points',
                            fontsize=13)

        plt.tight_layout()
        out = FIG_DIR / "figure6_prompt_steering_rer.png"
        plt.savefig(out, dpi=300)
        plt.close()
    print(f"Saved {out.name}")


MODEL_NAMES_MAP = {
    'gemma3n_e4b': 'Gemma 3n E4B',
    'phi4_multimodal': 'Phi-4 Multimodal',
    'qwen25_omni_3b': 'Qwen2.5-Omni 3B',
    'qwen25_omni_7b': 'Qwen2.5-Omni 7B',
    'qwen2_audio_7b': 'Qwen2-Audio 7B'
}


def generate_figure7_snr_performance():
    print("Generating Figure 7: SNR Effect on Performance for ASR and KWS (Side-by-Side)...")
    asr_path = SCORES_DIR / "asr.csv"
    kws_path = SCORES_DIR / "kws.csv"

    if not asr_path.exists() or not kws_path.exists():
        print("ASR or KWS scores file missing, skipping Figure 7.")
        return

    asr_df = pd.read_csv(asr_path)
    kws_df = pd.read_csv(kws_path)

    snr_order = [99, 10, 5, 0]
    snr_labels = ['Clean', '10 dB', '5 dB', '0 dB']

    asr_grp = asr_df.groupby(['model', 'snr_db'])['wer'].mean().reset_index()

    if 'correct' in kws_df.columns:
        kws_grp = kws_df.groupby(['model', 'snr_db'])['correct'].mean().reset_index()
    else:
        kws_grp = pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    markers = {
        'gemma3n_e4b': 'o',
        'phi4_multimodal': 's',
        'qwen25_omni_3b': '^',
        'qwen25_omni_7b': 'D',
        'qwen2_audio_7b': 'v'
    }

    ax1 = axes[0]
    for model in MODELS:
        m_df = asr_grp[asr_grp['model'] == model].copy()
        if m_df.empty: continue
        m_df['snr_pos'] = m_df['snr_db'].map(lambda x: snr_order.index(x) if x in snr_order else -1)
        m_df = m_df.sort_values('snr_pos')

        display_name = MODEL_NAMES_MAP.get(model, model)
        ax1.plot(
            range(len(snr_labels)),
            m_df['wer'],
            marker=markers.get(model, 'o'),
            linewidth=3.0,
            markersize=10,
            label=display_name
        )

    ax1.set_xticks(range(len(snr_labels)))
    ax1.set_xticklabels(snr_labels, fontsize=16)
    ax1.set_title("(a) ASR Performance across SNRs (WER ↓)", pad=15, fontsize=18, fontweight='bold')
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=17, labelpad=10)
    ax1.set_ylabel("Word Error Rate (WER)", fontsize=17, labelpad=10)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.legend(title="Model", loc='upper left', fontsize=14, title_fontsize=15, frameon=True, framealpha=0.95)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = axes[1]
    if not kws_grp.empty:
        for model in MODELS:
            if model == 'qwen2_audio_7b':
                continue
            m_df = kws_grp[kws_grp['model'] == model].copy()
            if m_df.empty: continue
            m_df['snr_pos'] = m_df['snr_db'].map(lambda x: snr_order.index(x) if x in snr_order else -1)
            m_df = m_df.sort_values('snr_pos')

            display_name = MODEL_NAMES_MAP.get(model, model)
            ax2.plot(
                range(len(snr_labels)),
                m_df['correct'],
                marker=markers.get(model, 'o'),
                linewidth=3.0,
                markersize=10,
                label=display_name
            )

    ax2.set_xticks(range(len(snr_labels)))
    ax2.set_xticklabels(snr_labels, fontsize=16)
    ax2.set_title("(b) KWS Performance across SNRs (Accuracy ↑)", pad=15, fontsize=18, fontweight='bold')
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=17, labelpad=10)
    ax2.set_ylabel("Accuracy", fontsize=17, labelpad=10)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.legend(title="Model", loc='lower left', fontsize=14, title_fontsize=15, frameon=True, framealpha=0.95)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out = FIG_DIR / "figure7_snr_performance.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out.name}")


def main():
    ap = argparse.ArgumentParser(description="Stage 9 — Make the paper's figures and tables")
    ap.parse_args()

    set_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    desc_df = pd.read_csv(DESCRIPTORS_CSV, index_col='bg_id')

    print("Loading evaluation metrics averaged over 10, 5, 0 dB SNR...")
    asr = pd.read_csv(SCORES_DIR / "asr.csv")
    kws = pd.read_csv(SCORES_DIR / "kws.csv")
    deg_df = task_degradations(asr, kws)
    merged, corr_df = compute_correlations(desc_df, deg_df)
    save_table2_descriptor_correlations(desc_df, corr_df, asr, kws)

    ranking = background_ranking(deg_df)
    save_table3_descriptor_values(desc_df, ranking)

    # corr_df is already sorted by mean |r_s|
    top_3 = corr_df.head(3)['Descriptor'].tolist()
    print(f"\nTop 3 data-driven descriptors selected: {top_3}")

    generate_figure2_descriptor_regressions(merged, top_3)
    generate_figure3_descriptor_correlations(corr_df)
    generate_figure4_background_ranking(ranking)
    generate_figure5_accent_degradation()
    generate_figure6_prompt_steering_rer()
    generate_figure7_snr_performance()

    print(f"\nAll figures and tables successfully generated in: {FIG_DIR} and {TABLE_DIR}")

if __name__ == '__main__':
    main()
