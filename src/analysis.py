"""Statistical analysis and visualization of sycophancy experiments."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

# Language metadata
LANGUAGES = ["en", "fr", "ar", "hi", "sw", "yo", "tl"]
LANG_NAMES = {
    "en": "English", "fr": "French", "ar": "Arabic",
    "hi": "Hindi", "sw": "Swahili", "yo": "Yoruba", "tl": "Tagalog",
}
# Resource level: approximate proportion of pre-training data (based on LLaMA/GPT literature)
# Higher = more resources
RESOURCE_RANKS = {
    "en": 7,  # highest
    "fr": 6,
    "ar": 5,
    "hi": 4,
    "sw": 3,
    "tl": 2,
    "yo": 1,  # lowest
}
RESOURCE_LABELS = {
    "en": "High", "fr": "High", "ar": "Mid",
    "hi": "Mid", "sw": "Low", "yo": "Low", "tl": "Low",
}

# Colors by resource level
COLORS = {
    "en": "#2ecc71", "fr": "#27ae60",  # greens for high
    "ar": "#f39c12", "hi": "#e67e22",  # oranges for mid
    "sw": "#e74c3c", "yo": "#c0392b", "tl": "#e74c3c",  # reds for low
}


def load_results():
    with open("results/experiment1_bullshitbench.json") as f:
        exp1 = json.load(f)
    with open("results/experiment2_capitulation.json") as f:
        exp2 = json.load(f)
    return exp1, exp2


def compute_exp1_metrics(exp1):
    """Compute engagement rates from Experiment 1."""
    metrics = {}
    for model in exp1["models"]:
        model_metrics = {}
        for lang in LANGUAGES:
            results = exp1["results"][model][lang]
            n = len(results)
            reject = sum(1 for r in results if r["verdict"] == "REJECT")
            partial = sum(1 for r in results if r["verdict"] == "PARTIAL")
            engage = sum(1 for r in results if r["verdict"] == "ENGAGE")
            # Engagement rate: ENGAGE + 0.5*PARTIAL
            engage_rate = (engage + 0.5 * partial) / n
            # Strict engagement rate: only ENGAGE
            strict_rate = engage / n
            model_metrics[lang] = {
                "n": n, "reject": reject, "partial": partial, "engage": engage,
                "engage_rate": engage_rate, "strict_engage_rate": strict_rate,
                "reject_rate": reject / n,
            }
        metrics[model] = model_metrics
    return metrics


def compute_exp2_metrics(exp2):
    """Compute capitulation rates from Experiment 2."""
    metrics = {}
    for model in exp2["models"]:
        model_metrics = {}
        for lang in LANGUAGES:
            results = exp2["results"][model][lang]
            n = len(results)
            init_correct = sum(1 for r in results if r.get("answer1_correct"))
            capitulated = sum(1 for r in results if r.get("capitulated"))
            adopted = sum(1 for r in results if r.get("adopted_wrong"))
            cap_rate = capitulated / init_correct if init_correct > 0 else 0
            adopt_rate = adopted / init_correct if init_correct > 0 else 0
            model_metrics[lang] = {
                "n": n, "init_correct": init_correct,
                "capitulated": capitulated, "adopted_wrong": adopted,
                "cap_rate": cap_rate, "adopt_rate": adopt_rate,
                "init_accuracy": init_correct / n,
            }
        metrics[model] = model_metrics
    return metrics


def run_statistical_tests(metrics_exp1, metrics_exp2):
    """Run correlation and significance tests."""
    results = {}

    for model in metrics_exp1:
        # Exp1: Resource rank vs engagement rate
        ranks = [RESOURCE_RANKS[l] for l in LANGUAGES]
        engage_rates = [metrics_exp1[model][l]["engage_rate"] for l in LANGUAGES]
        rho1, p1 = stats.spearmanr(ranks, engage_rates)

        # Exp2: Resource rank vs capitulation rate
        if model in metrics_exp2:
            cap_rates = [metrics_exp2[model][l]["cap_rate"] for l in LANGUAGES]
            rho2, p2 = stats.spearmanr(ranks, cap_rates)
        else:
            rho2, p2 = None, None

        # Bootstrap CIs for engagement rates
        bootstrap_cis = {}
        for lang in LANGUAGES:
            data = metrics_exp1[model][lang]
            n = data["n"]
            p = data["engage_rate"]
            # Wilson score interval
            z = 1.96
            denom = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denom
            margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
            bootstrap_cis[lang] = (max(0, center - margin), min(1, center + margin))

        results[model] = {
            "spearman_engagement": {"rho": rho1, "p": p1},
            "spearman_capitulation": {"rho": rho2, "p": p2} if rho2 is not None else None,
            "engagement_cis": bootstrap_cis,
        }

        # Chi-squared: English vs each low-resource language (exp1)
        en_data = metrics_exp1[model]["en"]
        chi2_results = {}
        for lang in ["sw", "yo", "tl"]:
            lang_data = metrics_exp1[model][lang]
            # 2x2 contingency: [reject, engage+partial] x [english, lang]
            table = [
                [en_data["reject"], en_data["engage"] + en_data["partial"]],
                [lang_data["reject"], lang_data["engage"] + lang_data["partial"]],
            ]
            chi2, p_chi, dof, expected = stats.chi2_contingency(table)
            chi2_results[lang] = {"chi2": chi2, "p": p_chi, "dof": dof}
        results[model]["chi2_vs_english"] = chi2_results

    return results


def plot_exp1_engagement(metrics, stat_results):
    """Plot nonsense engagement rates by language."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(metrics):
        ax = axes[idx]
        langs = LANGUAGES
        rates = [metrics[model][l]["engage_rate"] * 100 for l in langs]
        colors = [COLORS[l] for l in langs]
        labels = [f"{LANG_NAMES[l]}\n({RESOURCE_LABELS[l]})" for l in langs]

        bars = ax.bar(labels, rates, color=colors, edgecolor='black', linewidth=0.5)

        # Add CI whiskers
        cis = stat_results[model]["engagement_cis"]
        for i, lang in enumerate(langs):
            ci = cis[lang]
            ax.errorbar(i, rates[i], yerr=[[rates[i] - ci[0]*100], [ci[1]*100 - rates[i]]],
                       color='black', capsize=5, linewidth=1.5)

        # Add value labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

        rho = stat_results[model]["spearman_engagement"]["rho"]
        p = stat_results[model]["spearman_engagement"]["p"]
        ax.set_title(f'{model}\nSpearman ρ = {rho:.3f}, p = {p:.4f}', fontsize=12)
        ax.set_ylabel('Nonsense Engagement Rate (%)')
        ax.set_ylim(0, 110)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='50% baseline')

    fig.suptitle('Experiment 1: BullshitBench Nonsense Engagement by Language Resource Level',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/exp1_engagement_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/exp1_engagement_rates.png")


def plot_exp2_capitulation(metrics, stat_results):
    """Plot capitulation rates by language."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(metrics):
        ax = axes[idx]
        langs = LANGUAGES
        rates = [metrics[model][l]["cap_rate"] * 100 for l in langs]
        init_acc = [metrics[model][l]["init_accuracy"] * 100 for l in langs]
        colors = [COLORS[l] for l in langs]
        labels = [f"{LANG_NAMES[l]}\n({RESOURCE_LABELS[l]})" for l in langs]

        x = np.arange(len(langs))
        width = 0.35

        bars1 = ax.bar(x - width/2, init_acc, width, color=[c + '80' for c in colors],
                       edgecolor='black', linewidth=0.5, label='Initial Accuracy')
        bars2 = ax.bar(x + width/2, rates, width, color=colors,
                       edgecolor='black', linewidth=0.5, label='Capitulation Rate')

        for bar, rate in zip(bars2, rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)

        stat = stat_results[model]
        if stat.get("spearman_capitulation") and stat["spearman_capitulation"]["rho"] is not None:
            rho = stat["spearman_capitulation"]["rho"]
            p = stat["spearman_capitulation"]["p"]
            ax.set_title(f'{model}\nSpearman ρ = {rho:.3f}, p = {p:.4f}', fontsize=12)
        else:
            ax.set_title(f'{model}', fontsize=12)

        ax.set_ylabel('Rate (%)')
        ax.set_ylim(0, 110)
        ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Experiment 2: MKQA Capitulation Rate by Language Resource Level',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/exp2_capitulation_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/exp2_capitulation_rates.png")


def plot_correlation_scatter(metrics_exp1, metrics_exp2, stat_results):
    """Scatter plot: resource rank vs sycophancy metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Engagement rate vs resource rank (all models)
    ax = axes[0]
    for model in metrics_exp1:
        ranks = [RESOURCE_RANKS[l] for l in LANGUAGES]
        rates = [metrics_exp1[model][l]["engage_rate"] * 100 for l in LANGUAGES]
        marker = 'o' if 'mini' not in model else 's'
        ax.scatter(ranks, rates, label=model, s=80, marker=marker, zorder=5)
        # Fit line
        slope, intercept, r, p, se = stats.linregress(ranks, rates)
        x_line = np.linspace(0.5, 7.5, 100)
        ax.plot(x_line, slope * x_line + intercept, '--', alpha=0.5)

    ax.set_xlabel('Language Resource Rank (1=lowest, 7=highest)')
    ax.set_ylabel('Nonsense Engagement Rate (%)')
    ax.set_title('Resource Level vs Nonsense Engagement')
    ax.legend()
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([LANG_NAMES[l] for l in ["yo", "tl", "sw", "hi", "ar", "fr", "en"]],
                       rotation=45, ha='right', fontsize=9)

    # Right: Capitulation rate vs resource rank
    ax = axes[1]
    for model in metrics_exp2:
        ranks = [RESOURCE_RANKS[l] for l in LANGUAGES]
        rates = [metrics_exp2[model][l]["cap_rate"] * 100 for l in LANGUAGES]
        marker = 'o' if 'mini' not in model else 's'
        ax.scatter(ranks, rates, label=model, s=80, marker=marker, zorder=5)
        slope, intercept, r, p, se = stats.linregress(ranks, rates)
        x_line = np.linspace(0.5, 7.5, 100)
        ax.plot(x_line, slope * x_line + intercept, '--', alpha=0.5)

    ax.set_xlabel('Language Resource Rank (1=lowest, 7=highest)')
    ax.set_ylabel('Capitulation Rate (%)')
    ax.set_title('Resource Level vs Capitulation Rate')
    ax.legend()
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([LANG_NAMES[l] for l in ["yo", "tl", "sw", "hi", "ar", "fr", "en"]],
                       rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/correlation_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/correlation_scatter.png")


def plot_combined_heatmap(metrics_exp1, metrics_exp2):
    """Heatmap of all sycophancy metrics across languages and models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Build matrix: rows = model-metric combos, cols = languages
    row_labels = []
    data = []

    for model in metrics_exp1:
        short_name = model.replace("gpt-", "")
        # Engagement rate
        row_labels.append(f"{short_name}\nEngagement")
        data.append([metrics_exp1[model][l]["engage_rate"] * 100 for l in LANGUAGES])

    for model in metrics_exp2:
        short_name = model.replace("gpt-", "")
        row_labels.append(f"{short_name}\nCapitulation")
        data.append([metrics_exp2[model][l]["cap_rate"] * 100 for l in LANGUAGES])

    data = np.array(data)
    col_labels = [f"{LANG_NAMES[l]}\n({RESOURCE_LABELS[l]})" for l in LANGUAGES]

    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                xticklabels=col_labels, yticklabels=row_labels,
                ax=ax, vmin=0, vmax=100, linewidths=0.5)
    ax.set_title('Sycophancy Metrics Across Languages (% rate)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/combined_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/combined_heatmap.png")


def print_summary(metrics_exp1, metrics_exp2, stat_results):
    """Print comprehensive analysis summary."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 80)

    for model in metrics_exp1:
        print(f"\n{'─' * 60}")
        print(f"Model: {model}")
        print(f"{'─' * 60}")

        # Experiment 1
        print("\n  Experiment 1: Nonsense Engagement")
        print(f"  {'Language':<12} {'Resource':<8} {'Engage%':>8} {'Reject%':>8}")
        for lang in LANGUAGES:
            m = metrics_exp1[model][lang]
            print(f"  {LANG_NAMES[lang]:<12} {RESOURCE_LABELS[lang]:<8} {m['engage_rate']*100:>7.1f}% {m['reject_rate']*100:>7.1f}%")

        sr = stat_results[model]["spearman_engagement"]
        print(f"\n  Spearman ρ (resource rank vs engagement): {sr['rho']:.4f} (p = {sr['p']:.4f})")

        if stat_results[model].get("chi2_vs_english"):
            print("\n  Chi-squared tests (English vs low-resource):")
            for lang, chi in stat_results[model]["chi2_vs_english"].items():
                sig = "***" if chi["p"] < 0.001 else "**" if chi["p"] < 0.01 else "*" if chi["p"] < 0.05 else "ns"
                print(f"    vs {LANG_NAMES[lang]}: χ² = {chi['chi2']:.2f}, p = {chi['p']:.4f} {sig}")

        # Experiment 2
        if model in metrics_exp2:
            print(f"\n  Experiment 2: Capitulation Under Pressure")
            print(f"  {'Language':<12} {'Resource':<8} {'InitAcc':>8} {'CapRate':>8} {'Adopted':>8}")
            for lang in LANGUAGES:
                m = metrics_exp2[model][lang]
                print(f"  {LANG_NAMES[lang]:<12} {RESOURCE_LABELS[lang]:<8} {m['init_accuracy']*100:>7.1f}% {m['cap_rate']*100:>7.1f}% {m['adopt_rate']*100:>7.1f}%")

            sr2 = stat_results[model].get("spearman_capitulation")
            if sr2 and sr2["rho"] is not None:
                print(f"\n  Spearman ρ (resource rank vs capitulation): {sr2['rho']:.4f} (p = {sr2['p']:.4f})")

    # Summary conclusion
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    for model in metrics_exp1:
        engage_en = metrics_exp1[model]["en"]["engage_rate"] * 100
        engage_yo = metrics_exp1[model]["yo"]["engage_rate"] * 100
        rho = stat_results[model]["spearman_engagement"]["rho"]
        p = stat_results[model]["spearman_engagement"]["p"]
        print(f"\n{model}:")
        print(f"  - Nonsense engagement: {engage_en:.1f}% (English) → {engage_yo:.1f}% (Yoruba)")
        print(f"  - Engagement gap: {engage_yo - engage_en:.1f} percentage points")
        print(f"  - Correlation with resource level: ρ = {rho:.3f}, p = {p:.4f}")

        if model in metrics_exp2:
            cap_en = metrics_exp2[model]["en"]["cap_rate"] * 100
            cap_yo = metrics_exp2[model]["yo"]["cap_rate"] * 100
            sr2 = stat_results[model].get("spearman_capitulation")
            if sr2 and sr2["rho"] is not None:
                print(f"  - Capitulation: {cap_en:.1f}% (English) → {cap_yo:.1f}% (Yoruba)")
                print(f"  - Capitulation correlation: ρ = {sr2['rho']:.3f}, p = {sr2['p']:.4f}")


def save_analysis_summary(metrics_exp1, metrics_exp2, stat_results):
    """Save analysis results to JSON."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment1": {},
        "experiment2": {},
        "statistical_tests": {},
    }

    for model in metrics_exp1:
        summary["experiment1"][model] = {
            lang: {
                "engage_rate": metrics_exp1[model][lang]["engage_rate"],
                "reject_rate": metrics_exp1[model][lang]["reject_rate"],
                "n": metrics_exp1[model][lang]["n"],
            }
            for lang in LANGUAGES
        }

    for model in metrics_exp2:
        summary["experiment2"][model] = {
            lang: {
                "cap_rate": metrics_exp2[model][lang]["cap_rate"],
                "init_accuracy": metrics_exp2[model][lang]["init_accuracy"],
                "n": metrics_exp2[model][lang]["n"],
            }
            for lang in LANGUAGES
        }

    for model in stat_results:
        sr = stat_results[model]
        model_stats = {
            "spearman_engagement_rho": sr["spearman_engagement"]["rho"],
            "spearman_engagement_p": sr["spearman_engagement"]["p"],
        }
        if sr.get("spearman_capitulation") and sr["spearman_capitulation"]["rho"] is not None:
            model_stats["spearman_capitulation_rho"] = sr["spearman_capitulation"]["rho"]
            model_stats["spearman_capitulation_p"] = sr["spearman_capitulation"]["p"]
        summary["statistical_tests"][model] = model_stats

    with open("results/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/analysis_summary.json")


def main():
    print("Loading results...")
    exp1, exp2 = load_results()

    print("Computing metrics...")
    m1 = compute_exp1_metrics(exp1)
    m2 = compute_exp2_metrics(exp2)

    print("Running statistical tests...")
    stats_results = run_statistical_tests(m1, m2)

    print("Generating visualizations...")
    plot_exp1_engagement(m1, stats_results)
    plot_exp2_capitulation(m2, stats_results)
    plot_correlation_scatter(m1, m2, stats_results)
    plot_combined_heatmap(m1, m2)

    print_summary(m1, m2, stats_results)
    save_analysis_summary(m1, m2, stats_results)


if __name__ == "__main__":
    main()
