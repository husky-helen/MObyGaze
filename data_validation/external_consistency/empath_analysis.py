"""
Empath lexical analysis of the MObyGaze dataset.

Reproduces the methodology of the "Empath analysis" paragraph of the HateMM
paper (Das et al., ICWSM 2023), adapted to MObyGaze:

  - Run Empath (Fast, Chen & Bernstein, 2016) on the `subtitle` text of each
    annotated segment, using a curated subset of Empath categories relevant
    to (self-)objectification (body/appearance, power/dominance, emotion,
    speech/communication) instead of the full 194 categories.
  - For each Empath category, compare the per-segment normalized Empath
    score between two groups using the Mann-Whitney U test (as in HateMM).
  - Report significance both as raw p-value stars (matching the paper:
    * p<0.01, ** p<0.001, *** p<0.0001) and as Benjamini-Hochberg FDR
    q-values, since we are testing ~80 categories at once.
  - Plot the top-N most significant categories as a grouped bar chart of
    mean normalized Empath value per group (Figure-3-style in HateMM).

Two group comparisons are produced:
  1. Speech concept present vs absent (concepts column contains 'Speech').
  2. Objectification level: 'Easy Negative' vs {'Hard Negative','Sure'}
     ('Not Sure' rows are discarded).

Usage:
    python empath_analysis.py --csv mobygaze_dataframe.csv --outdir results
"""

import argparse
import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

try:
    from empath import Empath
except ImportError:
    sys.exit(
        "The 'empath' package is required. Install it with:\n"
        "    pip install empath"
    )


# --------------------------------------------------------------------------
# Curated Empath categories relevant to objectification analysis
# (analogous to HateMM's manual pruning from 189 -> 70 categories, dropping
# topics irrelevant to the phenomenon under study, e.g. technology, sports,
# travel, business). Feel free to edit this list.
# --------------------------------------------------------------------------
CURATED_CATEGORIES = [
    # Body / appearance
    "attractive", "masculine", "feminine", "sexual", "body", "appearance",
    "beauty", "ugliness", "clothing", "fashion", "youth", "exotic",
    "shape_and_size", "lust", "strength", "weakness",
    # Power / dominance / agency
    "dominant_heirarchical", "dominant_personality", "power", "leader",
    "independence", "order", "competing", "achievement", "gain",
    "stealing", "deception", "trust", "negotiate",
    # Emotion
    "cheerfulness", "pride", "shame", "fear", "nervousness", "horror",
    "disgust", "anger", "rage", "sadness", "joy", "contentment",
    "positive_emotion", "negative_emotion", "envy", "surprise",
    "disappointment", "timidity", "irritability", "exasperation",
    "torment", "suffering", "optimism", "zest", "confusion", "love",
    "affection", "warmth", "sympathy",
    # Speech / communication / conflict
    "politeness", "ridicule", "swearing_terms",
    "aggression", "dispute", "fight", "violence", "kill", "injury",
    "pain", "hate", "neglect", "childish",
]

# Concepts considered "audio" in nature. A Speech-tagged segment is
# classified as "audio-only" if ALL its concepts fall in this set (i.e.
# Speech alone, or combined only with Voice and/or Soundtrack), and as
# "visual-associated" if at least one concept outside this set is present
# (e.g. Body, Clothing, Look, Appearance, Posture, Activities, ...).
AUDIO_CONCEPTS = {"Speech", "Voice", "Soundtrack"}

# --------------------------------------------------------------------------
# Empath categories specifically related to the "Speech" concept, derived
# from the MObyGaze thesaurus's Text/Speech sub-definitions: comments on
# appearance, judgement on body, comparison to object/animal, self-loathing,
# denial of consent, mansterrupting/mansplaining, stereotyping discourse,
# sexualization, ordering/patronizing/infantilizing.
# --------------------------------------------------------------------------
SPEECH_CATEGORIES = sorted(set([
    "appearance", "attractive", "beauty",          # comments on appearance
    "body", "shape_and_size", "family",                      # judgement on body
    "animal",                                       # comparison to object/animal
    "shame", "negative_emotion", "sadness",        # self-loathing
    "dominant_personality", "dominant_heirarchical", "ridicule",  # denial of
    # consent / not taken seriously / mansterrupting / mansplaining
    "masculine", "feminine",                       # stereotyping discourse
    "sexual", "lust",                              # sexualization
    "childish", "order",                           # ordering/patronizing/infantilizing
]))

# --------------------------------------------------------------------------
# Empath categories related to each VISUAL concept, derived from the
# MObyGaze thesaurus's Vision box sub-definitions. "Narratology" is
# excluded (not present in the thesaurus / not lexically mappable).
# --------------------------------------------------------------------------
VISUAL_CONCEPT_CATEGORIES = {
    # male-perspective framing, woman-animal parallel shots
    "Type of shot": ["masculine", "animal"],
    # male gaze, voyeurism, non-reciprocal gaze
    "Look": ["masculine", "sexual", "lust"],
    # nudity, body parts suggesting sex
    "Body": ["body", "sexual", "attractive", "appearance", "shape_and_size"],
    # seduction/sexuality gestures, submission/inaction/dependence, skipping
    "Posture": ["sexual", "lust", "weakness", "childish"],
    # revealing/wet/transparent, impractical, infantile clothing
    "Clothing": ["clothing", "fashion", "sexual", "childish"],
    # appearance/context discrepancy, heavy makeup/accessories
    "Appearance": ["appearance", "beauty", "fashion", "attractive", "ugliness"],
    # asymmetric/stereotyped emotion expression (crying vs. dispassionate)
    "Exp of emotion": ["sadness", "negative_emotion", "positive_emotion", "emotional"],
    # domestic activities, sex-related actions, submissive actions, violence
    "Activities": ["domestic_work", "sexual", "weakness", "violence"],
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")

    # concepts column is a stringified python list, e.g. "['Speech']"
    def parse_concepts(x):
        if pd.isna(x):
            return []
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else []
        except (ValueError, SyntaxError):
            return []

    df["concepts_list"] = df["concepts"].apply(parse_concepts)
    df["has_speech"] = df["concepts_list"].apply(lambda lst: "Speech" in lst)

    def classify_speech_type(lst):
        if "Speech" not in lst:
            return None
        return "audio" if set(lst).issubset(AUDIO_CONCEPTS) else "visual"

    df["speech_type"] = df["concepts_list"].apply(classify_speech_type)

    # keep only rows with actual subtitle text
    df["subtitle"] = df["subtitle"].astype(str)
    df = df[df["subtitle"].str.strip().str.len() > 0]
    df = df[df["subtitle"].str.lower() != "nan"]

    return df.reset_index(drop=True)



# --------------------------------------------------------------------------
# Empath scoring
# --------------------------------------------------------------------------
def score_empath(texts: pd.Series, categories: list) -> pd.DataFrame:
    lexicon = Empath()
    records = []
    for text in texts:
        scores = lexicon.analyze(text, categories=categories, normalize=True)
        if scores is None:
            scores = {c: 0.0 for c in categories}
        records.append(scores)
    return pd.DataFrame.from_records(records, columns=categories)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def p_to_stars(p: float) -> str:
    if p < 0.0001:
        return "***"
    elif p < 0.001:
        return "**"
    elif p < 0.01:
        return "*"
    return ""


def rank_biserial_effect_size(u_stat: float, n1: int, n2: int) -> float:
    """
    Rank-biserial correlation effect size for Mann-Whitney U, sign-aligned
    with `diff` (mean_group_a - mean_group_b): positive means group A tends
    to have higher values than group B, negative means the opposite.

    scipy.stats.mannwhitneyu(x, y) returns U for sample x (U grows toward
    n1*n2 when x tends to be larger than y). The conventional formula
    r = 1 - 2U/(n1*n2) therefore yields a NEGATIVE value when x > y, which
    is unintuitive next to `diff`. We flip the sign here so that
    positive = group A stochastically greater, matching `diff`.
    """
    return (2 * u_stat) / (n1 * n2) - 1


def compare_groups(
    scores: pd.DataFrame,
    group_mask: pd.Series,
    group_a_name: str,
    group_b_name: str,
) -> pd.DataFrame:
    """
    Run Mann-Whitney U test per Empath category between two groups.
    group_mask is True for group A, False for group B.
    """
    results = []
    a = scores[group_mask]
    b = scores[~group_mask]
    for cat in scores.columns:
        xa, xb = a[cat].values, b[cat].values
        mean_a, mean_b = xa.mean(), xb.mean()
        # Mann-Whitney requires some variance; skip degenerate all-zero cases
        if np.all(xa == 0) and np.all(xb == 0):
            u_stat, p_val = np.nan, 1.0
        else:
            try:
                u_stat, p_val = mannwhitneyu(xa, xb, alternative="two-sided")
            except ValueError:
                u_stat, p_val = np.nan, 1.0
        eff = (
            rank_biserial_effect_size(u_stat, len(xa), len(xb))
            if not np.isnan(u_stat)
            else np.nan
        )
        results.append(
            {
                "category": cat,
                f"mean_{group_a_name}": mean_a,
                f"mean_{group_b_name}": mean_b,
                "diff": mean_a - mean_b,
                "u_stat": u_stat,
                "p_value": p_val,
                "effect_size_rank_biserial": eff,
                "n_" + group_a_name: len(xa),
                "n_" + group_b_name: len(xb),
            }
        )

    res_df = pd.DataFrame(results)

    # Benjamini-Hochberg FDR correction across all tested categories
    valid = res_df["p_value"].notna()
    q_vals = np.full(len(res_df), np.nan)
    if valid.sum() > 0:
        _, q_corrected, _, _ = multipletests(
            res_df.loc[valid, "p_value"], method="fdr_bh"
        )
        q_vals[valid.values] = q_corrected
    res_df["q_value_fdr_bh"] = q_vals

    res_df["stars_raw_p"] = res_df["p_value"].apply(p_to_stars)
    res_df["stars_fdr_q"] = res_df["q_value_fdr_bh"].apply(
        lambda q: p_to_stars(q) if not np.isnan(q) else ""
    )

    res_df = res_df.sort_values(
        "effect_size_rank_biserial", ascending=False, na_position="last"
    ).reset_index(drop=True)
    return res_df


# --------------------------------------------------------------------------
# Plotting (HateMM Figure-3 style)
# --------------------------------------------------------------------------
def plot_top_categories(
    res_df: pd.DataFrame,
    group_a_name: str,
    group_b_name: str,
    title: str,
    outfile: Path,
    top_n: int = 15,
    use_fdr: bool = False,
    q_threshold: float = 0.01,
):
    # Restrict to categories significant at FDR q < q_threshold, THEN take
    # the top_n by (already effect-size-sorted) res_df ordering. This avoids
    # surfacing large-effect-size-but-not-significant categories (typically
    # rare categories with very few nonzero matches).
    plot_df = res_df[res_df["q_value_fdr_bh"] < q_threshold].head(top_n).copy()
    if plot_df.empty:
        print(f"  (no categories with q < {q_threshold} for '{title}', skipping plot)")
        return
    # plot_df is already ordered by effect_size_rank_biserial (descending),
    # inherited from res_df's sort in compare_groups; keep that order as-is
    # for the bar chart's left-to-right layout.

    stars_col = "stars_fdr_q" if use_fdr else "stars_raw_p"
    labels = [
        f"{cat} ({stars})" if stars else cat
        for cat, stars in zip(plot_df["category"], plot_df[stars_col])
    ]

    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 0.6), 6))
    ax.bar(x - width / 2, plot_df[f"mean_{group_a_name}"], width, label=group_a_name)
    ax.bar(x + width / 2, plot_df[f"mean_{group_b_name}"], width, label=group_b_name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=14)
    ax.set_ylabel("normalized empath value", fontsize=14)
    #ax.set_title(title)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outfile}")


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run_comparison(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    mask: pd.Series,
    group_a_name: str,
    group_b_name: str,
    title: str,
    outdir: Path,
    tag: str,
    top_n: int = 15,
):
    res_df = compare_groups(scores, mask, group_a_name, group_b_name)
    csv_out = outdir / f"empath_results_{tag}.csv"
    res_df.to_csv(csv_out, index=False)
    print(f"Saved full results table: {csv_out}")
    sig_df = res_df[res_df["q_value_fdr_bh"] < 0.01]
    print(f"  ({len(sig_df)}/{len(res_df)} categories significant at q<0.01)")
    print(
        sig_df[
            [
                "category",
                f"mean_{group_a_name}",
                f"mean_{group_b_name}",
                "p_value",
                "stars_raw_p",
                "q_value_fdr_bh",
                "stars_fdr_q",
            ]
        ].head(top_n).to_string(index=False)
    )
    plot_top_categories(
        res_df,
        group_a_name,
        group_b_name,
        title + " (raw p-value stars)",
        outdir / f"empath_plot_{tag}_raw_p.png",
        top_n=top_n,
        use_fdr=False,
    )
    plot_top_categories(
        res_df,
        group_a_name,
        group_b_name,
        title + " (FDR-corrected q-value stars)",
        outdir / f"empath_plot_{tag}_fdr_q.png",
        top_n=top_n,
        use_fdr=True,
    )
    return res_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to mobygaze_dataframe.csv")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--top_n", type=int, default=15, help="Top categories to plot")
    parser.add_argument(
        "--annotator", choices=["annotator1", "annotator2"], default=None,
        help="Restrict the whole analysis to a single annotator's rows. "
        "If omitted, both annotators are pooled together.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(args.csv)
    if args.annotator is not None:
        before = len(df)
        df = df[df["annotator"] == args.annotator].reset_index(drop=True)
        print(f"Restricted to annotator='{args.annotator}': {len(df)}/{before} segments.")
    print(f"{len(df)} segments with non-empty subtitle text.")

    print("Scoring subtitles with Empath "
          f"({len(CURATED_CATEGORIES)} curated categories)...")
    scores = score_empath(df["subtitle"], CURATED_CATEGORIES)

    print("Scoring subtitles with Empath "
          f"({len(SPEECH_CATEGORIES)} Speech-thesaurus categories)...")
    speech_cat_scores = score_empath(df["subtitle"], SPEECH_CATEGORIES)

    # ---- Comparison 1: Speech concept present vs absent -----------------
    # Uses only the categories tied to the thesaurus's Speech sub-definitions
    # (comments on appearance, judgement on body, comparison to object/
    # animal, self-loathing, denial of consent, mansterrupting/mansplaining,
    # stereotyping discourse, sexualization, ordering/patronizing/
    # infantilizing) rather than the broader curated list.
    print("\n=== Comparison 1: Speech concept vs No Speech concept ===")
    speech_mask = df["has_speech"]
    run_comparison(
        df,
        speech_cat_scores,
        speech_mask,
        group_a_name="Speech",
        group_b_name="No_Speech",
        title="Empath categories: Speech vs No-Speech concept",
        outdir=outdir,
        tag="speech",
        top_n=args.top_n,
    )

    # ---- Comparison 2: Objectification level -----------------------------
    # Easy Negative vs (Hard Negative + Sure); Not Sure discarded.
    print("\n=== Comparison 2: Easy Negative vs (Hard Negative + Sure) ===")
    obj_df = df[df["label"] != "Not Sure"].copy()
    obj_scores = scores.loc[obj_df.index]
    objectifying_mask = obj_df["label"].isin(["Hard Negative", "Sure"])
    # note: mask True = Objectifying group ("Hard Negative"+"Sure"),
    # mask False = "Easy Negative"
    run_comparison(
        obj_df,
        obj_scores,
        objectifying_mask,
        group_a_name="Objectifying",  # Hard Negative + Sure
        group_b_name="Easy_Negative",
        title="Empath categories: Objectifying (Hard Neg.+Sure) vs Easy Negative",
        outdir=outdir,
        tag="objectification",
        top_n=args.top_n,
    )

    # ---- Comparison 3: within Speech-tagged segments, audio-only vs -----
    # visual-associated concept co-occurrence. "Audio-only" = Speech alone
    # or combined only with Voice/Soundtrack. "Visual-associated" = Speech
    # combined with at least one non-audio concept (Body, Clothing, Look,
    # Appearance, Posture, Activities, Exp of emotion, Narratology, Type of
    # shot). Easy Negative rows are absent from Speech-tagged segments in
    # this dataset (objectification concepts, including Speech, are only
    # annotated when an objectifying cue was actually identified), so this
    # is a direct two-group comparison rather than an objectification split.
    print("\n=== Comparison 3: Speech+Audio-only vs Speech+Visual-associated ===")
    speech_df = df[df["has_speech"]].copy()
    speech_scores = scores.loc[speech_df.index]
    audio_only_mask = speech_df["speech_type"] == "audio"
    print(
        f"  {audio_only_mask.sum()} audio-only segments, "
        f"{(~audio_only_mask).sum()} visual-associated segments"
    )
    run_comparison(
        speech_df,
        speech_scores,
        audio_only_mask,
        group_a_name="Speech_Audio_only",
        group_b_name="Speech_Visual_assoc",
        title="Empath categories: Speech+Audio-only vs Speech+Visual-associated",
        outdir=outdir,
        tag="speech_audio_vs_visual",
        top_n=args.top_n,
    )

    # ---- Comparisons 4.x: {c_v, Speech} concept pair vs rest of dataset --
    # For each visual concept c_v, independently test whether Empath
    # categories related to c_v (per the thesaurus) are significantly
    # different (two-sided) between segments where the concept PAIR
    # {c_v, Speech} is annotated together, versus the rest of the whole
    # dataset (all other segments, regardless of their own concepts).
    print("\n=== Comparisons 4.x: {visual concept, Speech} pairs vs rest ===")
    visual_categories_union = sorted(
        set(cat for cats in VISUAL_CONCEPT_CATEGORIES.values() for cat in cats)
    )
    print(
        "Scoring subtitles with Empath "
        f"({len(visual_categories_union)} visual-thesaurus categories)..."
    )
    visual_scores = score_empath(df["subtitle"], visual_categories_union)

    for c_v, cat_list in VISUAL_CONCEPT_CATEGORIES.items():
        slug = c_v.lower().replace(" ", "_")
        pair_mask = df["concepts_list"].apply(
            lambda lst, cv=c_v: (cv in lst) and ("Speech" in lst)
        )
        print(f"\n--- {c_v} + Speech vs rest ({pair_mask.sum()} pair segments) ---")
        if pair_mask.sum() < 2:
            print(f"  Skipping {c_v}: fewer than 2 segments with this pair.")
            continue
        run_comparison(
            df,
            visual_scores[cat_list],
            pair_mask,
            group_a_name=f"{slug}_Speech_pair",
            group_b_name="Rest",
            title=f"Empath categories related to '{c_v}': "
            f"{{{c_v}, Speech}} pair vs rest",
            outdir=outdir,
            tag=f"pair_{slug}_speech",
            top_n=args.top_n,
        )

    print("\nDone. See the outdir for CSV result tables and PNG plots.")


if __name__ == "__main__":
    main()