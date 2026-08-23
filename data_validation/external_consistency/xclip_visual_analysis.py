"""
Per-clip visual concept scores from pre-extracted X-CLIP features.

For each MObyGaze segment, this computes a "concept score" per visual
concept (e.g. Body, Clothing) by comparing the segment's X-CLIP video-window
embeddings against thesaurus-derived text prompts for that concept, using
the SAME X-CLIP checkpoint's text encoder so the comparison is in a
consistent joint embedding space. The resulting per-concept score is then
compared (Mann-Whitney U + effect size) between segments actually annotated
with that concept vs. not, to see whether the concept's annotation
correlates with the corresponding visual signal.

Key mapping (verified empirically on the dataset):
    start_clip = floor(start_frame / 16)
    end_clip   = floor((end_frame - 1) / 16)
i.e. `start_clip`/`end_clip` ARE the X-CLIP 16-frame window indices
overlapping a segment - no recomputation needed, just
range(start_clip, end_clip + 1) inclusive.

X-CLIP feature files are expected at:
    {xclip_root}/{imdb_key}/{imdb_key}_{window_idx}.pt
each containing a single torch.Tensor of shape [1, 512] (the pooled,
PROJECTED video embedding for that 16-frame window - i.e. already in the
same joint space as the text encoder's projected output, not a raw
pre-projection hidden state).

================================================================================
CLI OPTIONS
================================================================================
Required:
  --csv PATH              Path to mobygaze_dataframe.csv
  --xclip_root PATH       Root folder containing {imdb_key}/{imdb_key}_{idx}.pt

Output:
  --outdir DIR            Output directory (default: xclip_results)

Model:
  --checkpoint NAME       X-CLIP checkpoint for the TEXT encoder only (must
                           match the checkpoint used to extract the video
                           features). Default: microsoft/xclip-base-patch32-16-frames
  --device {cpu,cuda}     Device for text encoding (default: cpu; text
                           encoding is cheap, a GPU is not required)

Aggregation:
  --quantile Q            Quantile in (0,1) used to aggregate a segment's
                           per-window similarity scores instead of the mean
                           (e.g. 0.9 = "peak" presence across the segment).
                           Default: mean.
  --prompt_agg {mean,max} How to combine a concept's multiple thesaurus
                           sub-definition prompts into one concept score.
                           Default: max (any strongly-matching sub-cue
                           counts). Can also be set as the PROMPT_AGG
                           script constant below.

Subsetting:
  --annotator {annotator1,annotator2}
                           Restrict to one annotator's rows. Default: both.
  --max_segments N         Debug: cap number of segments processed.

================================================================================
EXAMPLES
================================================================================
# Basic run:
    python xclip_visual_analysis.py \\
        --csv mobygaze_dataframe.csv \\
        --xclip_root /workspace/tractive/xclip-base-patch32-16-frames_features \\
        --outdir xclip_results

# Using the 90th-percentile (peak) window score instead of the mean,
# annotator1 only:
    python xclip_visual_analysis.py \\
        --csv mobygaze_dataframe.csv \\
        --xclip_root /workspace/tractive/xclip-base-patch32-16-frames_features \\
        --outdir xclip_results_ann1_q90 \\
        --annotator annotator1 --quantile 0.9
================================================================================
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
    import torch
    from transformers import AutoTokenizer, XCLIPModel
except ImportError:
    sys.exit(
        "The 'torch' and 'transformers' packages are required. Install with:\n"
        "    pip install torch transformers"
    )


# ==============================================================================
# SCRIPT VARIABLES - edit prompts / concepts / aggregation here
# ==============================================================================
# One or more free-text prompts per visual concept, drawn from the MObyGaze
# thesaurus. Add more concepts/prompts here as needed (e.g. "Look",
# "Posture", ...) - the rest of the pipeline is generic over this dict.
'''VISUAL_CONCEPT_PROMPTS = {
    "Body": [
        "suggested, partial, or full nudity",
        "body parts suggesting sex",
    ],
    "Clothing": [
        "revealing, wet, or transparent clothing",
        "clothing impractical for the situation",
        "an older woman wearing infantile clothing",
    ]
}
'''
VISUAL_CONCEPT_PROMPTS = {
    "Body": [
        "suggested, partial, or full nudity",
        "body parts suggesting sex",
        "a photo of a woman with visible cleavage",
        "a low-cut top exposing cleavage",
        "a close-up of a chest with cleavage showing",
        "a person wearing a v-neck shirt",
        "a photo focused on the buttocks",
        "a close-up of a person's butt",
        "a rear view showing the buttocks",
        "a photo focused on the buttocks",
        "a close-up of a person's butt",
        "a rear view showing the buttocks",
        "a photo of a bare midriff or torso",
        "an exposed stomach and torso",
        "a cropped top showing the torso",
        "a close-up of full, plump lips",
        "a photo emphasizing large lips",
        "a close-up of lips wearing red lipstick",
        "glossy lipstick on lips",
        "a person with visible lipstick makeup",
        "a sexy woman",
        "a sexy man",
        "a bare torso",
        "a nude person",
        "bare arms",
        "bare legs"
    ],
    "Clothing": [
        "revealing, wet, or transparent clothing",
        "clothing impractical for the situation",
        "an older woman wearing infantile clothing",
        "a photo of a woman with visible cleavage",
        "a low-cut top exposing cleavage",
        "a close-up of a chest with cleavage showing",
        "a person wearing a v-neck shirt",
        "a top with a v-shaped neckline",
        "a v-neck sweater",
        "a person wearing a sleeveless tank top",
        "a tank top exposing the shoulders",
        "a fitted tank top",
        "a person wearing tight-fitting pants",
        "skinny jeans hugging the legs",
        "leggings showing body contour",
        "a woman wearing a short mini-skirt",
        "a skirt ending above mid-thigh",
        "a short skirt exposing the thighs",
        "a woman wearing high heel shoes",
        "high-heeled stiletto shoes"
    ],
    "Type of shot": [
        "a point-of-view shot looking at a woman's body",
        "a low angle shot looking up a woman's body",
        "a camera panning slowly up and down a woman's body",
        "a shot from a man's point of view looking at a woman",
        "a woman filmed in parallel with an animal",
        "a woman and an animal shown side by side",
        "a woman posed like an animal",
        "a shot visually comparing a woman to an animal",
    ],

    "Look": [
        "a man staring intently at a woman's body",
        "an objectifying gaze directed at a woman",
        "a person's eyes fixed on a woman's chest or body",
        "a voyeuristic shot of a woman undressing",
        "someone secretly watching a woman through a window",
        "a hidden or concealed camera view of a woman",
        "spying on a woman through a gap or opening",
        "a woman being stared at without noticing",
        "a woman unaware she is being watched",
        "one-sided staring at an unaware woman",
    ],

    "Posture": [
        "a woman posing seductively",
        "a person in a sexually suggestive pose",
        "a woman arching her back in a suggestive pose",
        "a provocative, sexualized body pose",
        "a woman looking passive and submissive",
        "a person cowering in a submissive posture",
        "a woman in a docile, passive posture",
        "a person looking helpless and dependent on someone else",
        "a woman skipping playfully",
        "a person skipping like a child",
    ],

    "Appearance": [
        "a woman overdressed for a casual, ordinary setting",
        "a woman in glamorous clothing in an everyday setting",
        "a mismatch between a woman's clothing and the situation",
        "a woman dressed inappropriately for the occasion",
        "a close-up of a woman applying makeup",
        "a woman putting on lipstick in a mirror",
        "a woman wearing heavy, elaborate makeup",
        "a woman applying mascara or eyeliner",
        "a close-up of a woman putting on high heel shoes",
        "a woman doing an elaborate beauty routine",
    ],

    "Exp of emotion": [
        "a woman crying",
        "a woman in tears, sobbing",
        "a woman being emotionally expressive and distraught",
        "a man with a completely emotionless expression",
        "a man showing no emotion, stoic and unmoved",
        "a calm, dispassionate man next to a distressed woman",
        "an emotionally expressive woman contrasted with an unemotional man",
    ],

    "Activities": [
        "a woman cooking in the kitchen",
        "a woman cleaning the house",
        "a woman doing laundry or domestic chores",
        "domestic housework being performed by a woman",
        "a couple kissing passionately",
        "an intimate physical act between two people",
        "a suggestive sexual gesture",
        "a person kneeling submissively before someone",
        "a woman being physically restrained or held down",
        "a person grabbing another's arm forcefully",
        "a woman being pulled by the arm against her will",
        "a person being hit or struck",
        "a violent physical altercation",
        "a person being pushed forcefully",
        "domestic violence between two people",
    ],
}

# Generic, non-objectifying "background" prompts used for two purposes:
#   1. Contrastive scoring: a concept's score = similarity(concept prompts)
#      - similarity(these negative prompts), instead of raw cosine
#      similarity, which cancels out the CLIP-family anisotropy floor
#      (unrelated text pairs sitting at ~0.6 cosine similarity rather than
#      ~0) and gives a genuinely discriminative margin.
#   2. Mean-centering: the mean embedding of ALL prompts (concept +
#      negative) is subtracted from every text embedding before comparison,
#      further removing the shared/generic component. Video window
#      embeddings are separately mean-centered by their own per-movie mean.
# Edit freely, same as VISUAL_CONCEPT_PROMPTS.
NEGATIVE_PROMPTS = [
    "a neutral, ordinary scene",
    "a person talking normally",
    "an everyday indoor or outdoor scene",
    "a professional or business setting",
    "a landscape or nature scene",
]

# How to combine a concept's multiple prompt scores into one concept score:
# "max" (any strongly-matching sub-cue counts) or "mean" (average signal
# across all sub-cues). Overridable via --prompt_agg.
PROMPT_AGG = "max"

# How to combine the negative prompts' scores into one "background" score
# per window (default: mean, since negatives represent a broad background
# class rather than specific sub-cues). Overridable via --neg_prompt_agg.
NEG_PROMPT_AGG = "mean"

XCLIP_CHECKPOINT = "microsoft/xclip-base-patch32-16-frames"
EMBED_DIM = 512
# bump this if the cached window array FORMAT changes (e.g. centering logic
# changes) so old, incompatible caches are never silently reused
WINDOW_CACHE_VERSION = "v2"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")

    def parse_concepts(x):
        if pd.isna(x):
            return []
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else []
        except (ValueError, SyntaxError):
            return []

    df["concepts_list"] = df["concepts"].apply(parse_concepts)
    for concept in VISUAL_CONCEPT_PROMPTS:
        slug = concept.lower().replace(" ", "_")
        df[f"has_{slug}"] = df["concepts_list"].apply(lambda lst, c=concept: c in lst)

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# X-CLIP text encoding
# --------------------------------------------------------------------------
def load_text_encoder(checkpoint: str, device: str):
    print(f"Loading X-CLIP text encoder from '{checkpoint}' (downloads on first use)...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = XCLIPModel.from_pretrained(checkpoint).to(device)
    model.eval()
    return tokenizer, model


def encode_prompts_raw(tokenizer, model, prompts: list, device: str) -> torch.Tensor:
    """Encode a list of text prompts into RAW (NOT normalized, NOT
    mean-centered) [n_prompts, EMBED_DIM] embeddings, using the explicit
    text_model -> pooler_output -> text_projection path (robust to
    `get_text_features` wrapper differences across transformers versions).
    Centering/normalization is applied separately (see
    `mean_center_and_normalize`), since centering must happen in raw space
    using a background computed jointly across ALL prompts."""
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
        )
        pooled = text_outputs.pooler_output
        text_embeds = model.text_projection(pooled)
    return text_embeds.detach().cpu()


def mean_center_and_normalize(embeds: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """Subtract a background/mean embedding (removes the shared, generic
    'anisotropy floor' component common to CLIP-family text embeddings),
    then L2-normalize. `background` must be a single [EMBED_DIM] vector."""
    centered = embeds - background
    return centered / centered.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# Movie-level window consolidation cache
# --------------------------------------------------------------------------
# Loading each 16-frame window as its own tiny .pt file is the main
# bottleneck: torch.load() has substantial fixed per-call overhead (opening
# a zip container, unpickling), and segments from different annotators
# overlapping the same movie cause the same window file to be re-read many
# times. Fix: the FIRST time a movie is needed, read all of its .pt files
# ONCE, stack them into a single array, and cache that consolidated array
# to disk (--cache_dir). Every subsequent segment from that movie - in this
# run or any future run pointed at the same --cache_dir - is then served
# from a single fast array load + in-memory slicing, no more per-window
# file I/O at all.
def consolidated_cache_path(cache_dir: Path, imdb_key: str) -> Path:
    return cache_dir / f"{imdb_key}_windows_{WINDOW_CACHE_VERSION}.npz"


def build_or_load_movie_windows(xclip_root: Path, imdb_key: str, cache_dir: Path | None, force_recompute: bool = False):
    """Returns (embeds [n,EMBED_DIM] float32 ndarray, MEAN-CENTERED by this
    movie's own mean embedding and L2-normalized, indices [n] int64 ndarray
    of window indices). Mean-centering removes the shared/generic component
    common to all windows of a movie (e.g. cinematography/color-grading
    style), analogous to the text-side centering in
    `mean_center_and_normalize`."""
    if cache_dir is not None and not force_recompute:
        cpath = consolidated_cache_path(cache_dir, imdb_key)
        if cpath.exists():
            try:
                with np.load(cpath) as data:
                    return data["embeds"], data["indices"]
            except Exception:
                pass  # corrupt cache entry -> rebuild below

    movie_dir = xclip_root / imdb_key
    files = sorted(
        movie_dir.glob(f"{imdb_key}_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not files:
        return None, None

    vecs, indices = [], []
    for fpath in files:
        try:
            idx = int(fpath.stem.rsplit("_", 1)[-1])
            t = torch.load(fpath, map_location="cpu", weights_only=False)
            vecs.append(t.reshape(-1).float().detach().numpy())
            indices.append(idx)
        except Exception:
            continue
    if not vecs:
        return None, None

    embeds = np.stack(vecs).astype(np.float32)  # [n, EMBED_DIM], raw (unnormalized)
    movie_mean = embeds.mean(axis=0, keepdims=True)
    embeds = embeds - movie_mean  # mean-center by this movie's own background
    embeds = embeds / np.linalg.norm(embeds, axis=-1, keepdims=True)
    indices = np.array(indices, dtype=np.int64)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_dir / f"{imdb_key}_windows.tmp.npz"
        cpath = consolidated_cache_path(cache_dir, imdb_key)
        try:
            np.savez_compressed(tmp_path, embeds=embeds, indices=indices)
            tmp_path.replace(cpath)
        except Exception as e:
            print(f"  warning: failed to cache windows for {imdb_key}: {e}")

    return embeds, indices


def slice_segment_embeddings(movie_embeds: np.ndarray, movie_indices: np.ndarray, start_idx: int, end_idx: int):
    """Select the rows of movie_embeds whose window index falls in
    [start_idx, end_idx] (inclusive). Returns (embeds [n,EMBED_DIM], n_found,
    n_expected)."""
    n_expected = end_idx - start_idx + 1
    mask = (movie_indices >= start_idx) & (movie_indices <= end_idx)
    n_found = int(mask.sum())
    if n_found == 0:
        return None, 0, n_expected
    return movie_embeds[mask], n_found, n_expected


# --------------------------------------------------------------------------
# Parallel cache pre-warming across movies
# --------------------------------------------------------------------------
# Building each movie's consolidated cache (reading its many .pt files) is
# embarrassingly parallel across movies - movie A's files are completely
# independent of movie B's. This is I/O + unpickling work with no GPU-
# accelerable math in it, so it benefits from more CPU cores/processes, not
# from a GPU. Must be a module-level function (not a closure) so it can be
# pickled and sent to worker processes.
def _prewarm_one_movie(args) -> tuple:
    xclip_root_str, imdb_key, cache_dir_str, force_recompute = args
    xclip_root = Path(xclip_root_str)
    cache_dir = Path(cache_dir_str)
    try:
        embeds, indices = build_or_load_movie_windows(xclip_root, imdb_key, cache_dir, force_recompute)
        if embeds is None:
            return imdb_key, "missing", 0
        return imdb_key, "ok", len(indices)
    except Exception as e:
        return imdb_key, f"error: {e}", 0


def prewarm_movie_cache(xclip_root: Path, imdb_keys: list, cache_dir: Path, force_recompute: bool, n_workers: int):
    """Populate the on-disk consolidated cache for all given movies in
    parallel, using n_workers worker processes. After this, the main
    (sequential) process_dataset call will hit a warm disk cache for every
    movie instead of reading individual .pt files itself."""
    import concurrent.futures as cf

    tasks = [(str(xclip_root), key, str(cache_dir), force_recompute) for key in imdb_keys]
    n_done, n_missing, n_error = 0, 0, 0
    with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_prewarm_one_movie, t): t[1] for t in tasks}
        for future in cf.as_completed(futures):
            imdb_key = futures[future]
            try:
                key, status, n_windows = future.result()
            except Exception as e:
                print(f"  [{imdb_key}] worker crashed: {e}")
                n_error += 1
                continue
            n_done += 1
            if status == "ok":
                print(f"  [{key}] cached {n_windows} windows ({n_done}/{len(tasks)} movies done)")
            elif status == "missing":
                print(f"  [{key}] no window files found ({n_done}/{len(tasks)} movies done)")
                n_missing += 1
            else:
                print(f"  [{key}] {status} ({n_done}/{len(tasks)} movies done)")
                n_error += 1
    print(f"Pre-warm complete: {n_done} movies processed, {n_missing} missing, {n_error} errors.")


# --------------------------------------------------------------------------
# Per-segment scoring
# --------------------------------------------------------------------------
def aggregate_over_windows(sims: np.ndarray, quantile: float | None) -> np.ndarray:
    """sims: [n_windows, n_prompts] -> [n_prompts] aggregated over windows."""
    if quantile is None:
        return sims.mean(axis=0)
    return np.quantile(sims, quantile, axis=0)


def aggregate_over_prompts(prompt_scores: np.ndarray, method: str) -> float:
    if method == "max":
        return float(np.max(prompt_scores))
    elif method == "mean":
        return float(np.mean(prompt_scores))
    raise ValueError(f"Unknown prompt_agg method: {method}")


def process_dataset(
    df: pd.DataFrame,
    xclip_root: Path,
    concept_prompt_embeds: dict,  # concept -> torch.Tensor [n_prompts, EMBED_DIM], already centered+normalized
    negative_prompt_embeds: torch.Tensor,  # [n_neg, EMBED_DIM], already centered+normalized
    quantile: float | None,
    prompt_agg: str,
    neg_prompt_agg: str,
    cache_dir: Path | None = None,
    force_recompute: bool = False,
    verbose_every: int = 200,
) -> pd.DataFrame:
    """Per-segment concept score = contrastive margin:
        aggregate_over_prompts(sim(concept prompts))
      - aggregate_over_prompts(sim(negative prompts))
    (each prompt-similarity itself first aggregated over the segment's
    windows via `quantile`/mean). This cancels the CLIP-family anisotropy
    floor (unrelated prompts sitting at a high baseline cosine similarity)
    instead of using raw, uncorrected cosine similarity."""
    concepts = list(concept_prompt_embeds.keys())
    # numpy versions of the prompt embeddings, computed once (avoids mixing
    # torch/numpy per segment in the hot loop)
    concept_prompt_embeds_np = {
        c: emb.detach().numpy().astype(np.float32) for c, emb in concept_prompt_embeds.items()
    }
    negative_prompt_embeds_np = negative_prompt_embeds.detach().numpy().astype(np.float32)

    n = len(df)
    # process grouped by movie so each movie's consolidated window array is
    # built/loaded ONCE and reused for all of that movie's segments, instead
    # of re-reading window files per segment
    rows_sorted = df.sort_values("imdb_key", kind="stable")

    records = []
    n_missing = 0
    n_processed = 0
    current_imdb_key = None
    movie_embeds = movie_indices = None
    for orig_idx, row in rows_sorted.iterrows():
        if verbose_every and n_processed % verbose_every == 0:
            print(f"  processing segment {n_processed}/{n}... ({n_missing} skipped so far)")
        n_processed += 1

        imdb_key = row["imdb_key"]
        if imdb_key != current_imdb_key:
            movie_embeds, movie_indices = build_or_load_movie_windows(
                xclip_root, imdb_key, cache_dir, force_recompute
            )
            current_imdb_key = imdb_key
            if movie_embeds is not None:
                print(f"  [{imdb_key}] {len(movie_indices)} windows available")

        if movie_embeds is None:
            n_missing += 1
            continue

        embeds, n_found, n_expected = slice_segment_embeddings(
            movie_embeds, movie_indices, int(row["start_clip"]), int(row["end_clip"])
        )
        if embeds is None:
            n_missing += 1
            continue

        neg_sims = embeds @ negative_prompt_embeds_np.T  # [n_windows, n_neg]
        neg_agg_per_prompt = aggregate_over_windows(neg_sims, quantile)  # [n_neg]
        neg_score = aggregate_over_prompts(neg_agg_per_prompt, neg_prompt_agg)

        rec = {"orig_index": orig_idx, "n_windows_found": n_found, "n_windows_expected": n_expected,
               "background_score": neg_score}
        for concept in concepts:
            slug = concept.lower().replace(" ", "_")
            rec[f"has_{slug}"] = row[f"has_{slug}"]
            prompt_embeds_np = concept_prompt_embeds_np[concept]  # [n_prompts, EMBED_DIM]
            sims = embeds @ prompt_embeds_np.T  # [n_windows, n_prompts]
            agg_per_prompt = aggregate_over_windows(sims, quantile)  # [n_prompts]
            pos_score = aggregate_over_prompts(agg_per_prompt, prompt_agg)
            rec[f"{slug}_raw_score"] = pos_score
            rec[f"{slug}_score"] = pos_score - neg_score  # contrastive margin
        records.append(rec)
    print(f"  successfully processed {len(records)}/{n} segments ({n_missing} skipped: no window files found).")
    return pd.DataFrame(records)




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
    """Positive = group A (concept annotated) stochastically greater."""
    return (2 * u_stat) / (n1 * n2) - 1


def effect_size_label(r: float) -> str:
    """Rough conventional bands for |rank-biserial r|."""
    r = abs(r)
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    return "large"


def concept_test(proc_df: pd.DataFrame, concept: str) -> dict:
    slug = concept.lower().replace(" ", "_")
    a = proc_df.loc[proc_df[f"has_{slug}"], f"{slug}_score"].values
    b = proc_df.loc[~proc_df[f"has_{slug}"], f"{slug}_score"].values
    a_raw = proc_df.loc[proc_df[f"has_{slug}"], f"{slug}_raw_score"].values
    b_raw = proc_df.loc[~proc_df[f"has_{slug}"], f"{slug}_raw_score"].values
    a_bg = proc_df.loc[proc_df[f"has_{slug}"], "background_score"].values
    b_bg = proc_df.loc[~proc_df[f"has_{slug}"], "background_score"].values
    u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
    eff = rank_biserial_effect_size(u_stat, len(a), len(b))
    # common language effect size: P(random annotated score > random
    # not-annotated score). 0.5 = coin flip / no effect, 1.0 = total
    # separation. Much more interpretable than the p-value alone at large N,
    # where even a negligible effect size can produce a tiny p-value.
    cles = (eff + 1) / 2
    return {
        "concept": concept,
        "mean_score_annotated": a.mean(),
        "mean_score_not_annotated": b.mean(),
        "diff": a.mean() - b.mean(),
        "mean_raw_score_annotated": a_raw.mean(),
        "mean_raw_score_not_annotated": b_raw.mean(),
        "mean_background_score_annotated": a_bg.mean(),
        "mean_background_score_not_annotated": b_bg.mean(),
        "u_stat": u_stat,
        "p_value": p_val,
        "effect_size_rank_biserial": eff,
        "effect_size_label": effect_size_label(eff),
        "common_language_effect_size": cles,
        "n_annotated": len(a),
        "n_not_annotated": len(b),
        "stars": p_to_stars(p_val),
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_concept_boxplot(proc_df: pd.DataFrame, concept: str, result: dict, outfile: Path):
    slug = concept.lower().replace(" ", "_")
    a = proc_df.loc[proc_df[f"has_{slug}"], f"{slug}_score"].values
    b = proc_df.loc[~proc_df[f"has_{slug}"], f"{slug}_score"].values

    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot([a, b], tick_labels=[f"{concept} annotated\n(n={len(a)})", f"Not annotated\n(n={len(b)})"],
                     showmeans=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("tab:blue")
    bp["boxes"][1].set_facecolor("tab:orange")
    ax.set_ylabel(f"X-CLIP '{concept}' contrastive score (concept - background similarity)")
    ax.set_title(
        f"{concept}: annotated vs not\n"
        f"p={result['p_value']:.2e} {result['stars']}  |  "
        f"effect size r={result['effect_size_rank_biserial']:.3f} ({result['effect_size_label']})  |  "
        f"P(annotated>not)={result['common_language_effect_size']:.1%}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outfile}")


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", required=True, help="Path to mobygaze_dataframe.csv")
    parser.add_argument("--xclip_root", required=True, help="Root folder containing {imdb_key}/{imdb_key}_{idx}.pt")
    parser.add_argument("--outdir", default="xclip_results", help="Output directory")
    parser.add_argument("--checkpoint", default=XCLIP_CHECKPOINT, help="X-CLIP checkpoint for the text encoder")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for text encoding")
    parser.add_argument("--quantile", type=float, default=None,
                         help="Quantile in (0,1) to aggregate per-window scores instead of the mean")
    parser.add_argument("--prompt_agg", default=PROMPT_AGG, choices=["mean", "max"],
                         help="How to combine a concept's multiple prompts into one score")
    parser.add_argument("--neg_prompt_agg", default=NEG_PROMPT_AGG, choices=["mean", "max"],
                         help="How to combine the negative/background prompts into one score")
    parser.add_argument("--annotator", choices=["annotator1", "annotator2"], default=None,
                         help="Restrict to one annotator's rows")
    parser.add_argument("--max_segments", type=int, default=None, help="Debug: cap number of segments processed")
    parser.add_argument(
        "--cache_dir", default=None,
        help="Directory to cache consolidated per-movie window arrays (one "
        ".npz per movie, containing ALL of that movie's window embeddings "
        "stacked together). This is the main speed lever: without it, every "
        "segment re-reads its window .pt files individually, which is very "
        "slow (~465k individual file loads across the full dataset) and "
        "redundant across overlapping annotator segments. With it, each "
        "movie's window files are read once (ever, across all runs sharing "
        "this --cache_dir) and reused instantly afterward. "
        "Default: '<outdir>/window_cache'.",
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Disable the consolidated window cache (always re-read individual "
        ".pt files fresh, never save). Not recommended - much slower.",
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Ignore any cached consolidated window arrays and rebuild them "
        "from the individual .pt files (cache is still written, overwriting "
        "old entries).",
    )
    parser.add_argument(
        "--n_workers", type=int, default=1,
        help="Number of parallel worker processes used to pre-build the "
        "consolidated per-movie window cache before scoring. Building each "
        "movie's cache (reading its many .pt files) is embarrassingly "
        "parallel across movies, and is CPU/I/O-bound rather than GPU-"
        "accelerable, so this benefits from more CPU cores (e.g. a SLURM "
        "allocation with --cpus-per-task > 1), not from a GPU. Requires "
        "--cache_dir to be enabled (default). Default: 1 (no parallelism).",
    )
    args = parser.parse_args()

    if args.quantile is not None and not (0.0 < args.quantile < 1.0):
        sys.exit("--quantile must be strictly between 0 and 1.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    xclip_root = Path(args.xclip_root)
    cache_dir = None
    if not args.no_cache:
        cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "window_cache"

    print("Loading data...")
    df = load_data(args.csv)
    if args.annotator is not None:
        before = len(df)
        df = df[df["annotator"] == args.annotator].reset_index(drop=True)
        print(f"Restricted to annotator='{args.annotator}': {len(df)}/{before} segments.")
    if args.max_segments is not None:
        before = len(df)
        df = df.head(args.max_segments).reset_index(drop=True)
        print(f"Restricted to --max_segments={args.max_segments}: {len(df)}/{before} segments.")
    print(f"{len(df)} segments.")
    for concept in VISUAL_CONCEPT_PROMPTS:
        slug = concept.lower().replace(" ", "_")
        print(f"  {concept}: {df[f'has_{slug}'].sum()} annotated / {len(df)}")
    if cache_dir is not None:
        print(f"Window cache directory: {cache_dir}"
              f"{' (forcing recompute)' if args.force_recompute else ''}")
    else:
        print("Window cache disabled (--no_cache) - this will be slow.")

    tokenizer, model = load_text_encoder(args.checkpoint, args.device)

    # Encode ALL prompts (every concept's + the negatives) RAW first, so the
    # mean-centering background is computed jointly across the full prompt
    # pool - this is what actually cancels the shared CLIP-family
    # anisotropy component, rather than any one subset's own mean.
    print(f"Encoding {len(NEGATIVE_PROMPTS)} negative/background prompt(s): {NEGATIVE_PROMPTS}")
    raw_negative_embeds = encode_prompts_raw(tokenizer, model, NEGATIVE_PROMPTS, args.device)
    raw_concept_embeds = {}
    for concept, prompts in VISUAL_CONCEPT_PROMPTS.items():
        print(f"Encoding {len(prompts)} prompt(s) for concept '{concept}': {prompts}")
        raw_concept_embeds[concept] = encode_prompts_raw(tokenizer, model, prompts, args.device)

    all_raw = torch.cat([raw_negative_embeds] + list(raw_concept_embeds.values()), dim=0)
    background = all_raw.mean(dim=0)  # [EMBED_DIM]
    print(f"Computed text background vector from {all_raw.shape[0]} total prompts "
          f"({len(NEGATIVE_PROMPTS)} negative + {sum(len(p) for p in VISUAL_CONCEPT_PROMPTS.values())} concept).")

    negative_prompt_embeds = mean_center_and_normalize(raw_negative_embeds, background)
    concept_prompt_embeds = {
        c: mean_center_and_normalize(emb, background) for c, emb in raw_concept_embeds.items()
    }

    if args.n_workers > 1:
        if cache_dir is None:
            print("Warning: --n_workers > 1 has no effect without a cache "
                  "(drop --no_cache to enable it). Proceeding without parallel pre-warm.")
        else:
            unique_movies = df["imdb_key"].unique().tolist()
            print(f"Pre-warming window cache for {len(unique_movies)} movies "
                  f"using {args.n_workers} parallel worker processes...")
            prewarm_movie_cache(xclip_root, unique_movies, cache_dir, args.force_recompute, args.n_workers)
            # cache is now guaranteed fresh; avoid redundantly forcing a
            # second rebuild from raw .pt files in the sequential pass below
            args.force_recompute = False

    print("Scoring segments against X-CLIP video window features...")
    proc_df = process_dataset(
        df, xclip_root, concept_prompt_embeds, negative_prompt_embeds,
        quantile=args.quantile, prompt_agg=args.prompt_agg, neg_prompt_agg=args.neg_prompt_agg,
        cache_dir=cache_dir, force_recompute=args.force_recompute,
    )
    if proc_df.empty:
        sys.exit("No segments could be processed - check --xclip_root and file naming.")
    proc_df.to_pickle(outdir / "processed_segments.pkl")

    results = pd.DataFrame([concept_test(proc_df, c) for c in VISUAL_CONCEPT_PROMPTS])
    _, q_vals, _, _ = multipletests(results["p_value"], method="fdr_bh")
    results["q_value_fdr_bh"] = q_vals
    results["stars_fdr_q"] = results["q_value_fdr_bh"].apply(p_to_stars)
    results_csv = outdir / "xclip_concept_results.csv"
    results.to_csv(results_csv, index=False)
    print(f"\nSaved results: {results_csv}")
    print(results.to_string(index=False))

    for _, row in results.iterrows():
        plot_concept_boxplot(proc_df, row["concept"], row.to_dict(), outdir / f"xclip_plot_{row['concept'].lower().replace(' ', '_')}.png")

    print("\nDone. See the outdir for CSV results and PNG plots.")


if __name__ == "__main__":
    main()