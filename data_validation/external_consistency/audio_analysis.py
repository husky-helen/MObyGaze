"""
Audio analysis of the MObyGaze dataset, replicating the "Audio analysis"
paragraph of the HateMM paper (Das et al., ICWSM 2023), Figures 7 and 8.

HateMM computes three audio features per video/audio file - Zero-Crossing
Rate (ZCR), Spectral Bandwidth, and Root Mean Square (RMS) Energy - as
per-frame time series, then plots the MEAN time series across all files in
each of their two classes (hate vs non-hate) to compare the two curves
visually (Figures 7 and 8).

Adapted here to MObyGaze:
  - Unit of analysis = per-segment (dataframe row), not per full movie.
  - Segment time range in the source movie = [start_frame, end_frame] /
    framerate (in seconds) - NOT start_clip/end_clip, which are indices
    into a 16x-subsampled frame representation (confirmed empirically:
    end_frame / end_clip ~= 16.00 across the whole dataset) and would give
    the wrong (16x too short) audio window if used directly.
  - Groups: segments whose `concepts` include 'Soundtrack' and/or 'Voice'
    (merged, i.e. Soundtrack OR Voice) vs. the rest of the dataset.
  - Each segment's audio is extracted from its movie file with ffmpeg,
    based on `imdb_key` (folder/file are named `{imdb_key}-*`), the three
    features are computed with librosa as per-frame time series, then each
    time series is resampled onto a common normalized time axis (0-100% of
    the segment's duration) so that curves from segments of different
    durations can be averaged together, exactly as HateMM must implicitly
    have done to average across variable-length videos.
  - Beyond HateMM (which shows the curves without a formal significance
    test), we add:
      (a) a pointwise Mann-Whitney U test at each of the normalized-time
          bins, FDR-corrected across bins, with significant bins shaded on
          the time-series plot;
      (b) a scalar-level test: per-segment MEAN feature value, compared
          between groups with a single Mann-Whitney U test + rank-biserial
          effect size, saved to CSV (consistent with the rest of this
          project's pipeline).

Video files are expected to be organized as:
    {video_root}/{imdb_key}-*/{imdb_key}-*.mp4   (or .avi)
matching by `imdb_key` prefix only (not the exact title slug), e.g.:
    /path/to/DVDs/tt0109830-Forrest_gump/tt0109830-Forrest_gump.mp4

================================================================================
CLI OPTIONS
================================================================================
Required:
  --csv PATH             Path to mobygaze_dataframe.csv
  --video_root PATH      Root folder containing {imdb_key}-* movie folders

Output:
  --outdir DIR           Output directory for CSVs/plots (default: audio_results)

Feature extraction:
  --n_time_bins N        Normalized-time resolution for the mean curves /
                          pointwise significance test (default: 100)
  --sr N                 Audio resample rate in Hz (default: 22050)
  --accurate_seek        Slower, frame-accurate ffmpeg seeking (default: fast
                          keyframe seeking, which can jitter by a fraction
                          of a second)
  --max_segments N       Debug: cap the number of segments processed

Scalar summary statistic (used only for the per-feature scalar Mann-Whitney
test, not the time-series curves/plots):
  --quantile Q           Quantile in (0,1) used as each segment's scalar
                          summary instead of the mean, e.g. 0.9 for the
                          90th-percentile ("peak") value of the segment's
                          per-frame feature curve. Default: mean (matches
                          HateMM).

Subsetting:
  --annotator {annotator1,annotator2}
                          Restrict the whole analysis to one annotator's
                          rows. Default: both annotators pooled.

Feature cache (avoids re-running ffmpeg/librosa across repeated runs):
  --cache_dir DIR        Directory to cache raw per-segment feature curves
                          as .npz files, keyed by movie_id+annotator+
                          clip_index. Reusing the same --cache_dir across
                          runs (even with different --outdir, --annotator,
                          --quantile, --n_time_bins) skips re-extraction for
                          any segment already cached. Default:
                          '<outdir>/feature_cache'.
  --no_cache              Disable the cache entirely (always re-extract,
                          never save).
  --force_recompute       Ignore existing cache entries and re-extract/
                          recompute everything (still overwrites the cache
                          with the fresh results).

================================================================================
EXAMPLES
================================================================================
# 1) Basic run, all defaults (mean summary, both annotators, fresh cache
#    written to audio_results/feature_cache/):
    python audio_analysis.py \\
        --csv mobygaze_dataframe.csv \\
        --video_root /media/data/Recherche/ANR_TRACTIVE/Donnees/Elisa/DVDs \\
        --outdir audio_results

# 2) Quick sanity check on a handful of segments before a full run:
    python audio_analysis.py \\
        --csv mobygaze_dataframe.csv \\
        --video_root /media/.../DVDs \\
        --outdir audio_results_debug \\
        --max_segments 50

# 3) Restrict to annotator1 only, using the 90th-percentile ("peak") value
#    per segment instead of the mean:
    python audio_analysis.py \\
        --csv mobygaze_dataframe.csv \\
        --video_root /media/.../DVDs \\
        --outdir audio_results_ann1_q90 \\
        --annotator annotator1 \\
        --quantile 0.9

# 4) Re-run the SAME statistics with a different quantile/annotator without
#    re-extracting audio, by pointing at a cache already populated by a
#    previous run (e.g. the one from example 1, which cached to
#    audio_results/feature_cache/):
    python audio_analysis.py \\
        --csv mobygaze_dataframe.csv \\
        --video_root /media/.../DVDs \\
        --outdir audio_results_ann2_q75 \\
        --cache_dir audio_results/feature_cache \\
        --annotator annotator2 \\
        --quantile 0.75

# 5) Force a full, frame-accurate re-extraction, ignoring/overwriting any
#    cached entries:
    python audio_analysis.py \\
        --csv mobygaze_dataframe.csv \\
        --video_root /media/.../DVDs \\
        --outdir audio_results_accurate \\
        --accurate_seek \\
        --force_recompute
================================================================================
"""

import argparse
import ast
import glob
import io
import subprocess
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
    import librosa
    import soundfile as sf
except ImportError:
    sys.exit(
        "The 'librosa' and 'soundfile' packages are required. Install with:\n"
        "    pip install librosa soundfile"
    )


FEATURES = ["zcr", "spectral_bandwidth", "rms"]
FEATURE_LABELS = {
    "zcr": "Zero-Crossing Rate",
    "spectral_bandwidth": "Spectral Bandwidth",
    "rms": "RMS Energy",
}


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
    df["has_soundtrack_or_voice"] = df["concepts_list"].apply(
        lambda lst: ("Soundtrack" in lst) or ("Voice" in lst)
    )

    # segment time range in the ORIGINAL movie, in seconds
    df["start_sec"] = df["start_frame"] / df["framerate"]
    df["end_sec"] = df["end_frame"] / df["framerate"]
    df["duration_sec"] = df["end_sec"] - df["start_sec"]
    df = df[df["duration_sec"] > 0.05].copy()  # drop degenerate segments

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Video file resolution (by imdb_key prefix, not exact title slug)
# --------------------------------------------------------------------------
def resolve_video_path(imdb_key: str, video_root: str, cache: dict) -> str | None:
    if imdb_key in cache:
        return cache[imdb_key]
    matches = []
    for ext in ("mp4", "avi"):
        matches.extend(
            glob.glob(str(Path(video_root) / f"{imdb_key}-*" / f"{imdb_key}-*.{ext}"))
        )
    path = matches[0] if matches else None
    cache[imdb_key] = path
    return path


# --------------------------------------------------------------------------
# Audio extraction + feature computation for one segment
# --------------------------------------------------------------------------
def extract_audio_segment(
    video_path: str,
    start_sec: float,
    duration_sec: float,
    sr: int = 22050,
    accurate_seek: bool = False,
) -> np.ndarray | None:
    """Extract a mono audio segment via ffmpeg, return as a float32 array."""
    if accurate_seek:
        # -ss AFTER -i: frame-accurate but slower (decodes from file start)
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", video_path,
            "-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}",
            "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", "-",
        ]
    else:
        # -ss BEFORE -i: fast (keyframe) seek, sub-second jitter possible
        cmd = [
            "ffmpeg", "-v", "error",
            "-ss", f"{start_sec:.3f}",
            "-i", video_path,
            "-t", f"{duration_sec:.3f}",
            "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", "-",
        ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            return None
        y, file_sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y
    except Exception:
        return None


def compute_feature_curves(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> dict:
    """Compute per-frame ZCR, Spectral Bandwidth, RMS Energy time series."""
    if y is None or len(y) < frame_length:
        return None
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    return {"zcr": zcr, "rms": rms, "spectral_bandwidth": bw}


def resample_to_normalized_time(curve: np.ndarray, n_bins: int) -> np.ndarray:
    """Linearly resample a 1D curve onto n_bins points over [0, 1]."""
    if len(curve) == 1:
        return np.full(n_bins, curve[0])
    x_orig = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, n_bins)
    return np.interp(x_new, x_orig, curve)


# --------------------------------------------------------------------------
# Raw feature-curve cache (one .npz per segment)
# --------------------------------------------------------------------------
# Caches the RAW, per-frame feature curves (before normalized-time resampling
# and before the mean/quantile summary), keyed by a stable, content-based
# segment identifier (movie_id + annotator + clip_index - NOT the dataframe's
# row position, which shifts between runs with different --annotator or
# --max_segments filters). This lets you freely change --quantile or
# --n_time_bins across runs and re-run the statistical analysis instantly,
# without re-extracting audio or recomputing librosa features, as long as
# the same --cache_dir is reused.
def segment_cache_key(row: pd.Series) -> str:
    raw = f"{row['movie_id']}_{row['annotator']}_{row['clip_index']}"
    # keep it filesystem-safe (clip_index values look like 'id_12', but be safe)
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(raw))


def load_cached_raw_curves(cache_dir: Path, key: str) -> dict | None:
    fpath = cache_dir / f"{key}.npz"
    if not fpath.exists():
        return None
    try:
        with np.load(fpath) as data:
            return {feat: data[feat] for feat in FEATURES}
    except Exception:
        return None  # corrupt/partial cache file - treat as a miss


def save_raw_curves_to_cache(cache_dir: Path, key: str, curves: dict) -> None:
    fpath = cache_dir / f"{key}.npz"
    tmp_path = cache_dir / f"{key}.tmp.npz"  # np.savez auto-appends .npz,
    # so the temp name must already end in .npz or we'd get a mismatched
    # "*.tmp.npz.npz" file instead of the intended temp path.
    try:
        np.savez_compressed(tmp_path, **curves)
        tmp_path.replace(fpath)  # atomic-ish: avoids leaving partial files
    except Exception as e:
        print(f"  warning: failed to cache {key}: {e}")


# --------------------------------------------------------------------------
# Per-dataset processing
# --------------------------------------------------------------------------
def process_dataset(
    df: pd.DataFrame,
    video_root: str,
    n_time_bins: int = 100,
    sr: int = 22050,
    accurate_seek: bool = False,
    max_segments: int | None = None,
    verbose_every: int = 100,
    quantile: float | None = None,
    cache_dir: Path | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    For each row, get the 3 raw per-frame feature curves (ZCR, Spectral
    Bandwidth, RMS Energy) - from `cache_dir` if already cached there,
    otherwise by extracting audio via ffmpeg and computing them with
    librosa (and caching the result for next time, unless cache_dir is
    None). Each raw curve is then resampled to n_time_bins and summarized
    (mean, or `quantile` if given). Returns a DataFrame with one row per
    successfully processed segment, containing:
      - orig_index, has_soundtrack_or_voice (bool)
      - for each feature: '<feature>_curve' (np.ndarray, length n_time_bins)
        and '<feature>_summary' (float, scalar summary over the RAW,
        non-resampled curve - the per-segment mean if `quantile` is None,
        otherwise the given quantile, e.g. quantile=0.9 -> the 90th
        percentile value of the segment's per-frame feature values, i.e.
        a "peak level" summary rather than an average level).
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    path_cache = {}
    records = []
    rows = df if max_segments is None else df.head(max_segments)
    n = len(rows)
    n_cache_hits = 0
    n_freshly_computed = 0
    n_failed = 0
    for i, (orig_idx, row) in enumerate(rows.iterrows()):
        if verbose_every and i % verbose_every == 0:
            print(
                f"  processing segment {i}/{n}... "
                f"({n_cache_hits} cache hits, {n_freshly_computed} extracted, "
                f"{n_failed} failed so far)"
            )
        key = segment_cache_key(row) if cache_dir is not None else None

        curves = None
        if cache_dir is not None and not force_recompute:
            curves = load_cached_raw_curves(cache_dir, key)
            if curves is not None:
                n_cache_hits += 1

        if curves is None:
            video_path = resolve_video_path(row["imdb_key"], video_root, path_cache)
            if video_path is None:
                n_failed += 1
                continue
            y = extract_audio_segment(
                video_path, row["start_sec"], row["duration_sec"],
                sr=sr, accurate_seek=accurate_seek,
            )
            curves = compute_feature_curves(y, sr)
            if curves is None:
                n_failed += 1
                continue
            n_freshly_computed += 1
            if cache_dir is not None:
                save_raw_curves_to_cache(cache_dir, key, curves)

        rec = {"orig_index": orig_idx, "has_soundtrack_or_voice": row["has_soundtrack_or_voice"]}
        ok = True
        for feat in FEATURES:
            c = curves[feat]
            if len(c) < 2:
                ok = False
                break
            rec[f"{feat}_curve"] = resample_to_normalized_time(c, n_time_bins)
            if quantile is None:
                rec[f"{feat}_summary"] = float(np.mean(c))
            else:
                rec[f"{feat}_summary"] = float(np.quantile(c, quantile))
        if ok:
            records.append(rec)
        else:
            n_failed += 1
    print(
        f"  successfully processed {len(records)}/{n} segments "
        f"({n_cache_hits} from cache, {n_freshly_computed} freshly extracted, "
        f"{n_failed} failed/skipped)."
    )
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
    """Positive = group A (x) stochastically greater than group B (y)."""
    return (2 * u_stat) / (n1 * n2) - 1


def scalar_level_test(proc_df: pd.DataFrame, feature: str) -> dict:
    a = proc_df.loc[proc_df["has_soundtrack_or_voice"], f"{feature}_summary"].values
    b = proc_df.loc[~proc_df["has_soundtrack_or_voice"], f"{feature}_summary"].values
    u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
    eff = rank_biserial_effect_size(u_stat, len(a), len(b))
    return {
        "feature": feature,
        "mean_SoundtrackOrVoice": a.mean(),
        "mean_Rest": b.mean(),
        "diff": a.mean() - b.mean(),
        "u_stat": u_stat,
        "p_value": p_val,
        "effect_size_rank_biserial": eff,
        "n_SoundtrackOrVoice": len(a),
        "n_Rest": len(b),
        "stars": p_to_stars(p_val),
    }


def pointwise_test(proc_df: pd.DataFrame, feature: str, n_time_bins: int) -> pd.DataFrame:
    a_curves = np.stack(
        proc_df.loc[proc_df["has_soundtrack_or_voice"], f"{feature}_curve"].values
    )
    b_curves = np.stack(
        proc_df.loc[~proc_df["has_soundtrack_or_voice"], f"{feature}_curve"].values
    )
    p_vals = np.empty(n_time_bins)
    for t in range(n_time_bins):
        try:
            _, p_vals[t] = mannwhitneyu(
                a_curves[:, t], b_curves[:, t], alternative="two-sided"
            )
        except ValueError:
            p_vals[t] = 1.0
    _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
    return pd.DataFrame(
        {
            "time_bin_pct": np.linspace(0, 100, n_time_bins),
            "mean_SoundtrackOrVoice": a_curves.mean(axis=0),
            "mean_Rest": b_curves.mean(axis=0),
            "p_value": p_vals,
            "q_value_fdr_bh": q_vals,
            "significant_q01": q_vals < 0.01,
        }
    )


# --------------------------------------------------------------------------
# Plotting (HateMM Figure 7/8 style: mean time series per group)
# --------------------------------------------------------------------------
def plot_feature_timeseries(pw_df: pd.DataFrame, feature: str, outfile: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = pw_df["time_bin_pct"]
    ax.plot(x, pw_df["mean_SoundtrackOrVoice"], label="Soundtrack/Voice", color="tab:blue")
    ax.plot(x, pw_df["mean_Rest"], label="Rest", color="tab:orange")

    # shade normalized-time regions significant at FDR q<0.01
    sig = pw_df["significant_q01"].values
    ymin, ymax = ax.get_ylim()
    in_run = False
    for i in range(len(sig)):
        if sig[i] and not in_run:
            run_start = x.iloc[i]
            in_run = True
        if in_run and (i == len(sig) - 1 or not sig[i + 1]):
            run_end = x.iloc[i]
            ax.axvspan(run_start, run_end, color="grey", alpha=0.2)
            in_run = False

    ax.set_xlabel("Normalized segment time (%)", fontsize=22)
    ax.set_ylabel(FEATURE_LABELS[feature], fontsize=22)
    '''ax.set_title(
        f"{FEATURE_LABELS[feature]}: Soundtrack/Voice vs Rest "
        "(shaded = FDR q<0.01)"
    )'''
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=24)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {outfile}")


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Path to mobygaze_dataframe.csv")
    parser.add_argument("--video_root", required=True, help="Root folder containing {imdb_key}-* movie folders")
    parser.add_argument("--outdir", default="audio_results", help="Output directory")
    parser.add_argument("--n_time_bins", type=int, default=100, help="Normalized-time resolution")
    parser.add_argument("--sr", type=int, default=22050, help="Audio resample rate")
    parser.add_argument("--accurate_seek", action="store_true", help="Slower, frame-accurate ffmpeg seeking")
    parser.add_argument("--max_segments", type=int, default=None, help="Debug: cap number of segments processed")
    parser.add_argument(
        "--quantile", type=float, default=None,
        help="Quantile in (0,1) used as the per-segment scalar summary "
        "instead of the mean, e.g. 0.9 for the 90th-percentile "
        "('peak') value of each segment's per-frame feature curve. "
        "If omitted, the per-segment mean is used (default, matches HateMM).",
    )
    parser.add_argument(
        "--annotator", choices=["annotator1", "annotator2"], default=None,
        help="Restrict the whole analysis to a single annotator's rows. "
        "If omitted, all annotators are used.",
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="Directory to cache raw (pre-resampling, pre-summary) per-segment "
        "feature curves as .npz files, keyed by movie_id+annotator+clip_index. "
        "Re-running with the same --cache_dir reuses already-extracted "
        "segments instead of re-running ffmpeg/librosa, so you can freely "
        "change --quantile or --n_time_bins and re-do the statistics "
        "quickly. Default: '<outdir>/feature_cache'. Point multiple runs "
        "(e.g. different --annotator or --outdir) at the same --cache_dir "
        "to share cached segments across them.",
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Disable the feature cache entirely (always re-extract, never save).",
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Ignore any cached entries and re-extract/recompute everything "
        "(cache is still written, overwriting old entries).",
    )
    args = parser.parse_args()

    if args.quantile is not None and not (0.0 < args.quantile < 1.0):
        sys.exit("--quantile must be strictly between 0 and 1.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = None
    if not args.no_cache:
        cache_dir = Path(args.cache_dir) if args.cache_dir else outdir / "feature_cache"

    print("Loading data...")
    df = load_data(args.csv)
    if args.annotator is not None:
        before = len(df)
        df = df[df["annotator"] == args.annotator].reset_index(drop=True)
        print(f"Restricted to annotator='{args.annotator}': {len(df)}/{before} segments.")
    print(f"{len(df)} segments with valid time range.")
    print(f"  {df['has_soundtrack_or_voice'].sum()} Soundtrack/Voice segments, "
          f"{(~df['has_soundtrack_or_voice']).sum()} rest.")
    if args.quantile is not None:
        print(f"Using quantile={args.quantile} as the per-segment scalar summary "
              "(instead of the mean).")
    if cache_dir is not None:
        print(f"Feature cache directory: {cache_dir}"
              f"{' (forcing recompute)' if args.force_recompute else ''}")
    else:
        print("Feature cache disabled (--no_cache).")

    print("Extracting audio and computing features (this can take a while)...")
    proc_df = process_dataset(
        df, args.video_root,
        n_time_bins=args.n_time_bins, sr=args.sr,
        accurate_seek=args.accurate_seek, max_segments=args.max_segments,
        quantile=args.quantile,
        cache_dir=cache_dir, force_recompute=args.force_recompute,
    )
    if proc_df.empty:
        sys.exit("No segments could be processed - check --video_root and file naming.")
    proc_df.to_pickle(outdir / "processed_segments.pkl")

    # ---- scalar-level tests (one row per feature) ------------------------
    scalar_results = pd.DataFrame([scalar_level_test(proc_df, f) for f in FEATURES])
    scalar_csv = outdir / "audio_scalar_results.csv"
    scalar_results.to_csv(scalar_csv, index=False)
    print(f"\nSaved scalar-level results: {scalar_csv}")
    print(scalar_results.to_string(index=False))

    # ---- pointwise time-series tests + plots (Figure 7/8 style) ---------
    for feature in FEATURES:
        pw_df = pointwise_test(proc_df, feature, args.n_time_bins)
        pw_csv = outdir / f"audio_timeseries_{feature}.csv"
        pw_df.to_csv(pw_csv, index=False)
        print(f"Saved time-series results: {pw_csv}")
        plot_feature_timeseries(pw_df, feature, outdir / f"audio_plot_{feature}.png")

    print("\nDone. See the outdir for CSV result tables and PNG plots.")


if __name__ == "__main__":
    main()
