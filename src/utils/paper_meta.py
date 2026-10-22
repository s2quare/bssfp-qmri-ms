"""
utils to access meta data in analysis scripts
"""
import pandas as pd
from pathlib import Path
from utils import paths
import yaml
METADATA_FP = paths.data / "251217-b_get_gender_age_participants" / \
    "251217-b_participant_metadata.csv"

# Figure 6


def prepare_demographics(metadata_path: Path) -> pd.DataFrame:
    """Return clean demographics table with subject_id, age_year, and gender."""
    metadata = pd.read_csv(metadata_path)
    demo = metadata[["subject_id", "age", "gender"]].copy()
    demo["subject_id"] = demo["subject_id"].astype(str).str.strip()
    demo["age_year"] = pd.to_numeric(demo["age"], errors="coerce")
    demo["gender"] = demo["gender"].astype(str).str.strip().str.upper()
    return demo[["subject_id", "age_year", "gender"]].dropna().drop_duplicates(
        "subject_id"
    )


def get_subject_id_mapping() -> dict[str, str]:
    """Return mapping from subject_name to subject_id (SID)."""
    mapping: dict[str, str] = {}
    for d in sorted(paths.tf_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("S"):
            continue
        fp = d / "subject_id.txt"
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines()[1:]:
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def get_hc_baseline_subject_ids(subject_map: dict[str, str]) -> list[str]:
    """Return baseline HC SIDs from a subject_name -> SID mapping."""
    return sorted({
        sid for name, sid in subject_map.items()
        if "HC" in name and "_repeat" not in name and "_reposition" not in name
    })


def get_age_matched_rrms_subject_ids(
    subject_map: dict[str, str],
    hc_subject_ids: list[str] | None = None,
    metadata_fp: Path | None = None,
) -> list[str]:
    """Return RRMS SIDs age-capped by max HC age and config exclusions."""
    metadata_path = METADATA_FP if metadata_fp is None else Path(metadata_fp)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Participant metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if hc_subject_ids is None:
        hc_subject_ids = get_hc_baseline_subject_ids(subject_map)

    hc_ages = metadata[metadata["subject_id"].isin(
        hc_subject_ids)]["age"].dropna()
    if hc_ages.empty:
        return []
    hc_age_max = float(hc_ages.max())

    config = yaml.safe_load(open(paths.default_config, "r", encoding="utf-8"))
    excluded_map = config.get("excluded_subjects", {})
    excluded_sids = set(excluded_map.keys())
    excluded_names = set(excluded_map.values())
    if excluded_names:
        excluded_from_names = metadata[metadata["subject_name"].isin(
            excluded_names)]["subject_id"]
        excluded_sids.update(str(sid)
                             for sid in excluded_from_names.dropna().tolist())

    rrms_candidates = metadata[
        metadata["subject_name"].str.contains("MS", na=False)
        & (metadata["age"] <= hc_age_max)
    ]

    # print already the range, mean, std of the age-matched RRMS candidates before SID filtering, to check the effect of age capping
    if not rrms_candidates.empty:
        print(f"Age-matched RRMS candidates (before exclusion due to artifacts):")
        print(f"  Count: {len(rrms_candidates)}")
        print(
            f"  Age range: {rrms_candidates['age'].min()} - {rrms_candidates['age'].max()}")
        print(f"  Age mean: {rrms_candidates['age'].mean():.1f}")
        print(f"  Age std: {rrms_candidates['age'].std():.1f}")

    valid_sids = set(subject_map.values())
    rrms_subjects = sorted({
        sid
        for sid in rrms_candidates["subject_id"].dropna().tolist()
        if isinstance(sid, str)
        and sid.startswith("S")
        and sid in valid_sids
        and sid not in excluded_sids
    })

    # now print the final list of age-matched RRMS subjects after applying SID filtering, to check the final count and age stats
    if rrms_subjects:
        final_rrms_metadata = metadata[metadata["subject_id"].isin(
            rrms_subjects)]
        print(f"Final age-matched RRMS subjects (after exclusion due to artifacts):")
        print(f"  Count: {len(rrms_subjects)}")
        print(
            f"  Age range: {final_rrms_metadata['age'].min()} - {final_rrms_metadata['age'].max()}")
        print(f"  Age mean: {final_rrms_metadata['age'].mean():.1f}")
        print(f"  Age std: {final_rrms_metadata['age'].std():.1f}")

    return rrms_subjects
