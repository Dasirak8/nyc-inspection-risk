"""Download a minimal public extract and reconstruct the inspection-day grain."""

import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

COLUMNS = (
    "camis", "inspection_date", "inspection_type", "action", "violation_code", "critical_flag",
    "score", "boro", "record_date",
)
LIMIT = 500_000
URL = "https://data.cityofnewyork.us/resource/43nn-pn8j.csv?" + urlencode({
    "$select": ",".join(COLUMNS), "$order": ":id", "$limit": LIMIT,
})
HISTORY_START = "2023-12-01"
STUDY_END = "2026-06-29"  # exclusive; a Monday
INITIAL = "Cycle Inspection / Initial Inspection"
CLOSED = ("Establishment Closed by DOHMH. Violations were cited in the following area(s) "
          "and those requiring immediate action were addressed.")
OPEN = "Violations were cited in the following area(s)."
ACTION = {
    CLOSED: 1, "Establishment re-closed by DOHMH.": 1, OPEN: 0,
    "No violations were recorded at the time of this inspection.": 0,
    "Establishment re-opened by DOHMH.": 0,
}


def load_csv(path):
    """Validate column presence and fail loudly at the single-response query cap."""
    frame = pd.read_csv(path, dtype="string", usecols=lambda name: name in COLUMNS)
    if missing := set(COLUMNS) - set(frame):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or len(frame) >= LIMIT:
        raise ValueError("Empty or possibly truncated download; inspect the source before running.")
    return frame.loc[:, list(COLUMNS)]


def describe(path: Path, url: str) -> dict:
    frame = load_csv(path)
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    dates = pd.to_datetime(frame.record_date, format="mixed", errors="coerce").dropna()
    return {"source_url": url, "sha256": sha, "rows": len(frame), "bytes": path.stat().st_size,
            "record_date_min": dates.min().isoformat() if len(dates) else None,
            "record_date_max": dates.max().isoformat() if len(dates) else None}


def acquire(path: Path, url: str = URL, *, refresh: bool = False) -> dict:
    """Use one response (no cross-page drift), retry transient HTTP failures, cache atomically."""
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() and not refresh:
        meta = describe(path, url)
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if meta["sha256"] != saved["sha256"]:
                raise ValueError("Cached input differs from its manifest; restore it or use --refresh.")
            return saved
        return meta
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as temp:
        incoming = Path(temp.name)
    try:
        for attempt in range(3):
            try:
                request = Request(url, headers={"User-Agent": "nyc-inspection-risk/0.1"})
                with urlopen(request, timeout=90) as response, incoming.open("wb") as f:
                    shutil.copyfileobj(response, f)
                break
            except (URLError, TimeoutError) as exc:
                transient = not isinstance(exc, HTTPError) or exc.code in {429, 500, 502, 503, 504}
                if attempt == 2 or not transient:
                    raise
                time.sleep(2 ** attempt)
        meta = describe(incoming, url)
        meta["fetched_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        incoming.replace(path)
        manifest_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    finally:
        incoming.unlink(missing_ok=True)
    return meta


def build_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return one history event per CAMIS/day and one labeled routine event per CAMIS/day."""
    data = raw.copy()
    data["date"] = pd.to_datetime(data.inspection_date, format="mixed", errors="coerce").dt.normalize()
    # The current extract has 8-digit keys; accept the catalog's wider integer field too.
    good = (data.camis.str.fullmatch(r"[0-9]{1,10}", na=False)
            & data.date.ge(HISTORY_START) & data.date.lt(STUDY_END))
    audit = {"raw_rows": len(data), "invalid_or_outside_rows": int((~good).sum()),
             "sentinel_rows": int(data.date.eq("1900-01-01").sum()),
             "unparseable_date_rows": int(data.date.isna().sum()),
             "missing_score_rows_in_window": int(data.loc[good, "score"].isna().sum())}
    data = data.loc[good].drop_duplicates(subset=list(COLUMNS)).copy()
    audit["duplicate_rows"] = int(good.sum() - len(data))
    data["closed"] = data.action.map(ACTION)
    audit["unrecognized_action_rows"] = int(data.closed.isna().sum())
    key = ["camis", "date"]
    initial = data.loc[data.inspection_type.eq(INITIAL)].copy()
    borough = initial.boro.str.strip().str.upper()
    initial["borough"] = borough.where(borough.isin(
        ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]), "UNKNOWN")
    by_initial = initial.groupby(key).agg(
        closed=("closed", "max"), distinct_outcomes=("closed", "nunique"),
        recognized=("closed", "count"), rows=("closed", "size"),
        borough=("borough", "min"), distinct_boroughs=("borough", "nunique"),
    )
    eligible = by_initial.distinct_outcomes.eq(1) & by_initial.recognized.eq(by_initial.rows)
    audit["ambiguous_initial_days"] = int((~eligible).sum())
    targets = by_initial.loc[eligible, ["closed", "borough"]].reset_index()
    targets.loc[by_initial.loc[eligible, "distinct_boroughs"].to_numpy() > 1, "borough"] = "UNKNOWN"
    targets["closed"] = targets.closed.astype("int8")
    history = data.loc[data.closed.notna()].copy()
    scores = pd.to_numeric(history.score, errors="coerce")
    history["numeric_score"] = scores.where(scores.ge(0) & scores.le(500))
    history["critical_code"] = history.violation_code.where(history.critical_flag.eq("Critical"))
    visits = history.groupby(key, as_index=False).agg(
        closed=("closed", "max"), score=("numeric_score", "max"),
        critical_codes=("critical_code", "nunique"),
    )
    audit |= {"recognized_history_rows": len(history), "history_days": len(visits),
              "labeled_routine_days": len(targets), "routine_closures_all_dates": int(targets.closed.sum())}
    return visits, targets, audit
