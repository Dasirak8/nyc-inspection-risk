import io
import pandas as pd
import pytest

from inspection_risk.data import COLUMNS, CLOSED, INITIAL, OPEN, acquire, build_cohort, load_csv


def row(date="2025-01-06", action=OPEN, **changes):
    values = dict(zip(COLUMNS, (
        "50000001", date, INITIAL, action, "10A", "Critical", "12", "BROOKLYN", "2026-11-20",
    ), strict=True))
    return values | changes


def test_fetch_caches_and_validates_before_install(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame([row()]).to_csv(source, index=False)
    target = tmp_path / "raw" / "cached.csv"
    digest = acquire(target, source.as_uri())["sha256"]
    source.write_text("wrong,header\na,b\n")
    assert acquire(target, source.as_uri())["sha256"] == digest
    with pytest.raises(ValueError, match="Missing columns"):
        acquire(target, source.as_uri(), refresh=True)
    assert load_csv(target).shape == (1, len(COLUMNS))
    assert not list(target.parent.glob("*.part"))


def test_schema_rejects_missing_fields():
    with pytest.raises(ValueError, match="Missing columns"):
        load_csv(io.StringIO("camis,inspection_date\n123,2025-01-06\n"))


def test_grain_duplicates_sentinel_conflicting_outcome_and_reinspection():
    events = [row(), row(), row(violation_code="10B"), row("1900-01-01"),
              row("2025-01-07", CLOSED), row("2025-01-07", OPEN),
              row("2025-01-08", CLOSED, inspection_type="Cycle Inspection / Re-inspection"),
              row("not a date"), row("2025-01-09", camis="bogus")]
    visits, targets, audit = build_cohort(pd.DataFrame(events, columns=COLUMNS))
    assert list(targets["closed"]) == [0]
    assert list(visits["critical_codes"]) == [2, 1, 1]
    assert audit["invalid_or_outside_rows"] == 3
    assert audit["duplicate_rows"] == 1
    assert audit["ambiguous_initial_days"] == 1


def test_unknown_actions_are_not_silently_negative():
    visits, targets, audit = build_cohort(pd.DataFrame([row(action="new category")]))
    assert visits.empty and targets.empty
    assert audit["unrecognized_action_rows"] == 1


def test_numeric_identifiers_are_preserved_as_strings():
    source = io.StringIO(pd.DataFrame([row(camis="1234567890")]).to_csv(index=False))
    _, targets, _ = build_cohort(load_csv(source))
    assert targets.camis.tolist() == ["1234567890"]
