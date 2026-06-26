import fairscope.core as core

EXPECTED = {
    "delong_auc_ci",
    "delong_paired_test",
    "delong_unpaired_test",
    "delong_by_group",
    "bootstrap_auc_test",
    "expected_calibration_error",
    "maximum_calibration_error",
    "ece_by_group",
    "reliability_diagram",
    "temperature_scale",
    "isotonic_recalibrate",
    "bonferroni",
    "benjamini_hochberg",
    "disparate_impact",
    "equalized_odds_difference",
    "subgroup_metrics",
}


def test_public_api_is_exported():
    assert EXPECTED.issubset(set(core.__all__))
    for name in EXPECTED:
        assert hasattr(core, name), f"missing export: {name}"


def test_all_matches_exactly():
    # No stray names; __all__ is exactly the public surface.
    assert set(core.__all__) == EXPECTED


def test_no_duplicate_exports():
    assert len(core.__all__) == len(set(core.__all__))
