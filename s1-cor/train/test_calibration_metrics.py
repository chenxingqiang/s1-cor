"""Tests for calibration_metrics."""

from calibration_metrics import compute_ece


def test_compute_ece_perfect_calibration():
    conf = [0.2, 0.4, 0.6, 0.8]
    acc = [0.2, 0.4, 0.6, 0.8]
    ece, bins = compute_ece(conf, acc, n_bins=4)
    assert ece == 0.0
    assert len(bins) == 4
