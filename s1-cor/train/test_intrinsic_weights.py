"""Tests for intrinsic_weights parsing."""

import pytest

from intrinsic_weights import (
    default_dimension_weights,
    normalize_dimension_weights,
    parse_dimension_weights,
)


def test_default_dimension_weights():
    w = default_dimension_weights()
    assert set(w.keys()) == {"consistency", "completeness", "accuracy", "clarity", "format"}
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_parse_dimension_weights_json():
    w = parse_dimension_weights('{"accuracy": 0.5, "consistency": 0.5}')
    assert w["accuracy"] == 0.5
    assert w["consistency"] == 0.5
    assert w["format"] == 0.0


def test_parse_dimension_weights_kv():
    w = parse_dimension_weights("accuracy=1.0")
    assert w["accuracy"] == 1.0


def test_parse_dimension_weights_empty_returns_default():
    assert parse_dimension_weights(None) == default_dimension_weights()
    assert parse_dimension_weights("") == default_dimension_weights()


def test_parse_dimension_weights_invalid_sum():
    with pytest.raises(ValueError):
        parse_dimension_weights("accuracy=0,consistency=0")


def test_normalize_dimension_weights():
    normed = normalize_dimension_weights({"accuracy": 1.0, "consistency": 1.0})
    assert abs(normed["accuracy"] - 0.5) < 1e-6
    assert abs(normed["consistency"] - 0.5) < 1e-6
