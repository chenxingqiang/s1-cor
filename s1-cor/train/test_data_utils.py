"""Tests for local CoR dataset loading."""

from data_utils import load_cor_dataset_from_disk


def test_load_cor_dataset_from_disk_deepseek():
    dataset = load_cor_dataset_from_disk("local_data/s1K_cor_deepseek")
    assert len(dataset) > 0
    assert "text_cor" in dataset.column_names
    assert dataset[0]["has_self_ratings"] in (True, False)
