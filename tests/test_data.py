"""Tests for apple_scaffold.data. Run from the repository root with ``pytest``."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from apple_scaffold import data


# ---------------------------------------------------------------- helpers
def write_png(path, array):
    """Save an array as PNG, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def fake_depth(value=100, size=(48, 64)):
    grey = np.full(size, value, np.uint8)
    grey[:5, :5] = 0                                   # a no-return patch
    return np.stack([grey] * 3, axis=-1)               # stored as 3 equal channels


def fake_rgb(size=(48, 64)):
    return np.zeros(size + (3,), np.uint8)


def fake_gt(size=(32, 32)):
    mask = np.zeros(size, np.uint8)
    mask[:, 14:18] = 255
    return mask


def write_tree(folder, tree, summer_time="12-41-4", winter_time="11-23-49", gt=True):
    """Write a complete labelled tree: summer D/RGB(/GT) and winter D/RGB."""
    write_png(folder / f"{tree}_Summer_D-{summer_time}.png", fake_depth())
    write_png(folder / f"{tree}_Summer_RGB-{summer_time}.png", fake_rgb())
    if gt:
        write_png(folder / f"{tree}_Summer_GT-{summer_time}.png", fake_gt())
    write_png(folder / f"{tree}_Winter_D-{winter_time}.png", fake_depth())
    write_png(folder / f"{tree}_Winter_RGB-{winter_time}.png", fake_rgb())


# ---------------------------------------------------------------- parsing
def test_parse_unpadded_time():
    fields = data.parse_filename("R01N03_Summer_GT-12-41-4.png")
    assert fields["tree_id"] == "R01N03"
    assert fields["row"] == "R01"
    assert fields["labelled"] is True
    assert fields["season"] == "Summer"
    assert fields["modality"] == "GT"
    assert fields["time_s"] == 12 * 3600 + 41 * 60 + 4
    assert fields["source_id"] is None


def test_parse_rematched_winter_name():
    fields = data.parse_filename("R04N15_Winter_RGB-R04N42_RGB_12-19-45.png")
    assert fields["tree_id"] == "R04N15"
    assert fields["season"] == "Winter"
    assert fields["modality"] == "RGB"
    assert fields["source_id"] == "R04N42"
    assert fields["time_s"] == 12 * 3600 + 19 * 60 + 45


def test_rematched_winter_files_pair_into_one_clean_frame(tmp_path):
    write_png(tmp_path / "R04N06_Winter_D-R04N18_RGB_12-18-07.png", fake_depth())
    write_png(tmp_path / "R04N06_Winter_RGB-R04N18_RGB_12-18-07.png", fake_rgb())
    files, unmatched = data.index_files([tmp_path])
    frames = data.build_frames(files)
    assert unmatched == []
    assert len(frames) == 1 and frames.iloc[0].problems == ""


def test_parse_unlabelled_tree():
    fields = data.parse_filename("R01U05_Winter_RGB-11-24-06.png")
    assert fields["tree_id"] == "R01U05"
    assert fields["labelled"] is False
    assert fields["modality"] == "RGB"


def test_parse_rejects_other_names():
    for name in ["desktop.ini", "R01N01_Summer_D-12-40-50.jpg",
                 "R01N01_Summer_X-12-40-50.png", "notes.png"]:
        assert data.parse_filename(name) is None, name


# ---------------------------------------------------------------- pairing
def test_complete_tree_gives_two_clean_frames(tmp_path):
    write_tree(tmp_path, "R01N01")
    files, unmatched = data.index_files([tmp_path])
    frames = data.build_frames(files)
    assert unmatched == []
    assert len(frames) == 2
    assert (frames["problems"] == "").all()
    summer = frames[frames.season == "Summer"].iloc[0]
    assert summer.path_GT is not None


def test_missing_gt_and_time_gap_are_flagged(tmp_path):
    write_tree(tmp_path, "R01N02", gt=False)
    write_png(tmp_path / "R01N02_Summer_GT-12-41-30.png", fake_gt())   # 26 s later
    frames = data.build_frames(data.index_files([tmp_path])[0])
    summer = frames[frames.season == "Summer"].iloc[0]
    assert "26s apart" in summer.problems

    write_tree(tmp_path / "other", "R01N03", gt=False)
    frames = data.build_frames(data.index_files([tmp_path / "other"])[0])
    assert "missing GT" in frames[frames.season == "Summer"].iloc[0].problems


def test_unlabelled_tree_needs_no_gt_and_duplicates_are_flagged(tmp_path):
    write_tree(tmp_path, "R01U01", gt=False)
    write_png(tmp_path / "copy" / "R01U01_Winter_D-11-23-49.png", fake_depth())
    frames = data.build_frames(data.index_files([tmp_path])[0])
    summer = frames[frames.season == "Summer"].iloc[0]
    winter = frames[frames.season == "Winter"].iloc[0]
    assert summer.problems == ""
    assert "2 D files" in winter.problems


# ---------------------------------------------------------------- loaders
def test_load_depth_collapses_identical_channels(tmp_path):
    write_png(tmp_path / "d.png", fake_depth(value=120))
    depth = data.load_depth(tmp_path / "d.png")
    assert depth.shape == (48, 64)
    assert depth.dtype == np.uint8
    assert depth[0, 0] == 0 and depth[-1, -1] == 120


def test_load_depth_rejects_colour_mapped_depth(tmp_path):
    coloured = fake_depth()
    coloured[..., 1] = 7
    write_png(tmp_path / "d.png", coloured)
    with pytest.raises(ValueError):
        data.load_depth(tmp_path / "d.png")


def test_load_gt_resize_keeps_mask_binary_and_area_fraction(tmp_path):
    write_png(tmp_path / "gt.png", fake_gt())
    native = data.load_gt(tmp_path / "gt.png", size=None)
    resized = data.load_gt(tmp_path / "gt.png", size=(640, 480))
    assert native.dtype == bool and resized.dtype == bool
    assert resized.shape == (480, 640)
    # nearest-neighbour resizing should preserve the mask's area fraction
    assert resized.mean() == pytest.approx(native.mean(), abs=0.01)


# ---------------------------------------------------------------- splitting
def make_frames(n_rows=4, trees_per_row=10):
    records = [{"tree_id": f"R{r:02d}N{t:02d}", "row": f"R{r:02d}", "season": s}
               for r in range(1, n_rows + 1) for t in range(1, trees_per_row + 1)
               for s in ("Summer", "Winter")]
    return pd.DataFrame(records)


def test_grouped_split_never_shares_a_tree():
    frames = make_frames()
    labels = data.grouped_split(frames, group_by="tree_id", seed=3)
    assert set(labels) == {"train", "val", "test"}
    trees = {name: set(frames.tree_id[labels == name]) for name in ("train", "val", "test")}
    assert not trees["train"] & trees["test"]
    assert not trees["train"] & trees["val"]
    assert not trees["val"] & trees["test"]


def test_grouped_split_by_row_keeps_rows_together():
    frames = make_frames(n_rows=10, trees_per_row=3)
    labels = data.grouped_split(frames, val_size=0.2, test_size=0.2, group_by="row")
    assert (frames.assign(split=labels).groupby("row")["split"].nunique() == 1).all()


def test_grouped_split_is_reproducible():
    frames = make_frames()
    first = data.grouped_split(frames, seed=7)
    second = data.grouped_split(frames, seed=7)
    assert first.equals(second)
