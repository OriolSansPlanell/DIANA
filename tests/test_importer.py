"""Importing externally segmented volumes (``phantoms.importer``)."""

import json
import sys

import numpy as np
import pytest

import neutron_xray_sim as nxs
from neutron_xray_sim.phantoms import importer as imp
from neutron_xray_sim.phantoms.importer import (
    SegmentationMetadata,
    load_segmentation_metadata,
    load_segmented_array,
    metadata_from_dict,
    phantom_from_array,
    phantom_from_segmented_volume,
    remap_labels_to_material_indices,
    resolve_segmentation_metadata,
)

META = {"name": "sample", "voxel_cm": 0.02,
        "class_map": {0: "air", 1: "aluminum", 2: "hdpe", 3: "copper"}}


def _labels(shape=(4, 5, 6), labels=(0, 1, 2, 3), seed=0):
    """Random label volume that uses every label at least once."""
    seg = np.random.default_rng(seed).choice(labels, size=shape)
    seg.flat[:len(labels)] = labels
    return seg


def _write_json(path, data):
    # json.dumps turns int keys into strings, exactly like a hand-written file.
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ── Metadata ─────────────────────────────────────────────────────────────────

def test_metadata_from_dict_normalises_fields():
    meta = metadata_from_dict({"voxel_cm": "0.05", "axis_order": "ZXY",
                               "class_map": {"0": "Air", 7: "ALUMINUM"}})
    assert meta == SegmentationMetadata(name="imported_phantom", voxel_cm=0.05,
                                        class_map={0: "air", 7: "aluminum"},
                                        axis_order="zxy")


@pytest.mark.parametrize("bad, exc, match", [
    ({"class_map": {0: "air"}}, ValueError, "voxel_cm"),
    ({"voxel_cm": 0, "class_map": {0: "air"}}, ValueError, "positive"),
    ({"voxel_cm": -0.1, "class_map": {0: "air"}}, ValueError, "positive"),
    ({"voxel_cm": 0.1, "axis_order": "xyz", "class_map": {0: "air"}},
     NotImplementedError, "zxy"),
    ({"voxel_cm": 0.1}, ValueError, "class_map"),
    ({"voxel_cm": 0.1, "class_map": ["air"]}, ValueError, "must be a mapping"),
    ({"voxel_cm": 0.1, "class_map": {}}, ValueError, "must not be empty"),
    ({"voxel_cm": 0.1, "class_map": {"one": "air"}}, ValueError, "integer"),
    ({"voxel_cm": 0.1, "class_map": {-1: "air"}}, ValueError, "non-negative"),
    ({"voxel_cm": 0.1, "class_map": {0: "air", 1: "unobtainium"}},
     ValueError, r"Unknown material.*unobtainium"),
])
def test_metadata_validation_errors(bad, exc, match):
    with pytest.raises(exc, match=match):
        metadata_from_dict(bad)


def test_metadata_must_be_a_mapping():
    with pytest.raises(TypeError, match="mapping"):
        metadata_from_dict([("voxel_cm", 0.1)])


def test_load_metadata_from_json(tmp_path):
    path = _write_json(tmp_path / "scan_042.json",
                       {"voxel_cm": 0.01, "class_map": {0: "air", 3: "copper"}})
    meta = load_segmentation_metadata(path)
    # JSON keys are strings; without a "name" the file stem is used.
    assert meta.class_map == {0: "air", 3: "copper"}
    assert meta.name == "scan_042" and meta.voxel_cm == 0.01

    _write_json(path, {**META, "name": "explicit"})
    assert load_segmentation_metadata(path).name == "explicit"


def test_load_metadata_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_segmentation_metadata(tmp_path / "missing.json")
    txt = tmp_path / "meta.txt"
    txt.write_text("{}")
    with pytest.raises(ValueError, match=r"\.json"):
        load_segmentation_metadata(txt)
    with pytest.raises(ValueError, match="JSON object"):
        load_segmentation_metadata(_write_json(tmp_path / "list.json", [1, 2]))


def test_resolve_metadata_accepts_object_dict_and_path(tmp_path):
    meta = metadata_from_dict(META)
    assert resolve_segmentation_metadata(meta) is meta
    assert resolve_segmentation_metadata(META) == meta
    path = _write_json(tmp_path / "sample.json", META)
    assert resolve_segmentation_metadata(path) == meta
    assert resolve_segmentation_metadata(str(path)) == meta


def test_resolve_metadata_revalidates_material_names():
    meta = SegmentationMetadata(name="x", voxel_cm=0.1, class_map={0: "kryptonite"})
    with pytest.raises(ValueError, match="kryptonite"):
        resolve_segmentation_metadata(meta)


# ── Label arrays ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("max_label, dtype", [
    (255, np.uint8), (256, np.uint16), (65535, np.uint16), (65536, np.uint32),
])
def test_label_dtype_is_smallest_unsigned_type(max_label, dtype):
    seg = np.array([[[0, max_label]]], dtype=np.int64)
    out = load_segmented_array(seg)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, seg)


def test_2d_array_becomes_single_slice():
    seg = np.arange(6).reshape(2, 3)
    out = load_segmented_array(seg)
    assert out.shape == (1, 2, 3)
    np.testing.assert_array_equal(out[0], seg)


def test_integer_valued_floats_are_accepted():
    out = load_segmented_array(np.array([[0.0, 2.0], [1.0, 2.0]]))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0], [[0, 2], [1, 2]])


@pytest.mark.parametrize("arr, match", [
    (np.array([[0.0, 1.5]]), "integer class labels"),
    (np.array([[0.0, np.nan]]), "integer class labels"),
    (np.array([["a", "b"]]), "integer class labels"),
    (np.array([[0, -1]]), "negative"),
    (np.zeros((0, 3), dtype=int), "empty"),
    (np.zeros(5, dtype=int), "2-D or 3-D"),
    (np.zeros((2, 2, 2, 2), dtype=int), "2-D or 3-D"),
])
def test_invalid_label_arrays(arr, match):
    with pytest.raises(ValueError, match=match):
        load_segmented_array(arr)


def test_load_npy(tmp_path):
    seg = _labels()
    np.save(tmp_path / "seg.npy", seg)
    np.testing.assert_array_equal(load_segmented_array(tmp_path / "seg.npy"), seg)
    np.testing.assert_array_equal(load_segmented_array(str(tmp_path / "seg.npy")), seg)


def test_load_npz_single_array_with_any_name(tmp_path):
    seg = _labels()
    np.savez(tmp_path / "seg.npz", whatever=seg)
    np.testing.assert_array_equal(load_segmented_array(tmp_path / "seg.npz"), seg)


def test_load_npz_prefers_known_keys(tmp_path):
    seg = _labels()
    # 'segmentation' outranks 'volume'; unrelated arrays are ignored.
    np.savez(tmp_path / "seg.npz", raw=np.full((2, 2), 1.5), volume=seg + 1,
             segmentation=seg)
    np.testing.assert_array_equal(load_segmented_array(tmp_path / "seg.npz"), seg)


def test_load_npz_ambiguous_archive(tmp_path):
    np.savez(tmp_path / "seg.npz", a=_labels(), b=_labels())
    with pytest.raises(ValueError, match="multiple arrays"):
        load_segmented_array(tmp_path / "seg.npz")


def test_unsupported_or_missing_volume_files(tmp_path):
    raw = tmp_path / "seg.raw"
    raw.write_bytes(b"\x00" * 8)
    with pytest.raises(ValueError, match="Unsupported"):
        load_segmented_array(raw)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_segmented_array(tmp_path / "nope.npy")


def test_load_tiff_stack_and_single_image(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    seg = _labels(shape=(5, 8, 9)).astype(np.uint8)
    tifffile.imwrite(tmp_path / "stack.tif", seg)
    np.testing.assert_array_equal(load_segmented_array(tmp_path / "stack.tif"), seg)

    tifffile.imwrite(tmp_path / "slice.TIFF", seg[1])  # suffix is case-insensitive
    out = load_segmented_array(tmp_path / "slice.TIFF")
    assert out.shape == (1, 8, 9)
    np.testing.assert_array_equal(out[0], seg[1])


def test_load_tiff_folder_is_sorted_by_name(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    seg = _labels(shape=(5, 6, 7)).astype(np.uint8)
    folder = tmp_path / "slices"
    folder.mkdir()
    for z in (3, 0, 4, 1, 2):  # written out of order
        tifffile.imwrite(folder / f"slice_{z:03d}.tif", seg[z])
    (folder / "notes.txt").write_text("not a slice")
    np.testing.assert_array_equal(load_segmented_array(folder), seg)


def test_tiff_folder_errors(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No .tif"):
        load_segmented_array(empty)

    cases = {
        "shapes": ([(4, 4), (4, 5)], "same shape"),
        "first_3d": ([(5, 4, 4)], "2-D slice"),
        "later_3d": ([(4, 4), (5, 4, 4)], "only 2-D slice"),
    }
    for name, (shapes, match) in cases.items():
        folder = tmp_path / name
        folder.mkdir()
        for i, shape in enumerate(shapes):
            tifffile.imwrite(folder / f"s{i}.tif", np.zeros(shape, np.uint8))
        with pytest.raises(ValueError, match=match):
            load_segmented_array(folder)


def test_tiff_without_tifffile_gives_install_hint(tmp_path, monkeypatch):
    path = tmp_path / "seg.tif"
    path.write_bytes(b"II*\x00")
    monkeypatch.setitem(sys.modules, "tifffile", None)  # makes the import fail
    with pytest.raises(ImportError, match="pip install tifffile"):
        load_segmented_array(path)


# ── Label remapping ──────────────────────────────────────────────────────────

def test_remap_compacts_arbitrary_labels():
    # Label 99 is mapped but unused: allowed, so one map can serve many scans.
    class_map = {0: "air", 10: "aluminum", 42: "iron", 99: "copper"}
    seg = _labels(shape=(6, 7, 8), labels=(0, 10, 42))
    label_vol, materials = remap_labels_to_material_indices(seg, class_map)
    assert label_vol.dtype == np.uint8 and label_vol.max() < len(materials)
    # Every voxel resolves to the material its external label was mapped to.
    names = np.array([m.name for m in materials])
    expected = np.vectorize(lambda lab: nxs.MATERIALS[class_map[lab]].name)(seg)
    np.testing.assert_array_equal(names[label_vol], expected)
    assert materials[0] is nxs.MATERIALS["air"]


def test_air_is_moved_to_index_zero():
    seg = np.array([[[5, 0, 0, 5, 7]]])
    label_vol, materials = remap_labels_to_material_indices(
        seg, {0: "iron", 5: "air", 7: "water"})
    assert [m.name for m in materials] == ["Air", "Iron", "Water"]
    np.testing.assert_array_equal(label_vol, [[[0, 1, 1, 0, 2]]])


def test_remap_without_air_keeps_label_order():
    label_vol, materials = remap_labels_to_material_indices(
        np.array([[[7, 3]]]), {3: "iron", 7: "water"})
    assert [m.name for m in materials] == ["Iron", "Water"]
    np.testing.assert_array_equal(label_vol, [[[1, 0]]])


def test_volume_label_missing_from_class_map():
    with pytest.raises(ValueError, match=r"missing from class_map: \[2\]"):
        remap_labels_to_material_indices(np.array([[[0, 1, 2]]]), {0: "air", 1: "iron"})


# ── Public constructors ──────────────────────────────────────────────────────

def test_phantom_from_array_attenuation_volumes():
    seg = _labels(shape=(3, 5, 7))  # non-cubic: storage order is (Nz, Nx, Ny)
    ph = phantom_from_array(seg, META)
    assert ph.shape == (3, 5, 7) and ph.voxel_cm == 0.02 and ph.name == "sample"
    for label, key in META["class_map"].items():
        mat, sel = nxs.MATERIALS[key], seg == label
        np.testing.assert_array_equal(ph.mu_n_vol[sel], np.float32(mat.mu_n))
        np.testing.assert_array_equal(ph.mu_n_abs_vol[sel], np.float32(mat.mu_n_abs))
        table = mat._mu_x_table.astype(np.float32)[:, None]
        np.testing.assert_array_equal(ph.mu_x_vols[:, sel],
                                      np.broadcast_to(table, (table.size, sel.sum())))
    assert phantom_from_array(seg, META, name="override").name == "override"


def test_phantom_from_2d_array_is_one_slice_thick():
    ph = phantom_from_array(np.array([[0, 1], [1, 1]]),
                            {"voxel_cm": 0.1, "class_map": {0: "air", 1: "iron"}})
    assert ph.shape == (1, 2, 2)
    np.testing.assert_array_equal(ph.label_vol[0], [[0, 1], [1, 1]])


def test_builder_phantom_round_trips_through_files(tmp_path, asymmetric_phantom):
    ph = asymmetric_phantom
    keys = {id(m): k for k, m in nxs.MATERIALS.items()}
    class_map = {i: keys[id(m)] for i, m in enumerate(ph.materials)}
    np.save(tmp_path / "labels.npy", ph.label_vol)
    _write_json(tmp_path / "meta.json",
                {"name": ph.name, "voxel_cm": ph.voxel_cm, "class_map": class_map})

    back = phantom_from_segmented_volume(tmp_path / "labels.npy", tmp_path / "meta.json")
    assert (back.name, back.shape, back.voxel_cm) == (ph.name, ph.shape, ph.voxel_cm)
    assert back.materials == ph.materials  # Material equality is identity
    np.testing.assert_array_equal(back.label_vol, ph.label_vol)
    for attr in ("mu_n_vol", "mu_n_abs_vol", "mu_n_coh_vol", "mu_n_inc_vol", "mu_x_vols"):
        np.testing.assert_array_equal(getattr(back, attr), getattr(ph, attr))


def test_file_and_memory_constructors_agree(tmp_path):
    seg = _labels()
    np.savez(tmp_path / "seg.npz", label_vol=seg, extra=np.ones(3))
    from_files = phantom_from_segmented_volume(
        tmp_path / "seg.npz", _write_json(tmp_path / "m.json", META), name="n")
    in_memory = phantom_from_array(seg, metadata_from_dict(META))
    assert from_files.name == "n"
    np.testing.assert_array_equal(from_files.label_vol, in_memory.label_vol)
    np.testing.assert_array_equal(from_files.mu_x_vols, in_memory.mu_x_vols)
    assert imp.phantom_from_files is phantom_from_segmented_volume
    assert nxs.phantom_from_segmented_volume is phantom_from_segmented_volume


def test_imported_phantom_feeds_the_projector():
    seg = np.zeros((8, 8, 8), int)
    seg[:, 2:6, 2:6] = 1
    seg[:, 3:5, 3:5] = 2
    ph = phantom_from_array(seg, {"voxel_cm": 0.05,
                                  "class_map": {0: "air", 1: "aluminum", 2: "iron"}})
    xs, ns = nxs.make_sinogram_pair(ph, n_angles=6, use_astra=False, verbose=False)
    assert xs["sino_lam"].shape[0] == ns["sino_lam"].shape[0] == 6
    assert xs["voxel_cm"] == 0.05
    assert np.isfinite(ns["sino_lam"]).all() and ns["sino_lam"].max() > 0


# ── Suspected bugs (strict xfail: they start passing once fixed) ──────────────

@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "BUG: a SegmentationMetadata built directly with string class-map keys "
    "(as read from JSON) is not normalised; seg == '1' never matches, so every "
    "voxel silently becomes material 0 (importer.py:180-182, 476)"))
def test_metadata_object_with_json_style_keys():
    seg = np.array([[[0, 1], [1, 2]]])
    meta = SegmentationMetadata(name="s", voxel_cm=0.1,
                                class_map={"0": "air", "1": "aluminum", "2": "iron"})
    try:
        ph = phantom_from_array(seg, meta)
    except (TypeError, ValueError):
        return  # rejecting the object would be an acceptable fix too
    np.testing.assert_array_equal(ph.label_vol, seg)


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason=(
    "BUG: a SegmentationMetadata object bypasses the voxel_cm / axis_order "
    "checks applied to dicts and JSON (importer.py:180-182)"))
@pytest.mark.parametrize("field, value, exc", [
    ("voxel_cm", -0.1, ValueError),
    ("axis_order", "xyz", NotImplementedError),
])
def test_metadata_object_is_validated_like_a_dict(field, value, exc):
    meta = SegmentationMetadata(**{"name": "s", "voxel_cm": 0.1,
                                   "class_map": {0: "air"}, field: value})
    with pytest.raises(exc):
        phantom_from_array(np.zeros((2, 2, 2), int), meta)


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason=(
    "BUG: NaN/inf voxel_cm (json.load accepts the NaN/Infinity literals) passes "
    "the 'voxel_cm <= 0' check (importer.py:119-121)"))
@pytest.mark.parametrize("voxel", [float("nan"), float("inf")])
def test_non_finite_voxel_size_is_rejected(voxel):
    with pytest.raises(ValueError, match="voxel_cm"):
        metadata_from_dict({"voxel_cm": voxel, "class_map": {0: "air"}})


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason=(
    "BUG: labels above 2**32 - 1 wrap around in the uint32 cast instead of "
    "raising, e.g. 2**32 + 1 becomes label 1 (importer.py:399-404)"))
def test_labels_too_large_for_uint32_are_rejected():
    with pytest.raises(ValueError):
        load_segmented_array(np.array([[[0, 1, 2**32 + 1]]], dtype=np.int64))
