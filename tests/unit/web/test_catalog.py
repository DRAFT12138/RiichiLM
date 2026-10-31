import json
from pathlib import Path
import pytest
from mahjong.contracts import MahjongError
from mahjong.web.catalog import Catalog


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    run = root / "run"
    write(
        run / "run.json", dict(run_id="R", experiment="test", created_at="2026-10-26")
    )
    write(
        run / "attempts/A/manifest.json", dict(attempt_id="A", source_checkpoint=None)
    )
    write(
        run / "checkpoints/C/manifest.json",
        dict(
            run_id="R",
            checkpoint_id="C",
            attempt_id="A",
            global_step=30,
            state_sha256="a" * 64,
        ),
    )
    (run / "checkpoints/C/state.pt").write_bytes(b"not read while browsing")
    return run


def test_list_never_loads_weights_and_refresh_does_not_reassign_identity(tmp_path):
    run = fixture(tmp_path)
    catalog = Catalog(tmp_path)
    cp = catalog.data["runs"][0]["checkpoints"][0]
    assert (
        cp["status"] == "待加载验证"
        and catalog.data["runs"][0]["attempts"][0]["outcome"] == "unknown"
    )
    assert catalog.resolve(cp["id"])["step"] == 30
    path = run / "checkpoints/C/manifest.json"
    value = json.loads(path.read_text())
    value["global_step"] = 40
    write(path, value)
    with pytest.raises(MahjongError):
        catalog.resolve(cp["id"])
    catalog.refresh()
    assert catalog.data["runs"][0]["checkpoints"][0]["id"] != cp["id"]
    with pytest.raises(MahjongError):
        catalog.resolve(cp["id"])
    new_id = catalog.data["runs"][0]["checkpoints"][0]["id"]
    (run / "checkpoints/C/state.pt").unlink()
    with pytest.raises(MahjongError):
        catalog.resolve(new_id)
    catalog.refresh()
    assert not catalog.data["runs"][0]["checkpoints"][0]["selectable"]


def test_missing_directory_corruption_nested_rl_and_outside_symlinks(tmp_path):
    assert Catalog(tmp_path / "absent").data["errors"]
    run = fixture(tmp_path / "rl")
    (run / "resolved_config.yaml").write_text("rl: {}\n")
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    fixture(outside)
    (tmp_path / "escape").symlink_to(outside / "run", target_is_directory=True)
    write(tmp_path / "bad/run.json", {"not": "run"})
    catalog = Catalog(tmp_path)
    assert len(catalog.data["runs"]) == 1 and catalog.data["errors"]
    assert not catalog.data["runs"][0]["checkpoints"][0]["selectable"]
    with pytest.raises(MahjongError):
        catalog.resolve("../../escape")


def test_corrupt_yaml_and_nested_summary_only_disable_bad_item(tmp_path):
    run = fixture(tmp_path)
    (run / "resolved_config.yaml").write_text("a: [bad")
    catalog = Catalog(tmp_path)
    assert catalog.data["errors"]
    (run / "resolved_config.yaml").unlink()
    write(run / "attempts/A/summary.json", {"completion": []})
    write(
        run / "checkpoints/bad/manifest.json",
        dict(
            run_id="R",
            checkpoint_id="bad",
            attempt_id="A",
            global_step="bad",
            state_sha256="b" * 64,
        ),
    )
    catalog.refresh()
    assert len(catalog.data["runs"][0]["checkpoints"]) == 1
    assert catalog.data["runs"][0]["errors"]


def test_weight_symlink_outside_root_is_not_selectable(tmp_path):
    run = fixture(tmp_path)
    state = run / "checkpoints/C/state.pt"
    state.unlink()
    outside = tmp_path.parent / (tmp_path.name + "-state.pt")
    outside.write_bytes(b"outside")
    state.symlink_to(outside)
    c = Catalog(tmp_path)
    assert not c.data["runs"][0]["checkpoints"][0]["selectable"]
