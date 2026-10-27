"""GenConfig: mapping from ft3 cfg dicts, diff, hash and run-id fragment, B0 loading."""
import json

import pytest

import tidemark.real.ft3_core as C
from tideforge.gen import config as gcfg
from tideforge.gen.config import FT3_MAIN_CFG, GenConfig, b0


def test_defaults_are_ft3_core_config():
    assert GenConfig() == GenConfig.from_ft3_cfg(dict(C.CONFIG))
    assert dict(GenConfig().gen_additions) == {**C.GEN_ADDITIONS_ALL, **C.CONFIG["gen_additions"]}


def test_to_ft3_cfg_round_trip():
    gc = GenConfig().with_changes(gen_additions={"breaks": True}, cov_r3=True, group_kind_probs=(0.2, 0.2, 0.2, 0.2, 0.2))
    full = dict(C.CONFIG, **gc.to_ft3_cfg())
    assert GenConfig.from_ft3_cfg(full) == gc


def test_univariate_cfg_has_no_groups():
    gc = GenConfig.from_ft3_cfg(dict(C.CONFIG, group_attn=False))
    assert (gc.group_lanes, gc.group_prob, gc.val_group_prob) == (1, 0.0, 0.0)


def test_unknown_addition_and_switch_raise():
    with pytest.raises(KeyError):
        GenConfig.from_ft3_cfg(dict(C.CONFIG, gen_additions={"nope": True}))
    with pytest.raises(KeyError):
        GenConfig(switches=(("nope", True),))


def test_hash_and_run_id():
    base = GenConfig()
    assert base.config_hash() == GenConfig().config_hash()
    assert len(base.config_hash()) == 8
    assert base.run_id_fragment(base) == f"v0-b0-{base.config_hash()}"
    br = base.with_changes(gen_additions={"breaks": True})
    assert br.diff(base) == {"gen_additions.breaks": True}
    assert br.run_id_fragment(base).startswith("v0-breaks-") and br.config_hash() != base.config_hash()
    two = br.with_changes(cov_dominant_p=0.2)
    assert two.run_id_fragment(base).startswith("v0-breaks+cov_dominant_p0.2-")
    off = base.with_changes(gen_additions={"repeat": False})
    assert off.run_id_fragment(base).startswith("v0-norepeat-")


def test_b0_requires_the_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        b0(tmp_path / "missing.json")
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(dict(C.CONFIG, cov_r3=True)))
    assert b0(p).cov_r3 is True


def test_b0_from_ft3_main():
    if not FT3_MAIN_CFG.exists():
        pytest.skip(f"{FT3_MAIN_CFG} is missing (the owner provides it)")
    gc = b0()
    assert gc.version == gcfg.VERSION
    assert gc.run_id_fragment(gc).startswith("v0-b0-")
