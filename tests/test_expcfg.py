"""configs/exp/<ID>.yaml loader and the tools' CLI config builder."""
import pytest

from tideforge import expcfg
from tideforge.gen.config import GenConfig


def test_example_config():
    e = expcfg.load_exp("EXAMPLE")
    assert e.base == GenConfig() and e.seeds == (0,)
    assert dict(dict(e.gen.switches)["test_post"]) == {"p": 0.25, "shift": 1.0, "point": "final_blocks", "priority": "observation"}
    assert e.run_id() == f"EXAMPLE-v0-test_post-{e.gen.config_hash()}"


def test_bad_configs(tmp_path):
    p = tmp_path / "X1.yaml"
    p.write_text("id: X2\nbase: defaults\n")
    with pytest.raises(ValueError, match="file name"):
        expcfg.load_exp(p)
    p.write_text("id: X1\nbase: defaults\ntrain: {}\n")
    with pytest.raises(KeyError, match="unknown key"):
        expcfg.load_exp(p)
    p.write_text("id: X1\nbase: defaults\ngen: {nope: 1}\n")
    with pytest.raises(KeyError, match="unknown generator setting"):
        expcfg.load_exp(p)
    p.write_text("id: X1\nbase: defaults\nswitches: {nope: true}\n")
    with pytest.raises(KeyError, match="unknown generator switch"):
        expcfg.load_exp(p)


def test_gc_from_cli():
    b, gc, label = expcfg.gc_from_cli("defaults", switch=["test_post"], param=["p=0.5"], set_=["gen_additions.breaks=true"])
    assert dict(dict(gc.switches)["test_post"])["p"] == 0.5 and dict(gc.gen_additions)["breaks"] is True
    assert label.startswith("v0-breaks+test_post-")
    _, gc2, _ = expcfg.gc_from_cli("defaults", switch=["test_post", "test_gaps"], param=["test_gaps.p=0.1"])
    assert dict(dict(gc2.switches)["test_gaps"])["p"] == 0.1
    with pytest.raises(ValueError, match="which switch"):
        expcfg.gc_from_cli("defaults", switch=["test_post", "test_gaps"], param=["p=0.1"])
    assert expcfg.gc_from_cli(exp="EXAMPLE")[2] == "EXAMPLE"
