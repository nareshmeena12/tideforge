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
    p.write_text("id: X1\nbase: defaults\ntraining: {}\n")
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


def test_model_section(tmp_path):
    a, c = expcfg.load_exp("G1.13a"), expcfg.load_exp("G1.13c")
    assert a.model == {"known_attn": "mask"} and c.model == {"known_attn": "mask", "known_lookahead": 32}
    b0 = expcfg.load_exp("E0.7")
    assert b0.model == {} and a.run_id() == b0.run_id().replace("E0.7", "G1.13a") + "-kamask"
    assert c.run_id().endswith("-kamask-kl32")
    p = tmp_path / "X1.yaml"
    p.write_text("id: X1\nbase: defaults\nmodel: {known_attn: off}\n")
    assert expcfg.load_exp(p).model == {}                     # spelling out off = B0's model and run id
    for bad, err in (("{known_attn: whole}", ValueError), ("{lanes: 16}", KeyError), ("{known_lookahead: 32}", ValueError),
                     ("{known_attn: mask, known_lookahead: 0}", ValueError)):
        p.write_text(f"id: X1\nbase: defaults\nmodel: {bad}\n")
        with pytest.raises(err):
            expcfg.load_exp(p)
