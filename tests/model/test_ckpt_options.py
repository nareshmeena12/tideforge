"""M1 adds 0 parameters, so a strict state-dict load cannot catch a lost option: the options must come back from the cfg."""
import torch

import tideforge.model as TM
from tideforge.model.m1 import M1Decoder


def test_loaded_checkpoint_keeps_known_attn(tmp_path):
    from tideforge.train.recipe import load
    cfg = load("G1.13a").ft3_cfg()
    assert cfg["known_attn"] == "mask"
    p = tmp_path / "c.pt"
    torch.save(dict(model=TM.build(cfg).state_dict(), cfg=cfg), p)
    loaded, got, _ = TM.load(p, torch.device("cpu"))
    assert isinstance(loaded, M1Decoder) and loaded.known_attn == "mask" and got["known_attn"] == "mask"
