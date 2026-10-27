"""MaskedLossDecoder: the reference decoder plus a loss mask. Same weights, exactly the reference loss when the mask is all True."""
import inspect
import textwrap

import pytest
import torch

from tidemark.real import ft3_core as C
from tidemark.real import model as M
from test_real_eval import tiny_cfg


def test_forward_source_differs_from_the_reference_only_in_the_marked_lines():
    ref = textwrap.dedent(inspect.getsource(C.CausalPatchDecoder.forward)).splitlines()
    new = textwrap.dedent(inspect.getsource(M.MaskedLossDecoder.forward)).splitlines()

    def strip(lines):
        out, in_doc = [], False
        for ln in lines:
            if '"""' in ln:                                   # drop the docstrings (they differ by design)
                n = ln.count('"""')
                if n == 1:
                    in_doc = not in_doc
                continue
            if in_doc:
                continue
            out.append(ln.split("   # CHANGED")[0].rstrip())
        return out

    r, n = strip(ref), strip(new)
    diff = [(a, b) for a, b in zip(r, n) if a != b]
    assert len(r) == len(n)
    assert len(diff) == 2, diff                               # the signature and the target mask, nothing else
    assert "loss_mask=None" in diff[0][1] and "loss_mask" in diff[1][1] and "ym = " in diff[1][1]


@pytest.fixture(scope="module")
def setup():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    ref = C.build_model(dict(cfg, dropout=0.0))
    new = M.build_masked_model(dict(cfg, dropout=0.0))
    res = new.load_state_dict(ref.state_dict(), strict=True)                      # same parameters
    assert not res.missing_keys and not res.unexpected_keys
    C.set_group_options(None)
    torch.manual_seed(3)
    b = C.PackedBatcher(dict(cfg, pack_rows=16), torch.device("cpu"), seed=3).next_batch()
    return ref.eval(), new.eval(), b


def test_loss_equals_the_reference_when_the_loss_mask_is_all_true(setup):
    ref, new, b = setup
    with torch.no_grad():
        o_ref = C.packed_forward(ref, b)
        o_new = M.packed_forward_masked(new, dict(b, loss_ok=torch.ones_like(b["mask"])))
        o_none = M.packed_forward_masked(new, b)
    assert torch.equal(o_ref["loss"], o_new["loss"]) and torch.equal(o_ref["loss"], o_none["loss"])
    assert torch.equal(o_ref["_q"], o_new["_q"])
    assert torch.equal(torch.nan_to_num(o_ref["per_series_loss"]), torch.nan_to_num(o_new["per_series_loss"]))


def test_masked_targets_leave_the_loss_and_the_input_is_unchanged(setup):
    ref, new, b = setup
    lm = torch.ones_like(b["mask"])
    lm[:, 100:140] = False                                                         # some targets not scored
    with torch.no_grad():
        full = M.packed_forward_masked(new, b)
        cut = M.packed_forward_masked(new, dict(b, loss_ok=lm))
    assert torch.equal(full["_q"], cut["_q"])                                      # what the model sees is the same
    assert not torch.equal(full["loss"], cut["loss"])
    none = M.packed_forward_masked(new, dict(b, loss_ok=torch.zeros_like(b["mask"])))
    assert float(none["loss"]) == 0.0                                              # nothing scored -> zero loss, not NaN
