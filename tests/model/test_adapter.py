"""The covariate adapter: equals the backbone at init, trained or not on lone series and target-only groups, firewall on its own path,
causality and packed isolation, gradients on adapter parameters only."""
import pytest
import torch

from tideforge import model as TM
from tideforge.model import adapter as AD
from tideforge.model.ft3_model import ROLE_KNOWN, ROLE_PAST, ROLE_TARGET

CFG = dict(patch_len=4, d_model=16, d_kv=4, n_heads=2, d_ff=32, n_layers=4, dropout=0.0, arcsinh=True, max_pass_patches=8,
           group_attn=True, group_lanes=4, horizon_loss_share=0.5, decoder="causal", skip_group_attn=True, mask_channel=True)
P, V = 6, 4
R, ROLES = 2 * V, [ROLE_TARGET, ROLE_TARGET, ROLE_PAST, ROLE_KNOWN]


def pair(every=1):
    torch.manual_seed(0)
    base = TM.build(dict(CFG)).eval()
    ad = TM.build(dict(CFG, adapter_every=every)).eval()
    ad.load_state_dict(base.state_dict(), strict=False)
    return base, ad


def batch(cov_blocks=(0, 1), targets_only=()):
    """Two blocks, one group each: lanes as ROLES where the block is a covariate block; lone lanes otherwise."""
    torch.manual_seed(1)
    x = torch.randn(R, P * 4)
    obs = torch.ones_like(x, dtype=torch.bool)
    role = torch.zeros(R, P, dtype=torch.long)
    gid = torch.zeros(R, P, dtype=torch.long)
    seg = torch.arange(R).unsqueeze(1).expand(R, P)
    for b in range(2):
        for v in range(V):
            r = b * V + v
            if b in cov_blocks:
                role[r], gid[r] = ROLES[v], b * V
            elif b in targets_only:
                gid[r] = b * V
            else:
                gid[r] = r
    return x, obs, seg, gid, role


def core(model, x, obs, seg, gid, role):
    seg_pt, st, attn = model._packed_geometry(seg, R)
    return model._core(x, obs, seg_pt, st, R, attn_mask=attn, lanes=V, gid=gid, role=role)[0]


def scramble(ad):
    with torch.no_grad():
        for p in AD.adapter_parameters(ad):
            p.add_(torch.randn_like(p))


def test_counts_and_options():
    _, one = pair(1)
    _, two = pair(2)
    assert len(AD.adapter_parameters(one)) == 2 * len(AD.adapter_parameters(two))
    assert [b.adapter is not None for b in two.blocks] == [False, True, False, True]
    assert TM.options_of(dict(CFG, adapter_every=2))["adapter_every"] == 2
    with pytest.raises(NotImplementedError):
        TM.build(dict(CFG, adapter_every=3))


def test_init_equals_backbone():
    base, ad = pair()
    b = batch()
    assert torch.equal(core(base, *b), core(ad, *b))


@pytest.mark.parametrize("spec", [dict(cov_blocks=()), dict(cov_blocks=(), targets_only=(0, 1))])
def test_no_covariate_equals_backbone_after_training(spec):
    base, ad = pair()
    scramble(ad)
    b = batch(**spec)
    assert torch.equal(core(base, *b), core(ad, *b))


def test_trained_adapter_changes_covariate_groups_only():
    base, ad = pair()
    scramble(ad)
    b = batch(cov_blocks=(0,))
    qa, qb = core(ad, *b), core(base, *b)
    assert not torch.equal(qa[:V], qb[:V]) and torch.equal(qa[V:], qb[V:])


def test_left_padded_path_skips_the_adapter():
    base, ad = pair()
    scramble(ad)
    x = torch.randn(3, 24)
    assert torch.equal(base(x)["quantile_preds_all"], ad(x)["quantile_preds_all"])


def tokens():
    torch.manual_seed(2)
    x = torch.randn(R, P, 16, requires_grad=True)
    _, _, _, gid, role = batch()
    return x, role, gid


def reads(a, x, role, gid, r_out, r_in):
    out = a(x, V, role, gid)
    g, = torch.autograd.grad(out[r_out].sum(), x, allow_unused=False)
    return g[r_in].abs().sum() > 0


def adapter_block():
    a = AD.CovAdapter(16, 4, 2, 1e-6)
    torch.nn.init.normal_(a.out_proj.weight)
    return a


@pytest.mark.parametrize("q,k,allowed", [(0, 1, True), (0, 2, True), (0, 3, True), (2, 3, True), (2, 0, False), (3, 0, False),
                                         (3, 2, False), (2, 1, False)])
def test_firewall(q, k, allowed):
    x, role, gid = tokens()
    assert bool(reads(adapter_block(), x, role, gid, q, k)) is allowed


def test_packed_isolation():
    x, role, gid = tokens()
    assert not reads(adapter_block(), x, role, gid, 0, V + 3)


def test_causal_through_the_model():
    _, ad = pair()
    scramble(ad)
    x, obs, seg, gid, role = batch()
    y = x.clone()
    y[0, 16:] += 5.0                                              # the target from patch 4 on
    a, b = core(ad, x, obs, seg, gid, role), core(ad, y, obs, seg, gid, role)
    assert torch.equal(a[:, :4], b[:, :4]) and not torch.equal(a[:, 4:], b[:, 4:])


def test_only_adapter_gets_gradients():
    _, ad = pair()
    n = AD.freeze_backbone(ad)
    assert n == sum(p.numel() for p in ad.parameters() if p.requires_grad)
    core(ad, *batch()).sum().backward()
    for name, p in ad.named_parameters():
        assert (p.grad is not None) == (".adapter." in name), name
