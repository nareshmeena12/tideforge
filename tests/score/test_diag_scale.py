"""diag_scale's default task names follow the manifest naming, and its name matching works on the release fixture."""
from tideforge.score import diag_scale as DS
from tideforge.score import release as RL


def test_default_tasks_use_manifest_naming(release):
    names = list(RL.Release(release).manifest["task"])
    assert all(n.count("/") == 2 for n in names)                       # fixture: <config>/<horizon>/<target|joint>
    for t in DS.TASKS:
        assert t.count("/") == 2 and t.endswith("/target"), t
        cfg, hc, _ = t.split("/")
        assert "__" in cfg and hc in ("short", "medium", "long"), t


def test_match_exact_and_prefix(release):
    names = [s["row"]["task"] for s in RL.Release(release).tasks("full", "tuning")]
    full = next(n for n in names if n.endswith("/target"))
    hit, miss = DS.match([full], names)
    assert hit == [full] and not miss
    pre = full.rsplit("/", 1)[0]
    hit, miss = DS.match([pre, "nope__x/short"], names)
    assert full in hit and all(n.rsplit("/", 1)[0] == pre for n in hit) and miss == ["nope__x/short"]
