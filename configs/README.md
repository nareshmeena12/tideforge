# configs

`ft3_main_cfg.json`: ft3_main's training config, dumped from the checkpoint (needed by the E0.1 parity test):

    python -c "import torch,json; c=torch.load('ft3_main.pt',map_location='cpu',weights_only=False)['cfg']; json.dump(c,open('configs/ft3_main_cfg.json','w'),indent=1,default=str)"

`exp/<ID>.yaml`: one config per experiment ID (generator settings and switches vs a base; see `tideforge/expcfg.py`). An
experiment is a config, never a code change. `exp/EXAMPLE.yaml` shows the format with the test switch `test_post`. Tools take
`--exp <ID>`: `python tools/gallery.py --exp <ID>` (PNG, off vs on, same seeds), `python tools/gen_audit.py --exp <ID>`.
