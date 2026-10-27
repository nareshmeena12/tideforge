# configs

`ft3_main_cfg.json`: ft3_main's training config, dumped from the checkpoint (needed by the E0.1 parity test):

    python -c "import torch,json; c=torch.load('ft3_main.pt',map_location='cpu',weights_only=False)['cfg']; json.dump(c,open('configs/ft3_main_cfg.json','w'),indent=1,default=str)"
