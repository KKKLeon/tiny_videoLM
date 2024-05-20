from omegaconf import OmegaConf

cfg = OmegaConf.load("eval_configs/timechat.yaml")
print(cfg.model)