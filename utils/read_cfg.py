from omegaconf import OmegaConf

def read_cfg(cfgdir):
    args = OmegaConf.load(cfgdir)
    return args