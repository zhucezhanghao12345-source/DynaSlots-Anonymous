import hydra
from omegaconf import OmegaConf
import pathlib
from train import DynaSlotsPolicyWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'dynaslots', 'config'))
)
def main(cfg):
    workspace = DynaSlotsPolicyWorkspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()
