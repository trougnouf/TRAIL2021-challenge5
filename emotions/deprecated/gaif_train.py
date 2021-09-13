"""Train an emotion recognizer on RAF-basic dataset w/ single class"""
from mmf_cli.run import run
from mmf.common.registry import registry
from gaif.utils.env import setup_imports

# Adds GAIF functionality to MMF
setup_imports()
# "Initialize" the registry
registry.mapping["state"] = {}

opts = [
    "config='configs/config_rafbasic.yaml'",
    "model=model_timm",
    "dataset=classification_raf_basic",
]

run(opts=opts)
