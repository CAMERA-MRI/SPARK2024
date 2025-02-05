from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data) # Instantiates the data module from the configuration

    log.info(f"Instantiating model <{cfg.model._target_}> with adapters") # cfg.model._target_ specified the Python class to instantiate. In the example, 'train.yaml', this is 'src.model.MNIST'.
    model: LightningModule = hydra.utils.instantiate(cfg.model) # Instantiates the model from the configuration; the model with adapters must be passed here. 'hydra.utils.instantiate(cfg.model)' uses the rest of the configuration in 'cfg.model' to instantiate the model. This includes parameters like 'num_classes', 'adapter_dim_ratio',..

    # Load pretrained model checkpoint
    if cfg.get("ckpt_path_for_finetuning"):
        log.info(f"Loading pretrained model from {cfg.ckpt_path_for_finetuning}")
        pretrained_checkpoint = torch.load(cfg.ckpt_path_for_finetuning)
        
        # Changing the keys in pretrained_checkpoint to match the keys in model_with_adapter_state_dict
        for layer_name in list(pretrained_checkpoint['state_dict'].keys()):
            if ('dec_block' in layer_name) or ('bottleneck' in layer_name) or ('enc_block' in layer_name):
                # net.enc_block_0.0.conv1.weight  -->  net.enc_block_0.0.0.conv1.weight
                # net.bottleneck.0.conv1.weight  -->  net.bottleneck.0.0.conv1.weight
                # net.dec_block_2.1.norm.bias  -->  net.dec_block_2.0.1.norm.bias
                
                layer_name_split = layer_name.split(".")
                layer_name_split.insert(2, "0")
                new_layer_name = ".".join(layer_name_split)
                # print("Old layer_name: ", layer_name, "New layer name", new_layer_name)

                # deletes the respective key, returns the associated value of that old key
                layer_values = pretrained_checkpoint['state_dict'].pop(layer_name)

                # assign the returned value to the new_layer_name
                pretrained_checkpoint['state_dict'][new_layer_name] = layer_values
            else:
                pass
            
        # assert if the len in new_layer_name (keys) in pretrained_checkpoint['state_dict'] matches those in new model with adapter state_dict, except the fully connected adapter's weights and biases
        assert len(list(pretrained_checkpoint['state_dict'].keys())) == len(list(x for x in model.state_dict().keys() if 'fc' not in x and 'dice_loss' not in x))

        # assert if every modified key on pretrained_checkpoint state dict have match in new model with adapter state_dict
        count = 0
        for modified_key in pretrained_checkpoint['state_dict'].keys():
            if modified_key not in model.state_dict().keys():
                count += 1
                print(modified_key)
        assert count == 0

        # Load weights/biases (i.e.state_dict) from pretrained checkpoint to new model with adapter that match the layer names. strict=False ignores non-matching keys, allowing the model to load even if the state dictionary does not perfectly match the model
        model.load_state_dict(pretrained_checkpoint['state_dict'], strict=False)

        # Freeze all layers except the adapter layers
        log.info("Freezing all layers except the adapter layers...")
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = { # collects all instantiated objects for easy access
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

# The configuration files are specified by the '@hydra.main' decorator in the sciprt. In this case, the 'config_path' is set to '../configs' and the 'config_name' is set to 'train.yaml'. This means Hydra will look for a 'train.yaml' file in the '../configs' directory relative to the script's location. 
# 'cfg' is a Hydra configuration object that contains all the settings for the experiment. It is created from the configuration files and command-line arguments. The 'model' section of 'cfg' specifies the class and parameters for the model and similarly, ...
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
