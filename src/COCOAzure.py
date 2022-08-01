"""Defines COCOAzureEvaluator and COCOAzureTrainer."""

# Imports Python builtins.
import os
import itertools

# Imports Detectron2 libraries.
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

# Imports other pip libraries.
from azureml.core.run import Run
import numpy as np
import pdb
from typing import Any, Dict, List, Set
import torch
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.data import MetadataCatalog, build_detection_train_loader
from d2.detr.dataset_mapper import DetrDatasetMapper

class COCOAzureEvaluator(COCOEvaluator):   
    """Evaluates a Detectron2 model on Azure ML."""

    def __init__(self, dataset_name, *args):   
        """Initializes a COCOAzureEvaluator."""

        super().__init__(dataset_name, *args)
        self.dataset_name = dataset_name

    def evaluate(self, img_ids=None):
        """Evaluates model and logs metrics to Azure ML."""

        metrics_dict = super().evaluate(img_ids)
        writer = Run.get_context()
       
        # Infers train/test from dataset name and logs metrics.
        if "train" in self.dataset_name:
            writer.log("Train/AP", np.float(metrics_dict["bbox"]["AP"]))
            writer.log("Train/AP50", np.float(metrics_dict["bbox"]["AP50"]))
            writer.log("Train/AP75", np.float(metrics_dict["bbox"]["AP75"]))
        else:
            writer.log("Test/AP", np.float(metrics_dict["bbox"]["AP"]))
            writer.log("Test/AP50", np.float(metrics_dict["bbox"]["AP50"]))
            writer.log("Test/AP75", np.float(metrics_dict["bbox"]["AP75"]))
        return metrics_dict

class COCOAzureTrainer(DefaultTrainer):
    """Trains a Detectron2 model on Azure ML."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Creates COCOAzureEvaluator for validating model."""
        # Creates default output folder.
        if output_folder is None:
                if cfg.OUTPUT_DIR:
                    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                    output_folder = cfg.OUTPUT_DIR
                else:
                    os.makedirs("COCO_eval", exist_ok=True)
                    output_folder = "COCO_eval"

        # Creates COCOAzureEvaluator.
        evaluator = COCOAzureEvaluator(dataset_name, cfg, True, output_folder)

        return evaluator


class COCOAzureTransformerTrainer(DefaultTrainer):
    """Trains a Detectron2 model on Azure ML."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Creates COCOAzureEvaluator for validating model."""
        # Creates default output folder.
        if output_folder is None:
                if cfg.OUTPUT_DIR:
                    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                    output_folder = cfg.OUTPUT_DIR
                else:
                    os.makedirs("COCO_eval", exist_ok=True)
                    output_folder = "COCO_eval"

        # Creates COCOAzureEvaluator.
        evaluator = COCOAzureEvaluator(dataset_name, cfg, True, output_folder)

        return evaluator
    @classmethod
    def build_train_loader(cls, cfg):
        if "Detr" == cfg.MODEL.META_ARCHITECTURE:
            mapper = DetrDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

