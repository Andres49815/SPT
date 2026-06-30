"""
SAM 3 wrapper for image segmentation fine-tuning with SPT support.

This wrapper uses the local `sam3` repository cloned into the workspace root.
It converts COCO box prompts into the prompt objects expected by the SAM3 image model.
"""

import os
import sys
from typing import Optional, Tuple
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use the train-safe code paths in SAM3 unless user explicitly overrides.
os.environ.setdefault("USE_PERFLIB", "0")

# Ensure the cloned sam3 repository is importable as `sam3.*`.
SAM3_REPO_ROOT = Path(__file__).resolve().parents[1] / "sam3"
if SAM3_REPO_ROOT.is_dir() and str(SAM3_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_REPO_ROOT))

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt


def _patch_sam3_train_safe_mlp() -> None:
    """Replace SAM3's inference-only fused MLP op with a grad-safe fallback.

    SAM3's vitdet MLP calls `addmm_act` from `sam3.perflib.fused`, which raises
    `ValueError("Expected grad to be disabled.")` when autograd is enabled.
    This patch keeps inference behavior acceptable while enabling fine-tuning.
    """
    try:
        import sam3.model.vitdet as sam3_vitdet
    except ImportError:
        return

    if getattr(sam3_vitdet, "_spt_train_safe_mlp_patched", False):
        return

    def _train_safe_addmm_act(activation, linear, mat1):
        out = linear(mat1)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return F.relu(out)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return F.gelu(out)
        raise ValueError(f"Unexpected activation {activation}")

    sam3_vitdet.addmm_act = _train_safe_addmm_act
    sam3_vitdet._spt_train_safe_mlp_patched = True


class SAM3ImageSegmentationWrapper(nn.Module):
    """Minimal SAM3 wrapper for image segmentation fine-tuning."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        freeze_prompt_encoder: bool = True,
        freeze_image_encoder: bool = False,
        num_classes: int = 1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.freeze_prompt_encoder = freeze_prompt_encoder
        self.freeze_image_encoder = freeze_image_encoder

        # Ensure SAM3 ViT blocks use a differentiable MLP path during training.
        _patch_sam3_train_safe_mlp()

        try:
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise ImportError(
                "sam3 package not found. Keep the cloned sam3 repo at the workspace root "
                "and install its dependencies before training."
            ) from exc

        if checkpoint and os.path.isfile(checkpoint):
            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint,
                load_from_HF=False,
                eval_mode=False,
            )
        else:
            # Default SAM3 checkpoint from Hugging Face is used by the model builder.
            self.model = build_sam3_image_model(eval_mode=False)

        backbone = getattr(self.model, "backbone", None)
        vision_backbone = getattr(backbone, "vision_backbone", None) if backbone is not None else None
        image_encoder = getattr(vision_backbone, "trunk", None) or vision_backbone or backbone
        if image_encoder is None:
            raise AttributeError("Could not locate the SAM3 vision backbone.")
        self.image_encoder = cast(nn.Module, image_encoder)

        if freeze_prompt_encoder:
            prompt_modules = [
                getattr(self.model, "geometry_encoder", None),
                getattr(getattr(self.model, "backbone", None), "language_backbone", None),
            ]
            for module in prompt_modules:
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False

        if freeze_image_encoder and self.image_encoder is not None:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) ImageNet-normalized tensors from the dataset.
            boxes: (B, 4) normalized XYXY boxes in [0, 1].

        Returns:
            masks: (B, 1, H, W) mask logits.
            scores: (B, 1) confidence scores.
        """
        original_height, original_width = images.shape[-2:]
        images = self._preprocess_images(images)
        batch_size = images.shape[0]

        backbone_out = {"img_batch_all_stages": images}
        backbone_out.update(self.model.backbone.forward_image(images))
        backbone_out.update(self.model.backbone.forward_text(["visual"], device=images.device))

        box_embeddings = self._xyxy_to_cxcywh(boxes).unsqueeze(0)
        box_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=images.device)
        box_labels = torch.ones((1, batch_size), dtype=torch.long, device=images.device)
        # SAM3 geometry encoder expects points with last dim = 2 (x, y), normalized.
        point_embeddings = torch.empty(0, batch_size, 2, device=images.device)
        point_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=images.device)
        point_labels = torch.empty(0, batch_size, dtype=torch.long, device=images.device)

        find_input = FindStage(
            img_ids=torch.arange(batch_size, device=images.device, dtype=torch.long),
            text_ids=torch.zeros(batch_size, device=images.device, dtype=torch.long),
            input_boxes=box_embeddings,
            input_boxes_mask=box_mask,
            input_boxes_label=box_labels,
            input_points=point_embeddings,
            input_points_mask=point_mask,
        )

        geometric_prompt = Prompt(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels,
        )

        # SAM3 forward_grounding computes matching during training and expects
        # a non-null find_target. For SPT segmentation, we optimize BCE/Dice on
        # predicted masks directly, so we use eval mode for this call to bypass
        # matcher while keeping gradients enabled.
        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
        finally:
            if was_training:
                self.model.train()

        pred_logits = out["pred_logits"]
        pred_masks = out["pred_masks"]

        best_idx = pred_logits.squeeze(-1).argmax(dim=1)
        batch_idx = torch.arange(pred_logits.shape[0], device=pred_logits.device)

        masks = pred_masks[batch_idx, best_idx].unsqueeze(1)
        scores = pred_logits[batch_idx, best_idx].sigmoid().unsqueeze(1)

        if masks.shape[-2:] != (original_height, original_width):
            masks = F.interpolate(
                masks,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )

        return masks, scores

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Convert ImageNet-normalized tensors into the preprocessing expected by SAM3."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images * std + mean).clamp(0.0, 1.0)
        images = F.interpolate(images, size=(1008, 1008), mode="bilinear", align_corners=False)
        return images * 2.0 - 1.0

    def _xyxy_to_cxcywh(self, boxes: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)

    def get_image_encoder(self) -> nn.Module:
        return self.image_encoder

    def no_weight_decay(self):
        return {"image_encoder.patch_embed", "image_encoder.pos_embed"}


def build_sam3_segmentation_model(
    checkpoint: Optional[str] = None,
    freeze_prompt_encoder: bool = True,
    freeze_image_encoder: bool = False,
    num_classes: int = 1,
) -> SAM3ImageSegmentationWrapper:
    return SAM3ImageSegmentationWrapper(
        checkpoint=checkpoint,
        freeze_prompt_encoder=freeze_prompt_encoder,
        freeze_image_encoder=freeze_image_encoder,
        num_classes=num_classes,
    )
