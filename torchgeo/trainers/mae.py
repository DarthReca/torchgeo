import os

import torch
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms.mae_transform import MAETransform
from timm.models import VisionTransformer, create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torchvision.models._api import WeightsEnum

from ..datasets.utils import Sample
from ..models import get_weight
from .base import BaseTask
from .utils import extract_backbone, load_state_dict


class MAE(BaseTask):
    def __init__(
        self,
        model: str = "vit_base_patch32_224",
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        transform: nn.Module | None = None,
        decoder_dim: int = 512,
        lr: float = 1.5e-4,
        decoder_num_heads: int = 8,
        weight_decay: float = 0.05,
    ) -> None:
        """MAE training

        Args:
            lr (float, optional): _Should be 1.5e-4 * batch_size / 256.
        """
        super().__init__()
        self.transform = transform if transform is not None else MAETransform()

        self.weights = weights
        self.lr = lr
        self.weight_decay = weight_decay

    def configure_losses(self) -> None:
        self.criterion = nn.MSELoss(reduction="none")

    def configure_models(self) -> None:
        model: str = self.hparams['model']
        weights = self.weights
        in_channels: int = self.hparams['in_channels']

        vit = create_model(
            model, in_chans=in_channels, num_classes=0, pretrained=weights is True)
        assert isinstance(vit, VisionTransformer), (
            "Only ViT models are supported"
        )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(
                    progress=True)  # type: ignore[invalid-argument-type]
            load_state_dict(self.backbone, state_dict)

        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]
        self.sequence_length = self.backbone.sequence_length

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=self.hparams['decoder_dim'],
            decoder_depth=1,
            decoder_num_heads=self.hparams['decoder_num_heads'],
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            in_chans=vit.patch_embed.proj.in_channels,
        )

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(
            x_masked, idx_keep, x_decode.type_as(x_masked)
        )

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            views = self.transform(batch["image"].float())
        images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)

        # per-sample loss for std logging
        loss_per_sample = self.criterion(
            x_pred, target
        )
        loss_per_sample = loss_per_sample.mean(
            dim=list(range(1, loss_per_sample.ndim))
        )  # (B,)
        loss = loss_per_sample.mean()

        psnr = -10.0 * torch.log10(loss.detach().clamp(min=1e-10))

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # Near-zero std indicates that the model is learning uniformly across samples.
        self.log(
            "train_loss_std",
            loss_per_sample.std(),
            on_step=True,
            on_epoch=False,
        )
        # Near-zero means that the model is predicting a constant value, which is a common failure mode for MAE training.
        self.log("pred_std", x_pred.std(), on_step=True, on_epoch=False)
        # If this is very low, the model is getting "easy" patches (smooth background) and the loss won't be meaningful.
        self.log("target_std", target.std(), on_step=True, on_epoch=False)
        # PSNR is a common metric for image reconstruction quality. Higher is better, and values above ~30 indicate good reconstruction.
        self.log("psnr", psnr, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.95)
        )
        max_epochs = 800
        if self.trainer and self.trainer.max_epochs is not None:
            max_epochs = self.trainer.max_epochs
        scheduler = CosineLRScheduler(
            optim,
            t_initial=max_epochs,
            lr_min=0,
            warmup_t=40,
            warmup_lr_init=0,
            cycle_limit=1
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op, does nothing."""

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op, does nothing."""
