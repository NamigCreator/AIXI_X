from collections import defaultdict
from pathlib import Path
import fire
import numpy as np
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import matplotlib.pyplot as plt
from neptune.types import File

import torch

torch.set_float32_matmul_precision("medium")
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

import sys

sys.path.append("code/")
from denoising.models import (
    Generator,
    Descriminator,
    VGG19_intermediate_layers_only,
    TransformerLike,
)
from denoising.data_utils import CTBrainDataModule, AIMIBrainDataModule, apply_canny


class Model(LightningModule):
    def __init__(
        self,
        lambda_style=10,
        lambda_perceptual=0.1,
        lambda_adversarial=0.5,
        patch_size=16,
        img_size=256,
        checkpoint_dir="checkpoints/",
        anomalies_dir="anomalies/",
        loss_only_masked=False,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.generator = Generator()
        self.descriminator = Descriminator()
        self.vgg19 = VGG19_intermediate_layers_only()

        self.lambda_sty = lambda_style
        self.lambda_per = lambda_perceptual
        self.lambda_adv = lambda_adversarial
        self.loss_only_masked = loss_only_masked

        self.img_size = img_size
        self.patch_size = patch_size

        # to store some examples in validation
        self.buffer = []
        self.checkpoint_dir = checkpoint_dir
        self.anomalies_dir = anomalies_dir

    def configure_optimizers(self):
        self.g_opt = SGD(
            params=self.generator.parameters(),
            lr=0.0005,
            momentum=0.9,
            weight_decay=5e-4,
        )
        self.d_opt = SGD(
            params=self.descriminator.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )

        self.scheduler_g = ReduceLROnPlateau(
            self.g_opt, mode="min", factor=0.1, patience=30, min_lr=5e-6
        )

        self.scheduler_d = ReduceLROnPlateau(
            self.d_opt, mode="min", factor=0.1, patience=30, min_lr=5e-6
        )

        return self.g_opt, self.d_opt

    def training_step(self, batch, batch_idx):

        self.g_opt.zero_grad()

        x_true, edges, target, name = batch

        mask_size = random.randint(16, 48)
        mask = self.get_random_mask(x_true, mask_size)
        x_masked = x_true * mask + (1 - mask)  # masked region is filled with 1

        x_masked_w_edges = torch.concat([x_masked, edges], dim=1)
        x_reconstructed = self.generator(x_masked_w_edges)

        if self.loss_only_masked:
            losses = self.calculate_losses(x_true * mask, x_reconstructed * mask, mask)
        else:
            losses = self.calculate_losses(x_true, x_reconstructed, mask)

        g_loss = (
            losses["reconstruction_loss"]
            + self.lambda_adv * losses["adversarial_loss"]
            + self.lambda_per * losses["perceptual_loss"]
            + self.lambda_sty * losses["style_loss"]
        )
        self.manual_backward(g_loss)
        self.g_opt.step()

        ### train disriminator
        self.d_opt.zero_grad()

        d_loss = losses["fake_loss"] + losses["real_loss"]

        if batch_idx % 10 == 0 and self.current_epoch > 1:
            # to prevent discriminator be too strong from the start
            self.manual_backward(d_loss)
            self.d_opt.step()

        losses.update({"g_loss": g_loss, "d_loss": d_loss})
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx % 500 == 0:
            self.__log_image(x_true, "training/real_images")
            # edge
            self.__log_image(x_masked_w_edges[:, -1, :, :], "training/edges")
            self.__log_image(x_reconstructed, "training/reconstructed_images")

        ### lr scheduler
        # if self.trainer.is_last_batch:
        # self.scheduler_g.step(self.trainer.callback_metrics["g_loss"])
        # self.scheduler_d.step(self.trainer.callback_metrics["d_loss"])

    def get_random_mask(self, x, mask_size=16):
        """tensor of ones with zeros on the place of a mask"""
        mask = torch.ones_like(x)

        rows = cols = x.shape[2]

        for i in range(mask.shape[0]):
            top_left_x = random.randint(rows // 2 - rows // 4, rows // 2 + rows // 4)
            top_left_y = random.randint(cols // 2 - cols // 4, cols // 2 + cols // 4)

            mask[
                i,
                :,
                top_left_x : top_left_x + mask_size,
                top_left_y : top_left_y + mask_size,
            ] = 0.0

        return mask

    def get_mask_grid(self, x, K):

        N, C, height, width = x.shape

        assert height == width

        # Calculate the number of patches along one dimension
        num_patches_x = width // K
        num_patches_y = height // K

        # Fill mask
        mask = torch.ones(
            size=(num_patches_x * num_patches_y, C, height, width), dtype=int
        )
        count = 0
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                mask[count, :, i * K : (i + 1) * K, j * K : (j + 1) * K] = 0
                count += 1

        return mask

    def calculate_losses(
        self, x_true, x_reconstructed, mask, calc_descriminator_loss=True
    ):

        rec_loss = self.reconstruction_loss(x_reconstructed, x_true)
        perc_loss, style_loss = self.vgg_based_losses(x_reconstructed, x_true, mask)

        rec_label = self.descriminator(x_reconstructed)
        true_label = self.descriminator(x_true)
        valid = torch.ones_like(rec_label).to(rec_label.device)
        fake = torch.zeros_like(rec_label).to(rec_label.device)

        losses = {
            "reconstruction_loss": rec_loss,
            "perceptual_loss": perc_loss,
            "style_loss": style_loss,
        }
        if calc_descriminator_loss:
            # adversarial loss for generator
            adv_loss = self.adversarial_loss(rec_label, valid)

            # parts of adversarial loss for discriminator
            real_loss = self.adversarial_loss(true_label, valid)
            fake_loss = self.adversarial_loss(rec_label.detach(), fake)
            losses.update(
                {
                    "adversarial_loss": adv_loss,
                    "real_loss": real_loss,
                    "fake_loss": fake_loss,
                }
            )
        return losses

    def adversarial_loss(self, x, targets):
        return F.binary_cross_entropy_with_logits(x, targets)

    def style_loss(self, x_true, x_rec, mask):
        intermediate_layers_true, intermediate_layers_rec = (
            self.get_intermediate_maps_from_vgg(x_true * mask, x_rec * mask)
        )
        style_loss = 0.0
        layer_names = self.vgg19.return_layers.values()
        for name in layer_names:
            style_loss += F.l1_loss(
                self.gram_matrix(intermediate_layers_rec[name]),
                self.gram_matrix(intermediate_layers_true[name]),
            )
        style_loss /= len(layer_names)
        return style_loss

    def perceptual_loss(self, x_true, x_rec):
        intermediate_layers_true, intermediate_layers_rec = (
            self.get_intermediate_maps_from_vgg(x_true, x_rec)
        )
        perceptual_loss = 0.0
        layer_names = self.vgg19.return_layers.values()
        for name in layer_names:
            perceptual_loss += F.l1_loss(
                intermediate_layers_rec[name], intermediate_layers_true[name]
            )
        perceptual_loss /= len(layer_names)
        return perceptual_loss

    def vgg_based_losses(self, x_true, x_rec, mask=None):
        perceptual_loss = self.perceptual_loss(x_true, x_rec)
        style_loss = self.style_loss(x_true, x_rec, mask)
        return perceptual_loss, style_loss

    def reconstruction_loss(self, x, y):
        return F.l1_loss(x, y)

    def get_intermediate_maps_from_vgg(self, x_true, x_rec):
        # vgg19 takes 3 channels as input
        x_true = x_true.repeat(1, 3, 1, 1)
        x_rec = x_rec.repeat(1, 3, 1, 1)
        intermediate_layers_true = self.vgg19(x_true)
        intermediate_layers_rec = self.vgg19(x_rec)
        return intermediate_layers_true, intermediate_layers_rec

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def predict_step(self, batch, batch_idx, mask_size=16, instead_of_mask=None):
        x_true, edges, target, name = batch
        N, C, H, W = x_true.shape
        if mask_size is None:
            x_masked = x_true
        else:
            mask = self.get_mask_grid(x_true, K=mask_size)
            x_masked = x_true.repeat(mask.shape[0], 1, 1, 1)
            edges = edges.repeat(mask.shape[0], 1, 1, 1)
            mask = mask.repeat(N, 1, 1, 1)
            mask = mask.to(x_true.device)
            if instead_of_mask is None:
                x_masked = x_masked * mask + (1 - mask)
                pass
            else:
                instead_of_mask = instead_of_mask.repeat(mask.shape[0], 1, 1, 1)
                x_masked = x_masked * mask + (1 - mask) * instead_of_mask

        x_masked_w_edges = torch.concatenate([x_masked, edges], dim=1)

        if x_masked_w_edges.shape[0] > 1000:
            n_chunks = x_masked_w_edges.shape[0] // 1000
            chunks = torch.split(x_masked_w_edges, n_chunks)
            results = []
            for chunk in chunks:
                x_rec_chunk = self.generator(chunk)
                results.append(x_rec_chunk)
            x_reconstructed = torch.concatenate(results, dim=0)
            del chunks
            del results
        else:
            x_reconstructed = self.generator(x_masked_w_edges)

        if mask_size is None:
            return x_reconstructed

        x_reconstructed = (1 - mask) * x_reconstructed
        x_reconstructed = x_reconstructed.view(N, -1, C, H, W).sum(dim=1)

        return x_reconstructed

    def validation_step(self, batch, batch_idx):

        x_true, edges, target, name = batch
        x_reconstructed = self.predict_step(batch, batch_idx, mask_size=16)

        mask = torch.ones_like(x_true)
        losses = self.calculate_losses(x_true, x_reconstructed, mask)
        self.buffer.append(losses)

        if True:
            self.__log_image(x_reconstructed, folder="validation/x_reconstructed")
            self.__log_image(
                torch.abs(x_reconstructed - x_true), folder="validation/x_diff"
            )
            self.__log_image(x_true, folder="validation/x_true")

    def on_train_epoch_end(self) -> None:

        self.trainer.save_checkpoint(
            Path(self.checkpoint_dir).joinpath(f"epoch_{self.current_epoch}.ckpt")
        )

    def on_validation_epoch_end(self) -> None:
        # pass
        name2losses = defaultdict(list)
        for example_losses in self.buffer:
            for name, loss in example_losses.items():
                name2losses[name].append(loss.cpu().item())
        for name, losses in name2losses.items():
            folder = f"validation/{name}"
            self.logger.experiment[folder].append(np.mean(losses))

        self.buffer = []

    def on_test_epoch_start(self) -> None:
        """remove models that do not take part in testing"""
        self.descriminator = None
        self.vgg19 = None
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        x_true, edges, target, name = batch
        # batch = torch.zeros_like(x_true).to(x_true.device), edges, target, name

        results = defaultdict(lambda: defaultdict(dict))
        mask_sizes = [None, 4, 8, 16, 24]
        # mask_sizes = [16, 24]
        name = name[0]
        for k in mask_sizes:

            results[name][k]["x_true"] = x_true.cpu().squeeze()

            x_reconstructed = self.predict_step(batch, batch_idx, mask_size=k)
            results[name][k]["x_rec"] = x_reconstructed.cpu().squeeze()

            x_diff = torch.abs(x_true - x_reconstructed)
            results[name][k]["x_diff"] = x_diff.cpu().squeeze()

            x_diff = self.post_process_image(x_diff)
            results[name][k]["x_diff_smooth"] = x_diff.cpu().squeeze()

            x_diff = self.post_process_image(x_diff)
            x_diff[x_diff < (20 / 255)] = 0
            results[name][k]["x_diff_threshold"] = x_diff.cpu().squeeze()

            x_second_step = self.generator(
                torch.concatenate([x_reconstructed, edges], dim=1)
            )
            results[name][k]["x_rec_as_input"] = x_second_step.cpu().squeeze()

            x_second_as_mask = self.predict_step(
                batch, batch_idx, mask_size=k, instead_of_mask=x_reconstructed
            )
            results[name][k]["x_rec_as_mask"] = x_second_as_mask.cpu().squeeze()

        self._log_collage(results, mask_sizes=mask_sizes, folder=self.anomalies_dir)
    
        return results
    
    def post_process_image(self, x):

        nb_channels = 1
        weights = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        weights = weights / 9.0
        weights = weights.to(x.device)
        weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)

        return F.conv2d(x, weights)

    def __log_image(self, x, folder, mask=None):
        """log first image from batch"""
        img = x[0]
        img = img.cpu()
        if mask is not None:
            # make rgb from grayscale
            img = img.repeat(3, 1, 1)
            mask = mask[0]
            mask = mask.cpu()
            if mask.shape[0] == 1:
                # make rgb from grayscale
                mask = mask.repeat(3, 1, 1)
            # invert zeros to ones
            red_mask = torch.ones_like(mask) - mask
            # zerofy green and blue channels
            red_mask[1, :, :] = 0.0
            red_mask[2, :, :] = 0.0
            img = img + red_mask

        # if a grayscale image (H, W) do nothing,
        # else transfer (C, H, W) -> (H, W, C)
        if len(img.shape) > 2:
            last_channel_image = torch.moveaxis(img, 0, -1)
        else:
            last_channel_image = img
        self.logger.experiment[folder].append(File.as_image(last_channel_image))

    def _log_collage(self, results, mask_sizes, folder):
        for name in results:
            fig, ax = plt.subplots(7, len(mask_sizes), figsize=(24, 24))
            for j, k in enumerate(mask_sizes):
                images = results[name][k]

                ax[0, j].set_title(f"{name}\nmask_size: {k}")
                ax[0, j].imshow(images["x_true"].numpy(), cmap="gist_gray")

                ax[1, j].set_title(f"reconstructed image", size=10)
                ax[1, j].imshow(images["x_rec"].numpy(), cmap="gist_gray")

                ax[2, j].set_title(f"|x_rec - x_true|", size=10)
                ax[2, j].imshow(images["x_diff"].numpy(), cmap="gist_gray")

                ax[3, j].set_title(f"|x_rec - x_true| smoothed", size=10)
                ax[3, j].imshow(images["x_diff_smooth"].numpy(), cmap="gist_gray")

                ax[4, j].set_title(f"|x_rec - x_true| smoothed+threshold", size=10)
                ax[4, j].imshow(images["x_diff_threshold"].numpy(), cmap="gist_gray")

                ax[5, j].set_title(f"x_rec as input", size=10)
                ax[5, j].imshow(images["x_rec_as_input"].numpy(), cmap="gist_gray")

                ax[6, j].set_title(f"x_rec_as_mask", size=10)
                ax[6, j].imshow(images["x_rec_as_mask"].numpy(), cmap="gist_gray")

            # plt.subplots_adjust(wspace=0.1, hspace=1.0)
            plt.tight_layout()
            plt.savefig(f"{folder}/{name}.png")

        return fig


class TransModel(Model):
    def __init__(
        self,
        lambda_style=10,
        lambda_perceptual=0.1,
        lambda_adversarial=0.5,
        patch_size=16,
        img_size=256,
        checkpoint_dir="checkpoints/",
        anomalies_dir="anomalies/",
    ):
        super().__init__(
            lambda_style=lambda_style,
            lambda_perceptual=lambda_perceptual,
            lambda_adversarial=lambda_adversarial,
            patch_size=patch_size,
            img_size=img_size,
            checkpoint_dir=checkpoint_dir,
            anomalies_dir=anomalies_dir,
        )
        self.generator = TransformerLike(
            image_size=256,
            patch_size=16,
            dim=1024,
            depth=2,
            heads=8,
            mlp_dim=2048,
            masking_ratio=0.1,
        )

        self.lambda_sty = lambda_style
        self.lambda_per = lambda_perceptual
        self.lambda_adv = lambda_adversarial

        self.img_size = img_size
        self.patch_size = patch_size

        # to store some examples in validation
        self.buffer = []
        self.checkpoint_dir = checkpoint_dir
        self.anomalies_dir = anomalies_dir

    def configure_optimizers(self):
        self.g_opt = AdamW(
            params=self.generator.parameters(),
            lr=8e-4,
            weight_decay=0.05,
        )
        self.d_opt = SGD(
            params=self.descriminator.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )

        self.scheduler_g = ReduceLROnPlateau(
            self.g_opt, mode="min", factor=0.1, patience=30, min_lr=5e-6
        )

        self.scheduler_d = ReduceLROnPlateau(
            self.d_opt, mode="min", factor=0.1, patience=30, min_lr=5e-6
        )

        return self.g_opt, self.d_opt

    def training_step(self, batch, batch_idx):

        x_true, edges, target, name = batch
        img = torch.cat([x_true, edges], dim=1)

        reconstructed_patches, patches, masked_indices = self.generator.train_forward(
            img
        )
        x_rec = self.generator.from_patch(reconstructed_patches)

        ls_masks = []
        for m_i, p_i in zip(masked_indices, reconstructed_patches):
            mask = [
                torch.zeros_like(p) if i in m_i else torch.ones_like(p)
                for i, p in enumerate(p_i)
            ]
            mask = torch.stack(mask)
            ls_masks.append(mask)
        mask = torch.stack(ls_masks)
        mask = self.generator.from_patch(mask)

        losses = self.calculate_losses(
            x_true * (1 - mask),
            x_rec * (1 - mask),
            mask=mask,
            calc_descriminator_loss=True,
        )

        g_loss = (
            losses["reconstruction_loss"]
            + self.lambda_adv * losses["adversarial_loss"]
            + self.lambda_per * losses["perceptual_loss"]
            + self.lambda_sty * losses["style_loss"]
        )
        self.manual_backward(g_loss)
        self.g_opt.step()

        ### train disriminator
        self.d_opt.zero_grad()

        d_loss = losses["fake_loss"] + losses["real_loss"]

        if batch_idx % 10 == 0 and self.current_epoch > 1:
            # to prevent discriminator be too strong from the start
            self.manual_backward(d_loss)
            self.d_opt.step()

        losses.update({"g_loss": g_loss, "d_loss": d_loss})
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False)

        print(
            f"x_norm, max: {x_rec.max().item()}, min: {x_rec.min().item()}, mean: {x_rec.mean().item()}"
        )

        if batch_idx % 500 == 0:
            # x_true = self.generator.from_patch(patches)
            # print(x_true.shape)
            # 2/0
            self.__log_image(
                x=self.generator.from_patch(patches)[:, :1, :, :],
                folder="training/real_images",
            )
            self.__log_image(
                x=self.generator.from_patch(patches)[:, :1, :, :] * (1 - mask),
                folder="training/real_images_masked",
            )
            self.__log_image(x=x_rec, folder="training/reconstructed_images")

            # normalized reconstructed images
            min_val = x_rec.min(-1)[0].min(-1)[0]
            max_val = x_rec.max(-1)[0].max(-1)[0]
            x_norm = (x_rec - min_val[:, :, None, None]) / (
                max_val[:, :, None, None] - min_val[:, :, None, None]
            )
            print(
                f"x_norm, max: {x_norm.max().item()}, min: {x_norm.min().item()}, mean: {x_norm.mean().item()}"
            )
            self.__log_image(
                x=x_norm, folder="training/reconstructed_images_normalized"
            )

    def test_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        x_true, edges, target, name = batch

        img = torch.cat([x_true, edges], dim=1)
        x_rec = self.generator.test_forward(img)
        mask = torch.ones_like(x_true)
        losses = self.calculate_losses(x_true, x_rec, mask)
        self.buffer.append(losses)

        if True:
            self.__log_image(x=x_rec, folder="validation/x_reconstructed")
            self.__log_image(x=torch.abs(x_rec - x_true), folder="validation/x_diff")
            self.__log_image(x=x_true, folder="validation/x_true")

            # normalized reconstructed images
            min_val = x_rec.min(-1)[0].min(-1)[0]
            max_val = x_rec.max(-1)[0].max(-1)[0]
            x_norm = (x_rec - min_val[:, :, None, None]) / (
                max_val[:, :, None, None] - min_val[:, :, None, None]
            )
            print(
                f"x_norm, max: {x_norm.max().item()}, min: {x_norm.min().item()}, mean: {x_norm.mean().item()}"
            )
            self.__log_image(
                x=x_norm, folder="validation/reconstructed_images_normalized"
            )

    def __log_image(self, x, folder, mask=None):
        """log first image from batch"""
        img = x[0]
        img = img.cpu()
        if mask is not None:
            # make rgb from grayscale
            img = img.repeat(3, 1, 1)
            mask = mask[0]
            mask = mask.cpu()
            if mask.shape[0] == 1:
                # make rgb from grayscale
                mask = mask.repeat(3, 1, 1)
            # invert zeros to ones
            red_mask = torch.ones_like(mask) - mask
            # zerofy green and blue channels
            red_mask[1, :, :] = 0.0
            red_mask[2, :, :] = 0.0
            img = img + red_mask

        # if a grayscale image (H, W) do nothing,
        # else transfer (C, H, W) -> (H, W, C)
        if len(img.shape) > 2:
            last_channel_image = torch.moveaxis(img, 0, -1)
        else:
            last_channel_image = img
        self.logger.experiment[folder].append(File.as_image(last_channel_image))


def train(checkpoint_dir, loss_only_masked=False):
    torch.cuda.empty_cache()
    #  load data
    proj_dir = Path(__file__).resolve().parent.parent
    img_dir = proj_dir.joinpath("data/train_npy_subset_005")
    info_table = proj_dir.joinpath("data/split_subset_005.pkl")
    datamodule = CTBrainDataModule(
        img_dir=img_dir, info_table=info_table, batch_size=24
    )

    logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhOTEzNzU0ZS1kYzRlLTRlNjQtYjJjYi0yNjY4M2JjYmE3MDAifQ==",
        project="brain",
        name="sgd_run",
    )


    model = Model(checkpoint_dir=checkpoint_dir, loss_only_masked=loss_only_masked)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", filename="model_{epoch}", every_n_epochs=1
    )
    trainer = Trainer(
        max_epochs=150,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        enable_checkpointing=True,
    )
    trainer.fit(model=model, datamodule=datamodule)


def train_on_aimi(checkpoint_dir, loss_only_masked=False):
    torch.cuda.empty_cache()
    #  load data
    proj_dir = Path(__file__).resolve().parent.parent
    img_dir = proj_dir.joinpath("data/train_npy_subset_005")
    info_table = proj_dir.joinpath("data/split_subset_005.pkl")
    data_dir = ("/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon",)
    healthy_patient_ids = (
        "/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon/healthy_patient_ids.npy",
    )

    datamodule = AIMIBrainDataModule(
        data_dir="/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon",
        healthy_patient_ids="/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon/healthy_patient_ids.npy",
        info_table=info_table,
        img_dir=img_dir,
        batch_size=24,
    )

    logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhOTEzNzU0ZS1kYzRlLTRlNjQtYjJjYi0yNjY4M2JjYmE3MDAifQ==",
        project="brain",
        name="sgd_run",
    )

    model = Model(checkpoint_dir=checkpoint_dir, loss_only_masked=loss_only_masked)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        max_epochs=150,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[lr_monitor],
        enable_checkpointing=True,
    )
    trainer.fit(model=model, datamodule=datamodule)


def train_transformer_on_aimi(checkpoint_dir):
    torch.cuda.empty_cache()
    #  load data
    proj_dir = Path(__file__).resolve().parent.parent
    img_dir = proj_dir.joinpath("data/train_npy_subset_005")
    info_table = proj_dir.joinpath("data/split_subset_005.pkl")
    data_dir = ("/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon",)
    healthy_patient_ids = (
        "/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon/healthy_patient_ids.npy",
    )

    datamodule = AIMIBrainDataModule(
        data_dir="/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon",
        healthy_patient_ids="/home/mark/Data/tmp/br/ctsinogram/head_ct_dataset_anon/healthy_patient_ids.npy",
        info_table=info_table,
        img_dir=img_dir,
        batch_size=40,
    )

    logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhOTEzNzU0ZS1kYzRlLTRlNjQtYjJjYi0yNjY4M2JjYmE3MDAifQ==",
        project="brain",
        name="transformer",
    )

    model = TransModel(checkpoint_dir=checkpoint_dir)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        max_epochs=150,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[lr_monitor],
        enable_checkpointing=True,
    )
    trainer.fit(model=model, datamodule=datamodule)


def continue_training(checkpoint_path):
    proj_dir = Path(__file__).resolve().parent.parent
    img_dir = proj_dir.joinpath("data/train_npy_subset_005")
    info_table = proj_dir.joinpath("data/split_subset_005.pkl")
    datamodule = CTBrainDataModule(
        img_dir=img_dir, info_table=info_table, batch_size=24
    )

    logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhOTEzNzU0ZS1kYzRlLTRlNjQtYjJjYi0yNjY4M2JjYmE3MDAifQ==",
        project="brain",
        name="sgd_run",
        with_id="BRAIN-231",
    )

    model = Model(checkpoint_dir="checkpoints")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", filename="model_{epoch}", every_n_epochs=1
    )
    trainer = Trainer(
        max_epochs=250,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        enable_checkpointing=True,
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)


def test_on_anomalies(checkpoint_path):

    proj_dir = Path(__file__).resolve().parent.parent
    img_dir = proj_dir.joinpath("data/train_npy_subset_005")
    info_table = proj_dir.joinpath("data/split_subset_005.pkl")
    datamodule = CTBrainDataModule(img_dir=img_dir, info_table=info_table, batch_size=1)
    datamodule.setup("test")

    model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path)

    trainer = Trainer()
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire()
