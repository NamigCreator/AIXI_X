from lightning import Trainer, LightningModule

# from pytorch_lightning import Trainer, LightningModule

from lightning.pytorch.loggers import NeptuneLogger

# from pytorch_lightning.loggers import NeptuneLogger

from neptune.types import File

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
import random

import sys

sys.path.append("code/")
from models import Generator, Descriminator, VGG19_intermediate_layers_only
from data_utils import CTBrainDataModule, anomalies


class Model(LightningModule):
    def __init__(
        self,
        lambda_style=10,
        lambda_perceptual=0.1,
        lambda_adversarial=0.5,
        patch_size=16,
        img_size=256,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.generator = Generator()
        self.descriminator = Descriminator()
        self.vgg19 = VGG19_intermediate_layers_only()

        self.lambda_sty = lambda_style
        self.lambda_per = lambda_perceptual
        self.lambda_adv = lambda_adversarial

        self.img_size = img_size
        self.patch_size = patch_size

        # to store some examples in validation
        self.stash = []

    def training_step(self, batch, batch_idx):

        self.g_opt.zero_grad()

        x_true, edges, target, name = batch

        mask_size = random.randint(16, 48)
        mask = self.get_random_mask(x_true, mask_size)
        x_masked = x_true * mask

        x_true_w_edges = self.concat_image_and_edges(x_masked, edges) #TODO apply mask after concat!?
        x_reconstructed = self.generator(x_true_w_edges)

        losses = self.calculate_losses(x_true, x_reconstructed)

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

        if batch_idx % 10 == 0:
            # to prevent discriminator be too strong
            self.manual_backward(d_loss)
            self.d_opt.step()

        losses.update({"g_loss": g_loss, "d_loss": d_loss})
        self.log_dict(
            losses,
            prog_bar=True,
        )

        if batch_idx % 100 == 0:
            self.__log_image(x_true, "training/real_images")
            # first edge
            self.__log_image(x_true_w_edges[:, -1, :, :], "training/edges_1")
            # second edge
            self.__log_image(x_true_w_edges[:, -2, :, :], "training/edges_2")
            # third edge
            self.__log_image(x_true_w_edges[:, -3, :, :], "training/edges_3")
            self.__log_image(x_masked, "training/masked_images", mask=mask)
            self.__log_image(x_reconstructed, "training/reconstructed_images")

    def __log_image(self, x, folder, mask=None, choose_random=False):
        """log first image from batch"""
        if choose_random:
            img_idx = random.randint(0, x.shape[0] - 1)
        else:
            img_idx = 0
        img = x[img_idx]
        img = img.cpu()
        if mask is not None:
            mask = mask[img_idx]
            mask = mask.cpu()
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

    def concat_image_and_edges(self, image, edges: dict):
        """(N, 3, H, W) + E * (N, 1, H, W) -> (N, 3+E, H, W)"""
        thresholds = sorted(list(edges.keys()))
        to_concat = [image]
        for thr in thresholds:
            edge = edges[thr]
            if edge.shape[0] != image.shape[0]:
                repeat_factor = image.shape[0] // edge.shape[0]
                edge = edge.repeat(repeat_factor, 1, 1, 1)
            to_concat.append(edge)
        image_and_edges = torch.concat(to_concat, dim=1)
        return image_and_edges

    def configure_optimizers(self):
        self.g_opt = SGD(
            params=self.generator.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.d_opt = SGD(
            params=self.generator.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        return self.g_opt, self.d_opt

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def style_loss(self, x_true, x_rec):
        return F.l1_loss(self.gram_matrix(x_true), self.gram_matrix(x_rec))

    def perceptual_loss(self, x_true, x_rec):
        intermediate_layers_true = self.vgg19(x_true)
        intermediate_layers_rec = self.vgg19(x_rec)
        loss = torch.tensor([0.0], requires_grad=True).to("cuda")
        layer_names = self.vgg19.return_layers.values()
        for name in layer_names:
            loss = loss + F.l1_loss(
                intermediate_layers_rec[name], intermediate_layers_true[name]
            )

        return loss / len(layer_names)

    def adversarial_loss(self, x, targets):
        return F.binary_cross_entropy_with_logits(x, targets)

    def reconstruction_loss(self, x, y):
        return F.l1_loss(x, y)

    def random_mask_image(self, x, K):
        mask = torch.ones_like(x)

        rows = cols = x.shape[2]
        top_left_x = random.randint(0, rows - K + 1)
        top_left_y = random.randint(0, cols - K + 1)

        mask[:, :, top_left_x : top_left_x + K, top_left_y : top_left_y + K] = 0

        return x * mask

    def get_random_mask(self, x, mask_size=16):

        mask = torch.ones_like(x)

        rows = cols = x.shape[2]

        for i in range(mask.shape[0]):
            # top_left_x = random.randint(0, rows - mask_size + 1)
            # top_left_y = random.randint(0, cols - mask_size + 1)
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

    def validation_step(self, batch, batch_idx):
        ### TODO: check that reconstruction loss for unhealthy
        ### is bigger
        x_true, edges, target, name = batch
        N, C, H, W = x_true.shape
        # mask has shape ((H // K) * (W // K), C, H, W)
        mask = self.get_mask_grid(x_true, K=32)
        x_masked = x_true.repeat(mask.shape[0], 1, 1, 1)
        mask = mask.repeat(N, 1, 1, 1)
        mask = mask.to(x_true.device)
        x_masked = x_masked * mask
        x_masked_w_edges = self.concat_image_and_edges(x_masked, edges) #TODO apply mask after concat?
        x_reconstructed = self.generator(x_masked_w_edges)
        # zerofy everything except masked region
        x_reconstructed = (~mask) * x_reconstructed

        x_reconstructed = x_reconstructed.view(N, -1, C, H, W).mean(1)
        losses = self.calculate_losses(x_true, x_reconstructed)

        # works only for batch size 1
        if name[0] in anomalies:
            self.stash.append((x_true, x_reconstructed))

        if batch_idx % 100 == 0:
            self.__log_image(x_reconstructed, folder="validation/x_reconstructed")
            self.__log_image(x_true, folder="validation/x_true")
            self.__log_image(x_masked, "validation/masked_images", mask=mask)

        rec_loss = losses["reconstruction_loss"]
        losses[f"reconstruction_loss_class:{int(target.item())}"] = rec_loss.item()
        self.log_dict(losses, prog_bar=True, on_step=True)

    def on_validation_epoch_end(self):
        print("self stsh: ", len(self.stash))
        for x_true, x_reconstructed in self.stash:
            self.__log_image(x_reconstructed, folder="anomalies/x_reconstructed")
            self.__log_image(x_true, folder="anomalies/x_true")
        self.stash = []

    def calculate_losses(self, x_true, x_reconstructed):

        valid = torch.Tensor(x_true.size(0), 1).fill_(1.0).to("cuda")
        fake = torch.Tensor(x_true.size(0), 1).fill_(0.0).to("cuda")

        rec_loss = self.reconstruction_loss(x_reconstructed, x_true)
        perc_loss = self.perceptual_loss(x_reconstructed, x_true)
        style_loss = self.style_loss(x_reconstructed, x_true)
        adv_loss = self.adversarial_loss(self.descriminator(x_reconstructed), fake)

        real_loss = self.adversarial_loss(self.descriminator(x_true), valid)
        fake_loss = self.adversarial_loss(
            self.descriminator(x_reconstructed.detach()), fake
        )

        return {
            "reconstruction_loss": rec_loss,
            "adversarial_loss": adv_loss,
            "perceptual_loss": perc_loss,
            "style_loss": style_loss,
            "real_loss": real_loss,
            "fake_loss": fake_loss,
        }


from pathlib import Path

if __name__ == "__main__":
    torch.cuda.empty_cache()
    #  load data
    proj_dir = Path(__file__).resolve().parent.parent
    img_dir = proj_dir.joinpath("data/train_npy_subset_005")
    info_table = proj_dir.joinpath("data/split_subset_005.pkl")
    datamodule = CTBrainDataModule(
        img_dir=img_dir, info_table=info_table, batch_size=16
    )

    logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhOTEzNzU0ZS1kYzRlLTRlNjQtYjJjYi0yNjY4M2JjYmE3MDAifQ==",
        project="brain",
        name="sgd_run",
    )

    model = Model()
    trainer = Trainer(
        max_epochs=10,
        # limit_val_batches=100,
        # limit_train_batches=1,
        log_every_n_steps=100,
        logger=logger,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.save_checkpoint("checkpoints/model.ckpt")
