import torch
import torch.nn.functional as F

import os

from typing import List, Optional
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import numpy as np
import streamlit as st

from pathlib import Path
import sys

_folder_current = Path(__file__).parent
_folder = _folder_current.parent
sys.path.append(str(_folder))

from denoising.train import Model
from denoising.data_utils import preprocess_dicom


def get_score(x_true, x_rec, threshold=0.0):
    x_true = x_true.cpu()
    x_rec = x_rec.cpu()

    diff = torch.abs(x_true - x_rec)
    diff = torch.where(diff > threshold, diff, torch.zeros_like(diff))
    black = (x_true == 0.0).int().sum(dim=[-1, -2, -3])

    # exponential smoothing
    bg_width = 3
    exp_ratio = 1
    bg_conv_weight = torch.ones(1, 1, bg_width * 2 + 1, bg_width * 2 + 1)
    bg_weight = F.conv2d(
        input=diff, weight=bg_conv_weight, bias=None, stride=1, padding=bg_width
    )
    diff = diff * exp_ratio
    diff = torch.exp(bg_weight) * diff

    pixel_num = diff.shape[-1] * diff.shape[-2] * diff.shape[-3]
    total_num = torch.Tensor([pixel_num] * diff.shape[0])

    denominator = total_num - black
    score = diff.sum(dim=[-1, -2, -3]) / denominator

    return score


def main(
    checkpoint_path="/home/mark/Desktop/fun/br/AIXI_X/checkpoints_reworked_style_loss/epoch_40.ckpt",
):
    # if "model" not in st.session_state:
    model = Model.load_from_checkpoint(checkpoint_path)
    model.vgg19 = None
    model.descriminator = None
    model.cuda()
    st.session_state["model"] = model

    uploaded_file = st.file_uploader(
        "Choose DICOM files", type="dcm", accept_multiple_files=True
    )

    if uploaded_file:
        os.write(1, b"File was uploaded.\n")

        images = []
        denoised_images = []
        concat_tensors = []
        diffs = []
        for f in uploaded_file:
            image_numpy, image_tensor, edges_tensor, concat_tensor = preprocess_dicom(f)
            images.append(image_tensor.squeeze(0))
            concat_tensors.append(concat_tensor.squeeze(0))

        images = torch.stack(images).cpu()
        os.write(1, f"images: {images.shape}".encode())
        concat_tensors = torch.stack(concat_tensors, dim=0)
        concat_tensors = concat_tensors.cuda()
        sh = f"{images.shape}"
        denoised_images = model.generator(concat_tensors).detach().cpu()
        x_diff = np.abs(denoised_images - images)
        os.write(1, f"diff: {x_diff.shape}".encode())
        os.write(1, f"denoised: {denoised_images.shape}".encode())

        scores = get_score(x_true=images, x_rec=denoised_images)

        # show_barplots(scores)

        if "z" not in st.session_state:
            st.session_state.z = 0

        st.slider(
            "My Slider",
            min_value=0,
            max_value=len(images) - 1,
            step=1,
            value=0,
            key="z",
            on_change=show_collage,
            args=[
                images.numpy(),
                denoised_images.numpy(),
                x_diff.numpy(),
                scores.numpy(),
                st.session_state.z,
            ],
        )


def show_barplots(bars):

    # Example list of values
    labels = [i for i in range(len(bars))]
    bars = np.where(bars < 100, bars, 100)
    # Creating the bar plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.bar(labels, bars)

    # Adding titles and labels
    plt.xlabel("Slice number")
    plt.ylabel("Score")
    plt.title("Bar Plot Example")
    st.pyplot(fig)


@st.cache_data
def show_collage(
    images,
    denoised_images,
    diff,
    bars,
    z: int = 0,
    names: List[str] = ["CT scan", "Denoised scan", "Difference"],
):

    img = images[z].squeeze()
    denoised_img = denoised_images[z].squeeze()
    diff_img = diff[z].squeeze()

    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax[0].imshow(img, cmap="gist_gray")
    # ax[1].imshow(denoised_img, cmap="gist_gray")
    # ax[2].imshow(diff_img, cmap="gist_gray")

    # plt.axis("off")
    # st.pyplot(fig)

    # Create a figure with a specific size
    fig = plt.figure(figsize=(10, 6))

    # Create a GridSpec object
    gs = gridspec.GridSpec(2, 3)

    # First row - three subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Second row - one subplot spanning all columns
    ax4 = fig.add_subplot(gs[1, :])
    ax1.imshow(img, cmap="gist_gray")
    ax2.imshow(denoised_img, cmap="gist_gray")
    ax3.imshow(diff_img, cmap="gist_gray")

    ax1.set_title("CT Scan")
    ax2.set_title("Denoised Scan")
    ax3.set_title("Difference")

    labels = [str(i) for i in range(len(bars))]
    bars = np.where(bars < 10_000, bars, 10_000)
    ax4.bar(labels, bars)

    ax4.set_xlabel("Slice number")
    ax4.set_ylabel("Score")

    rect = patches.Rectangle(
        (-0.5 + z, 0), 1, 1000, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax4.add_patch(rect)

    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
    # image_numpy, image_tensor, edges_tensor, concat_tensor  = preprocess_dicom()
