import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pydicom
from pydicom.errors import InvalidDicomError
import streamlit as st
from typing import List, Optional, Union, Tuple
from pathlib import Path
from tempfile import TemporaryDirectory
import pyvista as pv
from stpyvista import stpyvista
from pyvista.plotting.utilities import xvfb
from skimage import transform, measure
import pandas as pd
import json


_folder_current = Path(__file__).parent
_folder = _folder_current.parent
sys.path.append(str(_folder))
from classification.data.misc import (
    window_image,
    class_names,
    load_study,
    change_image_size,
)
from classification.pipeline import ClassificationPipeline
import classification.llm as llm 
from classification.mask import add_segmentation_mask_to_image
from classification.data.filter import FilterBrainPresent

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def load_dicom_files(uploaded_files: List[str]) -> List[pydicom.FileDataset]:
    slices = []
    for uploaded_file in uploaded_files:
        try:
            slice_ds = pydicom.dcmread(uploaded_file)
            slices.append(slice_ds)
        except InvalidDicomError:
            continue
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def convert_scores_to_colors(
    scores: np.ndarray,
    colors: np.ndarray = np.array(mpl.colormaps["tab10"].colors),
) -> np.ndarray:
    image = np.zeros((scores.shape[1], scores.shape[0], 3), np.float32)
    for i in range(scores.shape[1]):
        image[i] = 1 - (1 - np.expand_dims(colors[i], 0)) * np.expand_dims(
            scores[:, i], 1
        )
    return image


def show_slice_scores(
    scores: np.ndarray,
    class_names: Optional[List[str]] = class_names,
    select_index: Optional[int] = None,
    col=None,
):
    fig, ax = plt.subplots()
    image = convert_scores_to_colors(scores)
    ax.imshow(image)
    if select_index is not None:
        rect = mpl.patches.Rectangle(
            (select_index - 0.5, -0.5),
            1.0,
            scores.shape[1],
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_yticks(np.arange(scores.shape[1]))
    if class_names is not None:
        ax.set_yticklabels(class_names)
    if col is None:
        st.pyplot(fig)
    else:
        col.pyplot(fig)
    return


def show_slice(
    slice: pydicom.FileDataset,
    segm_mask: Optional[np.ndarray] = None,
    denoised_img: Optional[np.ndarray] = None,
    window_center: float = 40.0,
    window_width: float = 80.0,
    rescale_slope: float = 1.0,
    rescale_intercept: float = -1024.0,
    segm_mask_alpha: float = 0.3,
    col=None,
    show_denoised=False,
):
    fig, ax = plt.subplots(figsize=(3, 3))
    if not show_denoised:
        image = window_image(
            slice.pixel_array,
            window_center=window_center,
            window_width=window_width,
            slope=rescale_slope,
            intercept=rescale_intercept,
        )

    else:
        image = denoised_img
        image = image.squeeze()

    # print("img: ", image.shape)
    if segm_mask is not None:
        image = add_segmentation_mask_to_image(
            image, segm_mask, mask_alpha=segm_mask_alpha
        )
        # print("img after mask: ", image.shape)
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    if col is None:
        col = st
    col.pyplot(fig, use_container_width=False)
    return


def show_3d(
        slices: List[Union[np.ndarray, pydicom.FileDataset]], 
        masks: List[np.ndarray],
        rescale : Optional[Tuple[float, float, float]] = (5, 0.25, 0.25),
        as_mesh : bool = True,
        mesh_head_level : float = 0.5,
        mesh_mask_level : float = 0.5,
        window_center : Optional[float] = None,
        window_width : Optional[float] = None,
        rescale_slope : float = 1.,
        rescale_intercept : float = -1024.,
        head_opacity : float = 0.2,
        mask_opacity : float = 0.5,
        show : bool = False,
        flip_orientation : bool = True,
        ):
    xvfb.start_xvfb()

    if window_center is None:
        if as_mesh:
            window_center = 400
        else:
            window_center = 40
    if window_width is None:
        if as_mesh:
            window_width = 1800
        else:
            window_width = 80

    image = np.array([s.pixel_array if isinstance(s, pydicom.FileDataset) else s for s in slices], np.float32)
    image = window_image(image, window_center=window_center, window_width=window_width,
        intercept=rescale_intercept, slope=rescale_slope)
    
    if flip_orientation:
        image = image[::-1]
        masks = masks[::-1]
    
    try:
        pixel_spacing = slices[0].PixelSpacing
    except:
        pixel_spacing = slices[0].NominalScannedPixelSpacing
    z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    resolution = (z_spacing, pixel_spacing[0], pixel_spacing[1])

    if rescale is None:
        masks = np.array([change_image_size(m[0], image.shape[-2:]) for m in masks])
    else:
        image = transform.rescale(image, rescale)
        masks = transform.resize(masks[:, 0], image.shape)
        resolution = [r / n for r, n in zip(resolution, rescale)]

    # if flip_orientation:
    #     masks = np.flip(masks, axis=-1)
    #     image = np.flip(image, axis=-1)

    pl = pv.Plotter()
    if as_mesh:
        verts, faces, _, _ = measure.marching_cubes(image, level=mesh_head_level, spacing=resolution)
        mesh = pv.PolyData.from_regular_faces(verts, faces)
        pl.add_mesh(mesh, opacity=head_opacity)
        if np.any(masks > mesh_mask_level) and np.any(masks < mesh_mask_level):
            verts_mask, faces_mask, _, _ = measure.marching_cubes(masks, level=mesh_mask_level, spacing=resolution)
            mesh_mask = pv.PolyData.from_regular_faces(verts_mask, faces_mask)
            pl.add_mesh(mesh_mask, opacity=mask_opacity, color="red")
    else:
        pl.add_volume(image, resolution=resolution, opacity="linear", cmap="bone", show_scalar_bar=False)
        pl.add_volume(masks, resolution=resolution, opacity="linear", cmap=["red"])
    pl.camera_position = "zy"
    pl.camera.elevation = -45
    pl.camera.zoom(10)
    if show:
        stpyvista(pl, key="pv")
    return pl


def calculate_mask_volume(
        slices: List[pydicom.FileDataset], 
        masks: List[np.ndarray], 
        threshold : float = 0.5,
        ) -> float:
    volumes = []
    positions = []
    for index, (slice, mask) in enumerate(zip(slices, masks)):
        slice_size = slice.pixel_array.shape
        mask_size = mask.shape[1:]
        multiplier = slice_size[0] / mask_size[0] * slice_size[1] / mask_size[1]
        count = np.count_nonzero(mask >= threshold)
        try:
            spacing = slice.PixelSpacing
        except:
            spacing = slice.NominalScannedPixelSpacing
        vol = count * spacing[0] * spacing[1] * multiplier
        volumes.append(vol)
        positions.append(slice.ImagePositionPatient[2])
    volumes = np.array(volumes)
    positions = np.diff(positions)
    vol = np.sum(volumes[1:] * np.abs(positions))
    return vol / 1000


@st.fragment
def plot_with_slider(
        slices: List[pydicom.FileDataset],
        scores: np.ndarray,
        segm_masks : Optional[np.ndarray] = None,
        denoised_images : Optional[np.ndarray] = None,
        ):
    top_score_index = scores[:, -1].argmax()
    slice_idx = st.slider(
        "Select Slice", min_value=0, max_value=len(slices) - 1, 
        value=top_score_index, step=1,
    )
        
    show_slice_scores(scores, select_index=slice_idx)

    slice = slices[slice_idx]
    rescale_slope = slice.RescaleSlope if "RescaleSlope" in slice else 1
    rescale_intercept = (
        slice.RescaleIntercept if "RescaleIntercept" in slice else -1024.0
    )
    
    expander = st.expander("Windows")
    window_width = expander.slider(
        "Window Width", min_value=1, max_value=400, value=80, step=1
    )
    window_center = expander.slider(
        "Window Center", min_value=-100, max_value=300, value=40, step=1
    )
    
    if segm_masks is not None:
        to_show_segm = st.toggle("Show segmentation mask", True)
    else:
        to_show_segm = False
    if denoised_images is not None:
        to_show_denoised = st.toggle("Show denoised image", False)
    else:
        to_show_denoised = False
    
    show_slice(
        slice=slice,
        segm_mask=(
            segm_masks[slice_idx]
            if (segm_masks is not None and to_show_segm)
            else None
        ),
        denoised_img=(
            denoised_images[slice_idx]
            if (denoised_images is not None and to_show_denoised)
            else None
        ),
        # scores=np.zeros(6),
        window_center=window_center,
        window_width=window_width,
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        # col=col2,
        show_denoised=to_show_denoised,
    )
    
    st.subheader(f"Slice scores:")
    score_strings = [f"{s:.2f}" for s in scores[slice_idx]]
    st.write(pd.DataFrame({"Type": class_names, "Score": score_strings}))
    return


def write_study_scores(scores: np.ndarray, threshold : Optional[float] = None):
    if "volume" in st.session_state:
        if st.session_state.volume > 0:
            st.subheader(f"Hemorrhage volume: {st.session_state.volume:.1f}mL")
        elif threshold is not None and np.all(scores) < threshold:
            st.subheader("No hemorrhage detected")
    if scores is not None:
        if threshold is None or np.any(scores >= threshold):
            st.subheader(f"Study scores:")
            score_strings = [f"{s:.2f}" for s in scores]
            st.write(pd.DataFrame({"Type": class_names, "Score": score_strings}))
    return


# @st.fragment
# def plot_with_slider(
#         slices: List[pydicom.FileDataset],
#         scores: np.ndarray,
#         segm_masks : Optional[np.ndarray] = None,
#         denoised_images : Optional[np.ndarray] = None,
#         patient_scores : Optional[np.ndarray] = None,
#         ):
#     col1, col2 = st.columns(2)
    
#     with col1:
#         slice_idx = st.slider(
#             "Select Slice", min_value=0, max_value=len(slices) - 1, value=0, step=1
#         )

#         if segm_masks is not None:
#             to_show_segm = st.toggle("Show segmentation mask", True)
#         if denoised_images is not None:
#             to_show_denoised = st.toggle("Show denoised image", False)

#         show_slice_scores(scores, select_index=slice_idx)

#         slice = slices[slice_idx]
#         rescale_slope = slice.RescaleSlope if "RescaleSlope" in slice else 1
#         rescale_intercept = (
#             slice.RescaleIntercept if "RescaleIntercept" in slice else -1024.0
#         )

#         # Windowing parameters input
#         expander = st.expander("Windows")
#         window_width = expander.slider(
#             "Window Width", min_value=1, max_value=400, value=80, step=1
#         )
#         window_center = expander.slider(
#             "Window Center", min_value=-100, max_value=300, value=40, step=1
#         )

#         if "volume" in st.session_state:
#             st.subheader(f"Hemorrhage volume: {st.session_state.volume:.1f}mL")
#         if patient_scores is not None:
#             st.subheader(f"Study scores:")
#             score_strings = [f"{s:.2f}" for s in patient_scores]
#             st.write(pd.DataFrame({"Type": class_names, "Score": score_strings}))

#     with col2:
#         show_slice(
#             slice=slice,
#             segm_mask=(
#                 segm_masks[slice_idx]
#                 if (segm_masks is not None and to_show_segm)
#                 else None
#             ),
#             denoised_img=(
#                 denoised_images[slice_idx]
#                 if (denoised_images is not None and to_show_denoised)
#                 else None
#             ),
#             # scores=np.zeros(6),
#             window_center=window_center,
#             window_width=window_width,
#             rescale_slope=rescale_slope,
#             rescale_intercept=rescale_intercept,
#             # col=col2,
#             show_denoised=to_show_denoised,
#         )

#         st.subheader(f"Slice scores:")
#         score_strings = [f"{s:.2f}" for s in scores[slice_idx]]
#         st.write(pd.DataFrame({"Type": class_names, "Score": score_strings}))
#     return

def main():
    title = "AISI X - AI Platform for Brain Diseases"
    st.set_page_config(page_title=title, layout="wide")
    # st.markdown("""
    #     <style>
    #         .header {
    #             font-size: 24px;
    #             font-weight: bold;
    #             color: gray;
    #             position: fixed;
    #             top: 0;
    #             left: 0;
    #             margin: 10px;
    #             z-index: 1000;
    #         }
    #     </style>
    #     <div class="header">AISI X - AI Platform for Brain Diseases</div>
    # """, unsafe_allow_html=True)

    st.markdown("""
    <style>
               /* Remove blank space at top and bottom */ 
               .block-container {
                   padding-top: 1rem;
                   padding-bottom: 0rem;
                }

        </style>
    """, unsafe_allow_html=True)
    st.title(title)

    if "model" not in st.session_state:
        st.session_state["model"] = ClassificationPipeline(
            model_name="2d_v15_rn50_ml_fix",
            seq_model_name="seq_2",
            # model_name="2d_v18_rn101_ml",
            # seq_model_name="seq_4",
            segm_model_name="segm_8_split1_2d_n1",
            checkpoint_name="best",
            device="cpu",
        )
        
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm.load_model()

    model = st.session_state.model

    with st.form("File upload", clear_on_submit=True):
        st.session_state["_uploaded_files"] = st.file_uploader(
            "Choose DICOM files",
            type="dcm",
            accept_multiple_files=True,
        )
        st.session_state["_submitted"] = st.form_submit_button("submit")

    # if st.session_state.model.segm_model is None:
    #     col1 = st.columns(1)
    # else:
    #     col1, col2 = st.columns(spec=[2, 1])
    col1, col2, col3 = st.columns(spec=[1, 1, 1])

    # if uploaded_files:
    if st.session_state.get("_submitted", False):
        uploaded_files = st.session_state.get("_uploaded_files")
        if (
                "uploaded_files" not in st.session_state
                or len(uploaded_files) != len(st.session_state.uploaded_files)
                or not all(
                    (f1 == f2)
                    for f1, f2 in zip(uploaded_files, st.session_state.uploaded_files)
                )
            ):
            update_state = True
            st.session_state.uploaded_files = uploaded_files
        else:
            update_state = False
        
        if "slices" not in st.session_state or update_state:
            if True:
                with TemporaryDirectory() as tmpdir:
                    folder = Path(tmpdir)
                    filenames = []
                    for file in uploaded_files:
                        fname = folder.joinpath(file.name)
                        with open(fname, "wb") as f:
                            f.write(file.getbuffer())
                            filenames.append(fname)
                    # st.session_state.slices = load_study(names=filenames, as_array=False, as_pydicom=True)
                    st.session_state.slices = load_study(
                        folder, as_array=False, as_pydicom=True
                    )
            else:
                st.session_state.slices = load_dicom_files(uploaded_files)
        slices = st.session_state.slices
        # slices = load_dicom_files(uploaded_files)

        if "scores" not in st.session_state or update_state:
            pred_data = model.predict(slices, get_full_data=True)
            scores = pred_data["preds"]
            scores[:, -1] = 1 - scores[:, -1]
            st.session_state.scores = scores
            patient_scores = pred_data["patient_preds"]
            patient_scores[:, -1] = 1 - patient_scores[:, -1]
            st.session_state.patient_scores = patient_scores[0]
            if "segm_masks" in pred_data:
                segm_masks = pred_data["segm_masks"]
                st.session_state.segm_masks = segm_masks
                for i, s in enumerate(scores):
                    if np.all(s < 0.01):
                        segm_masks[i][:] = 0
                st.session_state.volume = calculate_mask_volume(slices, segm_masks)
            else:
                segm_masks = None
            if "denoised_images" in pred_data:
                denoised_images = pred_data["denoised_images"]
                st.session_state.denoised_images = denoised_images

            slice = slices[0]
            rescale_slope = slice.RescaleSlope if "RescaleSlope" in slice else 1
            rescale_intercept = (
                slice.RescaleIntercept if "RescaleIntercept" in slice else -1024.0
            )
            
            if segm_masks is not None:
                plotter = show_3d(slices, segm_masks, as_mesh=True,
                    rescale_slope=rescale_slope, 
                    rescale_intercept=rescale_intercept,
                    show=False,
                )
                st.session_state.plotter = plotter
                
        scores = st.session_state.scores
        segm_masks = st.session_state.get("segm_masks", None)
        denoised_images = st.session_state.get("denoised_images", None)
        patient_scores = st.session_state.get("patient_scores", None)
        
        if "llm_model" in st.session_state and ("llm_output" not in st.session_state or update_state):
            sel_inds = [i for i, s in enumerate(slices) 
                if FilterBrainPresent().filter(s.pixel_array, intercept=rescale_intercept, slope=rescale_slope) is not None]
            sel_inds = np.asarray(sel_inds)
            selected_slice_inds = llm.select_representative_slices(scores[sel_inds], score_threshold=None)
            selected_slice_inds = sel_inds[selected_slice_inds]
            sel_images = [slices[i] for i in selected_slice_inds]
            sel_scores = scores[selected_slice_inds]
            output = llm.run_model_multiple(*st.session_state["llm_model"], 
                sel_images, scores=scores, volume=st.session_state.volume)
            st.session_state["llm_output"] = output

        if len(slices) > 0:

            # with col1:
            #     plot_with_slider(
            #         slices=slices,
            #         scores=scores,
            #         segm_masks=segm_masks,
            #         denoised_images=denoised_images,
            #         patient_scores=patient_scores,
            #     )

            # if "plotter" in st.session_state:
            #     with col2:
            #         stpyvista(st.session_state.plotter)
            #         if "llm_output" in st.session_state:
            #             s = st.session_state.llm_output
            #             s = pd.DataFrame(json.loads(s[7:-3]))
            #             st.table(s)
            
            with col1:
                plot_with_slider(
                    slices=slices,
                    scores=scores,
                    segm_masks=segm_masks,
                    denoised_images=denoised_images,
                )
            
            with col2:
                if "plotter" in st.session_state:
                    with col2:
                        stpyvista(st.session_state.plotter, panel_kwargs={"aspect_ratio": 1, "height": 720})
                write_study_scores(patient_scores)
                
            with col3:
                if "llm_output" in st.session_state:
                    s = st.session_state.llm_output
                    # s = pd.DataFrame(json.loads(s[7:-3]))
                    # st.table(s)
                    st.write(s)
                    
        else:
            st.write("No valid DICOM files were uploaded.")

    return


if __name__ == "__main__":
    main()