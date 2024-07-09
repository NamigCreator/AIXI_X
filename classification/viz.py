import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pydicom
from pydicom.errors import InvalidDicomError
import streamlit as st
from typing import List, Optional
from pathlib import Path

_folder_current = Path(__file__).parent
_folder = _folder_current.parent
sys.path.append(str(_folder))
from classification.data.misc import window_image, class_names
from classification.pipeline import ClassificationPipeline


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
        colors : np.ndarray = np.array(mpl.colormaps["tab10"].colors),
        ) -> np.ndarray:
    image = np.zeros((scores.shape[1], scores.shape[0], 3), np.float32)
    for i in range(scores.shape[1]):
        image[i] = 1 - (1 - np.expand_dims(colors[i], 0)) * np.expand_dims(scores[:, i], 1)
    return image

def show_slice_scores(
        scores: np.ndarray, 
        class_names : Optional[List[str]] = class_names,
        select_index : Optional[int] = None,
        ):
    fig, ax = plt.subplots()
    image = convert_scores_to_colors(scores)
    ax.imshow(image)
    if select_index is not None:
        rect = mpl.patches.Rectangle(
            (select_index-0.5, -0.5), 1., scores.shape[1], 
            linewidth=1, edgecolor="black", facecolor="none")
        ax.add_patch(rect)
    ax.set_yticks(np.arange(scores.shape[1]))
    if class_names is not None:
        ax.set_yticklabels(class_names)    
    st.pyplot(fig)
    return


def show_slice(
        slice: pydicom.FileDataset, 
        scores : Optional[np.ndarray] = None,
        class_names : Optional[List[str]] = class_names,
        window_center: float = 40., 
        window_width: float = 80., 
        rescale_slope : float = 1., 
        rescale_intercept : float = -1024.,
        ):
    fig, ax = plt.subplots()
    windowed_data = window_image(slice.pixel_array, 
        window_center=window_center, 
        window_width=window_width,
        slope=rescale_slope, 
        intercept=rescale_intercept,
    )
    ax.imshow(windowed_data, cmap='gray')
    plt.axis("off")
    st.title(f"Z-position: {slice.ImagePositionPatient[2]:.1f}")
    st.header(f"Scores:")
    for c, s in zip(class_names, scores):
        st.text(f"\t{c:16s} : {s:.3f}")
    st.pyplot(fig)
    return
    

def main():
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

    st.title('AISI X - AI Platform for Brain Diseases')
    
    if "model" not in st.session_state:
        st.session_state["model"] = ClassificationPipeline(
            model_name="2d_v15_rn50_ml_fix",
            seq_model_name="seq_2",
            checkpoint_name="best",
            device="cpu",
        )
    model = st.session_state.model
    
    # if "uploaded_files" not in st.session_state:
    #     st.session_state.uploaded_files = st.file_uploader("Choose DICOM files", type='dcm', accept_multiple_files=True)
    # uploaded_files = st.session_state.uploaded_files
    uploaded_files = st.file_uploader("Choose DICOM files", type='dcm', accept_multiple_files=True)
    
    if uploaded_files:
        if "slices" not in st.session_state:
            st.session_state.slices = load_dicom_files(uploaded_files)
        slices = st.session_state.slices
        # slices = load_dicom_files(uploaded_files)
        if "scores" not in st.session_state:
            scores = model.predict(slices)
            scores[:, -1] = 1 - scores[:, -1]
            st.session_state.scores = scores
        scores = st.session_state.scores
        # scores = model.predict(slices)

        if len(slices) > 0:
            slice_idx = st.slider("Select Slice", min_value=0, max_value=len(slices) - 1, value=0, step=1)
        
            show_slice_scores(scores, select_index=slice_idx)
        
            slice = slices[slice_idx]
            rescale_slope = slice.RescaleSlope if 'RescaleSlope' in slice else 1
            rescale_intercept = slice.RescaleIntercept if 'RescaleIntercept' in slice else -1024.

            # Windowing parameters input
            window_width = st.slider("Window Width", min_value=1, max_value=400, value=80, step=1)
            window_center = st.slider("Window Center", min_value=-100, max_value=300, value=40, step=1)
            
            show_slice(
                slice=slice, 
                scores=scores[slice_idx],
                window_center=window_center, 
                window_width=window_width, 
                rescale_slope=rescale_slope, 
                rescale_intercept=rescale_intercept,
            )
        else:
            st.write("No valid DICOM files were uploaded.")


if __name__ == "__main__":
    main()