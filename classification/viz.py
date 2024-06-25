import sys
import numpy as np
from matplotlib import pyplot as plt
import pydicom
from pydicom.errors import InvalidDicomError
import streamlit as st
from typing import List, Optional
from copy import deepcopy
from pathlib import Path

_folder_current = Path(__file__).parent
_folder = _folder_current.parent
sys.path.append(str(_folder))
from classification.data.misc import window_image, class_names
from classification.pipeline import ClassificationPipeline
class_names = deepcopy(class_names)
class_names[-1] = "none"


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

def show_slice(
        slice: pydicom.FileDataset, 
        scores : Optional[np.ndarray] = None,
        class_names : Optional[List] = class_names,
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
    titles = [
        f"Z-position: {slice.ImagePositionPatient[2]}",
    ]
    if scores is not None:
        titles.append(f"Scores:")
        titles += [f"\t{c:15s} : {s:.3f}" for c, s in zip(class_names, scores)]
    ax.set_title("\n".join(titles))
    plt.axis("off")
    st.pyplot(fig)
    

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
            model_name="2d_v11_seresnext50_lr1-4", 
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
        if "scores" not in st.session_state:
            st.session_state.scores = model.predict(slices)
        scores = st.session_state.scores
        
        if len(slices) > 0:
            slice_idx = st.slider("Select Slice", min_value=0, max_value=len(slices) - 1, value=0, step=1)
            slice = slices[slice_idx]
            rescale_slope = slice.RescaleSlope if 'RescaleSlope' in slice else 1
            rescale_intercept = slice.RescaleIntercept if 'RescaleIntercept' in slice else 0

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