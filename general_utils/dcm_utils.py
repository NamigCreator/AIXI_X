import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import pydicom
import cv2
import warnings

warnings.filterwarnings("ignore")


def get_metadata_from_dcm(filename: Path) -> Dict[str, Any]:
    dcm = pydicom.dcmread(filename)
    try:
        window_center = dcm.WindowCenter
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = float(window_center[0])
        window_width = dcm.WindowWidth
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = float(window_width[0])
    except AttributeError:
        window_center, window_width = None, None
        print(f"No window center or width found for {filename}")

    data = {
        "sop_id": dcm.SOPInstanceUID[3:],
        "study_id": dcm.StudyInstanceUID[3:],
        "series_id": dcm.SeriesInstanceUID[3:],
        "patient_id": dcm.PatientID[3:],
        "slope": float(dcm.RescaleSlope),
        "intercept": float(dcm.RescaleIntercept),
        "window_center": window_center,  # already float or None
        "window_width": window_width,  # same as above
        "position_x": float(dcm.ImagePositionPatient[0]),
        "position_y": float(dcm.ImagePositionPatient[1]),
        "position_z": float(dcm.ImagePositionPatient[2]),
        "orientation_0": float(dcm.ImageOrientationPatient[0]),
        "orientation_1": float(dcm.ImageOrientationPatient[1]),
        "orientation_2": float(dcm.ImageOrientationPatient[2]),
        "orientation_3": float(dcm.ImageOrientationPatient[3]),
        "orientation_4": float(dcm.ImageOrientationPatient[4]),
        "orientation_5": float(dcm.ImageOrientationPatient[5]),
        "pixel_spacing_x": float(dcm.PixelSpacing[0]),
        "pixel_spacing_y": float(dcm.PixelSpacing[1]),
    }
    return data


def change_image_size(
    img: np.ndarray,
    image_size: Union[int, Tuple[int, int]] = (256, 256),
) -> np.ndarray:
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    img = cv2.resize(img, dsize=image_size, interpolation=cv2.INTER_CUBIC)
    return img


def get_image_from_dcm(
    filename: Path,
    image_size: Optional[Union[int, Tuple[int, int]]] = (256, 256),
) -> np.ndarray:
    try:
        dcm = pydicom.dcmread(str(filename), force=True)
    except:
        dcm = pydicom.dcmread(filename, force=True)
        
    try:
        img = dcm.pixel_array
        if image_size is not None:
            img = change_image_size(img, image_size=image_size)
        return img
    except AttributeError:
        print(f"Something bad with {filename}")


def window_image(
    img: np.ndarray,
    window_center: float = 40,
    window_width: float = 80,
    intercept: float = -1024.0,
    slope: float = 1.0,
) -> np.ndarray:
    img = img.astype(np.float32) * slope + intercept
    img = np.clip(
        img, window_center - window_width / 2, window_center + window_width / 2
    )
    img = (img - (window_center - window_width / 2)) / window_width
    return img
