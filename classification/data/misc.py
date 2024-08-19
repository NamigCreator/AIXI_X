import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List, Literal
import pydicom
import cv2
import warnings
warnings.filterwarnings("ignore")
import SimpleITK as sitk
from tempfile import NamedTemporaryFile, TemporaryDirectory
from copy import deepcopy
import matplotlib


class_name_to_index = {
    "epidural": 0,
    "intraparenchymal": 1,
    "intraventricular": 2,
    "subarachnoid": 3,
    "subdural": 4,
    "any": 5,
}
n_classes = len(class_name_to_index)
class_names = list(class_name_to_index.keys())


def get_metadata_from_dcm(filename: Path) -> Dict[str, Any]:
    dcm = pydicom.dcmread(filename)
    window_center = dcm.WindowCenter
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = window_center[0]
    window_width = dcm.WindowWidth
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = window_width[0]
    data = {
        "sop_id": dcm.SOPInstanceUID[3:],
        "study_id": dcm.StudyInstanceUID[3:],
        "series_id": dcm.SeriesInstanceUID[3:],
        "patient_id": dcm.PatientID[3:],
        "slope": float(dcm.RescaleSlope),
        "intercept": float(dcm.RescaleIntercept),
        "window_center": float(window_center),
        "window_width": float(window_width),
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


_cv2_interpolation_dict = {
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "MAX": cv2.INTER_MAX,
}

def change_image_size(
        img: np.ndarray, 
        image_size : Union[int, Tuple[int, int]] = (256, 256),
        interpolation : Literal["cubic", "nearest", "max"] = "cubic",
        ) -> np.ndarray:
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    img = cv2.resize(img.astype(np.float32), 
        dsize=image_size, interpolation=_cv2_interpolation_dict[interpolation])
    return img


def get_image_from_dcm(
        filename: Path,
        image_size : Optional[Union[int, Tuple[int, int]]] = (256, 256),
        ) -> np.ndarray:
    dcm = pydicom.dcmread(str(filename))
    img = dcm.pixel_array
    if image_size is not None:
        img = change_image_size(img, image_size=image_size)
    return img


def window_image(
        img: np.ndarray,
        window_center : float = 40,
        window_width : float = 80,
        intercept : float = -1024.,
        slope : float = 1.,
        ) -> np.ndarray:
    img = img.astype(np.float32) * slope + intercept
    img = np.clip(img, window_center - window_width / 2, window_center + window_width / 2)
    img = (img - (window_center - window_width / 2)) / window_width
    return img


def remove_non_head(
        image: sitk.Image,
        threshold_connected_lower : int = -100,
        threshold_connected_upper : int = 4000,
        dilate_radius : int = 4,
        threshold_binary_lower : int = -400,
        threshold_binary_upper : int = 4000,
        default_lower_value : int = -2048,         
        ) -> sitk.Image:
    s = image.GetSize()
    seed = (s[0]//2, s[1]//2, s[2]//2)
    region_growing = sitk.ConnectedThreshold(image, seedList=[seed],
        lower=threshold_connected_lower, upper=threshold_connected_upper)
    
    binary_dilate = sitk.BinaryDilateImageFilter()
    binary_dilate.SetKernelRadius(dilate_radius)
    region_growing = binary_dilate.Execute(region_growing)
    
    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(threshold_binary_lower)
    threshold_filter.SetUpperThreshold(threshold_binary_upper)
    image_thr = threshold_filter.Execute(image)
    
    not_filter = sitk.NotImageFilter()
    and_filter = sitk.AndImageFilter()
    image_thr = and_filter.Execute(not_filter.Execute(region_growing), image_thr)
    
    mask_filter = sitk.MaskImageFilter()
    mask_filter.SetOutsideValue(default_lower_value)
    image_masked = mask_filter.Execute(image, not_filter.Execute(image_thr))

    return image_masked


def read_sitk_image(
        path : Optional[Path] = None, 
        names : Optional[List[str]] = None,
        as_array : bool = False,
        ) -> Union[sitk.Image, np.ndarray]:
    if path is not None and path.is_file():
        image = sitk.ReadImage(path)
    else:
        reader = sitk.ImageSeriesReader()
        if path is not None:
            if names is None:
                names = reader.GetGDCMSeriesFileNames(str(path))
            else:
                names = [path.joinpath(f"{n}.dcm") for n in names]
                if not any(n.exists() for n in names):
                    names = [path.joinpath(f"ID_{n}.dcm") for n in names]
        names = [str(n) for n in names]
        names = tuple(names)
        reader.SetFileNames(names)
        image = reader.Execute()
    if as_array:
        image = sitk.GetArrayFromImage(image)
    return image


def reorient(image: sitk.Image, default_value : int = -2048) -> sitk.Image:
    m = image.GetDirection()
    m = np.reshape(m, (3, 3))
    m = m[1:3, 1:3]
    if np.all(np.trace(m) == 1):
        return image
    transform = sitk.AffineTransform(2)
    transform.SetMatrix(np.reshape(m, -1))
    resampled = deepcopy(image)
    for i in range(image.GetSize()[2]):
        i_slice = image[:, :, i]
        resampled[:, :, i] = sitk.Resample(i_slice,
            transform=transform, interpolator=sitk.sitkLinear, defaultPixelValue=default_value)
    return resampled


def crop_image(
        image: sitk.Image, 
        threshold_lower : int = -400, 
        threshold_upper : int = 4000,
        square : bool = True,
        default_value : int = -2048,
        get_indexes : bool = False,
        leave_all_slices : bool = True,
        ) -> Union[sitk.Image, Tuple[sitk.Image, Tuple[int, int]]]:
    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(threshold_lower)
    threshold_filter.SetUpperThreshold(threshold_upper)
    mask = threshold_filter.Execute(image)

    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(mask)
    bounding_box = lsif.GetBoundingBox(1)

    if leave_all_slices:
        bounding_box = list(bounding_box)
        bounding_box[2] = 0
        bounding_box[5] = image.GetSize()[2]

    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetRegionOfInterest(bounding_box)
    result = roi_filter.Execute(image)
    
    if square:
        size = result.GetSize()
        sm = max(size[0], size[1])
        origin = list(result.GetOrigin())
        origin[0] += (size[0]-sm)/2 * result.GetSpacing()[0]
        origin[1] += (size[1]-sm)/2 * result.GetSpacing()[1]
        origin = tuple(origin)
        result = sitk.Resample(result, 
            size=(sm, sm, size[2]),
            outputOrigin=origin,
            outputSpacing=result.GetSpacing(),
            outputDirection=result.GetDirection(),
            defaultPixelValue=default_value,
        )
    if get_indexes:
        return result, [bounding_box[2], bounding_box[5]]
    else:
        return result


def write_sitk_image_as_dicoms(
        image: sitk.Image, 
        folder: Path, 
        series_id : Optional[str] = None,
        ) -> List[str]:
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkInt16)

    if not isinstance(folder, Path):
        folder = Path(folder)
    folder.mkdir(exist_ok=True)

    direction = image.GetDirection()
    tag_values = [("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
        direction[1],direction[4],direction[7]))))]

    out_names = []
    for i in range(image.GetDepth()):
        image_slice = image[:, :, i]
        image_slice = castFilter.Execute(image_slice)
        for tag, value in tag_values:
            image_slice.SetMetaData(tag, value)
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str,image.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i))
        if series_id is not None:
            image_slice.SetMetaData("0020|000e", series_id)
        name = f"{i}"
        filename = folder.joinpath(f"{name}.dcm")
        writer.SetFileName(str(filename))
        writer.Execute(image_slice)
        out_names.append(name)
    return out_names


def load_study(
        folder : Optional[Path] = None,
        names : Optional[List[str]] = None,
        reorient_image : bool = False,
        remove : bool = True,
        crop : bool = True,
        as_array : bool = True, 
        as_pydicom : bool = False,
        ) -> Union[np.ndarray, sitk.Image]:
    image = read_sitk_image(folder, names=names)
    if reorient_image:
        image = reorient(image)
    if remove:
        image = remove_non_head(image)
    if crop:
        image = crop_image(image)
    if as_array:
        return sitk.GetArrayFromImage(image)
    elif as_pydicom:
        # dicoms = []
        # with NamedTemporaryFile(suffix=".dcm") as tmpfile:
        #     writer = sitk.ImageFileWriter()
        #     writer.KeepOriginalImageUIDOn()

        #     direction = image.GetDirection()
        #     tag_values = [("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
        #         direction[1],direction[4],direction[7]))))]

        #     for i in range(image.GetDepth()):
        #         image_slice = image[:, :, i]
        #         for tag, value in tag_values:
        #             image_slice.SetMetaData(tag, value)
        #         image_slice.SetMetaData("0020|0032", '\\'.join(map(str,image.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        #         image_slice.SetMetaData("0020|0013", str(i))
        #         writer.SetFileName(tmpfile.name)
        #         writer.Execute(image_slice)
        #         dcm = pydicom.dcmread(tmpfile.name)
        #         dicoms.append(dcm)
        with TemporaryDirectory() as tmpdir:
            out_names = write_sitk_image_as_dicoms(image, tmpdir)
            dicoms = [pydicom.dcmread(Path(tmpdir).joinpath(f"{name}.dcm")) for name in out_names]
        return dicoms
    else:
        return image
    

def get_colors(n: int) -> np.ndarray:
    cmap = matplotlib.cm.get_cmap("hsv")
    s = 1./(n+1) / 2.
    step = 1. / (n+1)
    colors = []
    for i in range(n):
        colors.append(cmap(s + step*i))
    return np.array(colors, np.float32)

def make_image_from_slice_channel(slice: np.ndarray, channel: int = 0) -> np.ndarray:
    channel = slice[channel]
    if not isinstance(channel, np.ndarray):
        channel = channel.numpy()
    return np.array([channel]*3)

def make_image_from_classes(
        mask: np.ndarray, 
        n_classes : Optional[int] = None,
        colors : Optional[np.ndarray] = None,
        ) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
        n_classes = None
    if mask.ndim == 2 and n_classes is None:
        if colors is None:
            return np.array([mask]*3)
        else:
            if colors.ndim == 2:
                colors = colors[0]
            image = np.zeros((3, mask.shape[0], mask.shape[1]), np.float32)
            image[0] = mask * colors[0]
            image[1] = mask * colors[1]
            image[2] = mask * colors[2]
            return image
    elif mask.ndim == 2 and n_classes is not None:
        if colors is None:
            colors = get_colors(n_classes-1)
        elif colors.ndim == 1:
            colors = np.expand_dims(colors, 0)
        image = np.zeros((3, mask.shape[0], mask.shape[1]), np.float32)
        for i in range(n_classes-1):
            m = mask == i+1
            image[0, m] = colors[i, 0]
            image[1, m] = colors[i, 1]
            image[2, m] = colors[i, 2]
        return image
    else:
        n_classes = mask.shape[0]
        if colors is None:
            colors = get_colors(n_classes-1)
        image = np.zeros((3, mask.shape[1], mask.shape[2]), np.float32)
        for i in range(n_classes-1):
            image[0] += mask[i+1] * colors[i, 0]
            image[1] += mask[i+1] * colors[i, 1]
            image[2] += mask[i+1] * colors[i, 2]
        return image