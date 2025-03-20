from typing import List, Optional
import numpy as np
from PIL import Image, ImageFile
import pydicom
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .data.misc import window_image, class_names
from utils import init_logger
from .mask import add_segmentation_mask_to_image

logger = init_logger(__name__)


def convert_images(images: List[np.ndarray]) -> List[ImageFile.ImageFile]:
    images_out = []
    for img in images:
        if img.ndim == 3:
            img = img[0]
        img = Image.fromarray((img[0]*255).astype(np.uint8))
        images_out.append(img)
    return images_out


def load_model(model_id : str = "mistral-community/pixtral-12b"):
    logger.debug("Initializing LLM model")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        load_in_8bit=True,
    )
    logger.debug("Finished initializing LLM model")
    return [processor, model]


_default_prompt = """
You are provided with a slice of CT study of a brain.
List all the abnormalities detected on the image, provide description and specify location.
Pay more attention to potential hemorrhage.
Do not provide findings or recommendations. 
Be concise and accurate.
"""

def run_model_single_slice(processor, model, image: np.ndarray, prompt : str = _default_prompt) -> str:
    chat = [{
        "role": "user",
        "content": [
            {"type": "text", "content": prompt},
            {"type": "image"},
        ]
    }]
    prompt = processor.apply_chat_template(chat)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=2000)
    output = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output


_default_prompt_merge = """
You are provided with output of multiple models describing different slices of the same CT study of a brain.
Summarize these into a single report.
"""

def summarize_output(processor, model, outputs: List[str], prompt : str = _default_prompt_merge) -> str:
    texts = [_default_prompt_merge] + outputs
    chat = [{
        "role": "user",
        "content": [{"type": "text", "content": s} for s in texts]
    }]
    prompt = processor.apply_chat_template(chat)
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=2000)
    output = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output


"""
Output should be in the following format:
1. Type of abnormality 1
    Location: location
    Description: description
2. Type of abnormality 2
    Location: location
    Description: description
...
You should not write anything else except these points.

Provide output as dict in json format.
{"abnormalities": [{"type": "<abnormality type>", "location": "<location in brain>", "description": "<description of abnormality>"}], 
"global": "<some global description of study>"}


        "description": "Blood is visible within the ventricular system, indicating intraventricular hemorrhage. This can occur as a result of intraparenchymal hemorrhage extension or direct bleeding into the ventricles."
        "description": "A localized area of high density within the brain parenchyma suggests the presence of an intraparenchymal hemorrhage. This is typically indicative of bleeding within the brain tissue itself, which can result from various causes including hypertension, vascular malformations, or trauma."
        "description": There is evidence of blood in the subarachnoid space, particularly noted around the cerebral convexities and the sulci. This indicates a subarachnoid hemorrhage, which is a serious condition often associated with conditions such as aneurysmal rupture or vascular malformations."


"""

enduser_token = "[ENDUSER]"
_default_prompt_mult = """
You are provided with multiple slices of a single CT study of a brain.
List all the abnormalities detected on images, provide description and specify location.
Pay more attention to potential hemorrhage.
Do not provide findings or recommendations. 
If there is the same abnormality on different slices do not duplicate information. 
If at least one abnormality is detected, do not write anything else except these points.
Or, if everything is fine, just describe the brain in overall.
""" 
_default_prompt_mult_1 = """
You are provided with multiple slices of a single CT study of a brain.
"""
_default_prompt_mult_2 = """
Describe these hemorrhages and provide the following information:
location in brain, description of hemorrhage.
Do not provide any other information, just several items about detected hemorrhages.
List only hemorrhages that were described here.

Provide output in json format as a list.
Look at this example of output from some other study:

```json
[
    {
        "type": "intraventricular",
        "location": "left frontal lobe",
        "description": "Hemorrhage within the left lateral ventricle, likely causing increased intracranial pressure and potential obstruction of cerebrospinal fluid flow."
    },
    {
        "type": "subarachnoid",
        "location": "around the cerebral convexities and the sulci",
        "description": There is evidence of blood in the subarachnoid space, particularly noted around the cerebral convexities and the sulci. This indicates a subarachnoid hemorrhage, which is a serious condition often associated with conditions such as aneurysmal rupture or vascular malformations."
    }
]```

"""

_default_prompt_mult_2 = """
Describe your findings of this study, focusing on the following points: 
 - CT-attenuation values
 - basal ganglia
 - thalami
 - brainstem
 - cerebellar hemispheres
 - vermis
 - ventricular system
 - midline shift
 - basal cisterns
 - sell and suprasellar regions
 - skull vault
All of these should go as itemized list, each point should be a separate item.

If there are hemorrhages found, provide additional points with localization and detailed description of each detected hemorrhage.
If the same hemorrhage is found on different slices, do not duplicate this information, just say that it was detected on several slices.

Provide the overall impression of the study in the end.

In total, there should be two headings: Findings and Impression.
"""

_default_prompt_mult_2 += enduser_token

def run_model_multiple(processor, model, 
        images: np.ndarray, 
        # prompt : str = _default_prompt_mult,
        clear_output : bool = True,
        segm_masks : Optional[np.ndarray] = None,
        scores : Optional[np.ndarray] = None,
        score_threshold : float = 0.1,
        volume : Optional[str] = None,
        ) -> str:

    if scores is not None and len(scores) > 0:
        score_string = [
            "Scores of hemorrhages detected with classes:",
        ]
        for slice_index, score in enumerate(scores):
            string = []
            for i in np.where(score >= score_threshold)[0]:
                if class_names[i] != "any" or len(string) == 0:
                    string.append(f"{class_names[i]}: {score[i]:.3f}")
            if len(string) > 0:
                string = "; ".join(string)
            # else:
            #     string = "no hemorrhage detected"
                score_string.append(f"Slice {slice_index}: {string}")
        if len(score_string) > 1:
            score_string = "\n".join(score_string) + "\n"
        else:
            score_string = "No hemorrhages detected by classification model."
    else:
        score_string = "No hemorrhages detected by classification model."
        
    # chat = [{
    #     "role": "user",
    #     "content": [
    #         {"type": "text", "content": _default_prompt_mult_1},
    #     ] + [{"type": "image"} for _ in images] + [
    #         {"type": "text", "content": score_string},
    #         {"type": "text", "content": _default_prompt_mult_2},
    #     ]
    # }]
    content = [{"type": "text", "content": _default_prompt_mult_1}]
    if segm_masks is not None and volume is not None and volume > 0:
        content.append({"type": "text", 
            "content": "Hemorrhage detected by hemorrhage detection model are colored red on images."})
    content += [{"type": "image"} for _ in images]
    if score_string is not None:
        content.append({"type": "text", "content": score_string})
    if volume is not None and volume > 0:
        volume_string = f"Volume of the detected hemorrhage is {volume:.1f} mL.\n"
        content.append({"type": "text", "content": volume_string})
    content.append({"type": "text", "content": _default_prompt_mult_2})
    
    chat = [{"role": "user", "content": content}]
    if isinstance(images[0], pydicom.FileDataset):
        rescale_slope = images[0].RescaleSlope if "RescaleSlope" in images[0] else 1
        rescale_intercept = images[0].RescaleIntercept if "RescaleIntercept" in images[0] else -1024.0
        images = [window_image(im.pixel_array, slope=rescale_slope, intercept=rescale_intercept)
            for im in images]
    if segm_masks is not None:
        images = [add_segmentation_mask_to_image(image, mask, mask_alpha=0.3)
            for image, mask in zip(images, segm_masks)]
    if isinstance(images[0], np.ndarray):
        images = convert_images(images)
    logger.debug(f"Running LLM model with inputs from [{len(images)}] images")
    prompt = processor.apply_chat_template(chat)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=2000)
    output = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
    if clear_output:
        istart = output.find(enduser_token) + len(enduser_token)
        output = output[istart:]
    return output


def select_representative_slices(
        scores: np.ndarray,
        score_threshold : Optional[float] = 0.1,
        max_num : Optional[int] = 10,
        min_dist : Optional[int] = 4,
        min_index : Optional[int] = 0,
        max_index : Optional[int] = 0,
        ) -> np.ndarray:
    logger.debug(f"Selecting representative slices from [{len(scores)}]")
    class_index = -1
    scores = scores[:, class_index]
    inds = np.argsort(scores)[::-1]
    if min_index is not None and min_index > 0:
        inds = inds[inds >= min_index]
    if max_index is not None and max_index > 0:
        inds = inds[inds < len(scores)-max_index]
    if min_dist is not None:
        inds_rel = [inds[0]]
        for i in inds[1:]:
            if all(abs(i-j)>=min_dist for j in inds_rel):
                inds_rel.append(i)
        inds = np.array(inds_rel)
        logger.debug(f"\tAfter distance-based filter [dist={min_dist}] : [{len(inds)}]")
    if score_threshold is not None:
        inds = inds[np.where(scores[inds] >= score_threshold)[0]]
        logger.debug(f"\tAfter score threshold [{score_threshold:.2f}] : [{len(inds)}]")
    if max_num is not None:
        inds = inds[:max_num]
    logger.debug(f"Representative slices [{len(inds)}] : {inds}")
    return inds