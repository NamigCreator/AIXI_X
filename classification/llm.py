from typing import List, Optional
import numpy as np
from PIL import Image, ImageFile
import pydicom
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .data.misc import window_image
from utils import init_logger

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


_default_prompt_mult = """
You are provided with multiple slices of a single CT study of a brain.
List all the abnormalities detected on images, provide description and specify location.
Pay more attention to potential hemorrhage.
Do not provide findings or recommendations. 
If there is the same abnormality on different slices do not duplicate information.
"""
def run_model_multiple(processor, model, images: np.ndarray, prompt : str = _default_prompt_mult) -> str:
    chat = [{
        "role": "user",
        "content": [
            {"type": "text", "content": prompt},
        ] + [{"type": "image"} for _ in images]
    }]
    if isinstance(images[0], pydicom.FileDataset):
        rescale_slope = images[0].RescaleSlope if "RescaleSlope" in images[0] else 1
        rescale_intercept = images[0].RescaleIntercept if "RescaleIntercept" in images[0] else -1024.0
        images = [window_image(im.pixel_array, slope=rescale_slope, intercept=rescale_intercept)
            for im in images]
    if isinstance(images[0], np.ndarray):
        images = convert_images(images)
    logger.debug(f"Running LLM model with inputs from [{len(images)}] images")
    prompt = processor.apply_chat_template(chat)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, max_new_tokens=2000)
    output = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output


def select_representative_slices(
        scores: np.ndarray,
        score_threshold : Optional[float] = 0.1,
        max_num : Optional[int] = 10,
        min_dist : Optional[int] = 4,
        ) -> np.ndarray:
    logger.debug(f"Selecting representative slices from [{len(scores)}]")
    class_index = -1
    scores = scores[:, class_index]
    inds = np.argsort(scores)[::-1]
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