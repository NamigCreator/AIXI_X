import sys
from pathlib import Path
import argparse
from typing import Optional, List, Dict
import numpy as np
import pickle

_folder_current = Path(__file__).parent
_folder = _folder_current.parent.parent
sys.path.append(str(_folder))
from classification.pipeline import ClassificationPipeline


def predict(
        model_name: str,
        folder: Path,
        ids : Optional[Path] = None,
        filename_out : Optional[Path] = None,
        **kwargs,
        ) -> Dict[str, np.ndarray]:
    
    model = ClassificationPipeline(model_name=model_name, **kwargs)

    result = model.predict(folder=folder, ids=ids, get_full_data=True)
    if filename_out is not None:
        pickle.dump(result, open(filename_out, "wb"))
    return result


def _parse_args(args : Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Runs model inference")
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("folder", type=Path, 
        help="Path to folder containing input .dcm files")
    parser.add_argument("out", type=Path, default=None,
        help="Output .pkl file with results")
    parser.add_argument("--ids", type=Path, default=None,
        help="Path to .txt file with list of ids of .dcm files")
    parser.add_argument("--checkpoint", type=str, default="best",
        help="Model checkpoint name")
    parser.add_argument("--batch_size", type=int, default=None,
        help="Batch size")
    return parser.parse_args(args=args)


if __name__ == "__main__":
    args = _parse_args()
    predict(
        model_name=args.model_name,
        folder=args.folder,
        ids=args.ids,
        filename_out=args.out,
        checkpoint_name=args.checkpoint,
        batch_size=args.batch_size,
    )