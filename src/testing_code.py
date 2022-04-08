from typing import Tuple, List

import cv2
from path import Path #This is used to specify the path for the files.

from dataloader_iam import Batch 
from model import Model, DecoderType 
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_size(type_of_model) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words)."""
    if type_of_model:
        return 256,32
    else:
        return 128, 32


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def infer(model: Model, fn_img: Path, type_of_model) -> None:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(type_of_model), dynamic_width=True, padding=16,model_mode=type_of_model)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True, 1)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')
    return recognized[0], probability[0]
