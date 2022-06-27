from typing import Tuple, List

import cv2
from path import Path #This is used to specify the path for the files.

from dataloader_iam import Batch 
from model_testing import Model, DecoderType 
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_size() -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words)."""
    return 128, 32


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def infer(model: Model, fn_img: Path) -> None:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)
    

    batch = Batch([img], None, 1) # Batch(imgs, gt_texts, batch_size)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')
    return recognized[0], probability[0]
