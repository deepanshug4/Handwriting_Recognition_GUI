import argparse
import json
from pickle import TRUE
from typing import Tuple, List

import editdistance
from path import Path #This is used to specify the path for the files.

from dataloader_iam import DataLoaderIAM 
from model_training import Model, DecoderType 
from preprocessor import Preprocessor


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size() -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or lines)."""
    
    return 256, get_img_height() 
    

def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def train(model: Model,
          loader: DataLoaderIAM,
          early_stopping: int = 25):
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    best_char_error_rate = float('inf')  # best validation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)
        preprocessor = Preprocessor(get_img_size(), data_augmentation=True)
        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save() #to save the model
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: DataLoaderIAM) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size())
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch) # to get the output of the test images 

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line. Used for adding different modifications to the training model."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate'], default='train')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch'], default='beamsearch')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()


def main():
    """Main function."""

    # parse arguments
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch}

    # set CTC decoder
    decoder_type = decoder_mapping[args.decoder]

    if args.mode == 'train':
        # train the model
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        char_list = loader.char_list

        model = Model(char_list, decoder_type) 
        train(model, loader, early_stopping=args.early_stopping)
    
    elif args.mode == 'validate':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        model = Model(char_list_from_file(), decoder_type, must_restore=True)
        validate(model, loader)


if __name__ == '__main__':
    main()
