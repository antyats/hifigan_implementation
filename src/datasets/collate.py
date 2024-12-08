import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from src.transforms.spectrogram import MelSpectrogram

def collate_fn(dataset_items):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    get_spectrogram = MelSpectrogram()

    batch = {}

    if dataset_items[0]['audio'] != '':
        batch["audio"] = pad_sequence(
            [x for item in dataset_items for x in item["audio"]], 
            batch_first=True
        )

        batch["spectrogram"] = pad_sequence(
            [x for item in dataset_items for x in get_spectrogram(item["audio"])], 
            batch_first=True,
            padding_value=-11,
        )

    batch["path"] = [item["path"] for item in dataset_items]

    batch["text"] = [item["text"] for item in dataset_items]


    return batch