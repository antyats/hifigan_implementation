from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, dir_path, transcription_dir='transcriptions', *args, **kwargs):
        data = []
        for path in Path(f'{dir_path}/{transcription_dir}').iterdir():
            entry = {}
            entry['path'] = str(path)
            if path.suffix in ['.txt']:
                with path.open() as f:
                    entry['text'] = f.read().strip()
            entry['audio_len'] = 0
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)