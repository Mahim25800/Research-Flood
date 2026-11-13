"""Prepare private Kaggle submission directories for the full v2 notebooks."""
import json
from pathlib import Path
from shutil import copyfile


def main():
    generated = Path(__file__).resolve().parent / 'generated'
    for dataset in ('s1f', 'ks', 'mm'):
        name = f'{dataset}_full.ipynb'
        folder = generated / 'kaggle_full_v2' / dataset
        folder.mkdir(parents=True, exist_ok=True)
        copyfile(generated / 'baselines_v2' / name, folder / name)
        metadata = dict(id=f'mahim2580/px1-baselines-v2-{dataset}-full',
                        title=f'PX1 baselines v2 {dataset.upper()} full', code_file=name,
                        language='python', kernel_type='notebook', is_private=True,
                        enable_gpu=True, enable_internet=True,
                        dataset_sources=['mahim2580/px1-flood-bundle'],
                        competition_sources=[], kernel_sources=[])
        (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(folder)


if __name__ == '__main__':
    main()
