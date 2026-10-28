"""Build an isolated G05 source overlay from an already installed upstream repo."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.repo.resolve() / 'src/g05'
    if not source.is_dir():
        parser.error('Expected --repo/src/g05')
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    if output.is_relative_to(source):
        parser.error('Output must be outside the source package')
    patch = Path(__file__).resolve().parents[1] / 'patches/g05/g05/models/g05/inferencer.py'
    data = patch.read_bytes()
    shutil.copytree(source, output / 'g05', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (output / 'g05/models/g05/inferencer.py').write_bytes(data)
    print('Set g05_overlay to:', output)


if __name__ == '__main__':
    main()
