"""Print required small-asset hashes for a selected checkpoint; never modify it."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['openpi', 'groot_isaac'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-id')
    args = parser.parse_args()
    if args.backend == 'openpi':
        if not args.dataset_id:
            parser.error('OpenPI requires --dataset-id, matching assets/<dataset-id>/')
        names = ['train_config.py', f'assets/{args.dataset_id}/norm_stats.json']
    else:
        names = ['config.json', 'processor/processor_config.json', 'processor/statistics.json',
                 'processor/embodiment_id.json', 'model.safetensors.index.json']
    root = args.checkpoint.resolve()
    hashes = {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Invalid or missing asset: ' + name)
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError('Expected a small configuration/index file: ' + name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({'asset_sha256': hashes}, indent=2))


if __name__ == '__main__':
    main()
