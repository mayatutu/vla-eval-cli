"""Validate configuration without connecting to SSH, cameras, GPU or robot."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from model_catalog import read_catalog, select_profile, validate_runtime_profile, contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    catalog = read_catalog(args.catalog)
    ids = args.models or list(catalog['models'])
    if not ids:
        parser.error('The catalog is empty')
    for name in ids:
        profile = select_profile(catalog, name)
        validate_runtime_profile(profile)
        print(name, contract(profile)['profile_sha256'])
    print('Configuration values validated; file availability and devices have not been tested.')


if __name__ == '__main__':
    main()
