"""Standalone entry point; --help/--plan do not start devices."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime'))
from evaluate import main
if __name__ == '__main__':
    main()
