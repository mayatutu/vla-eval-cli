"""Check transport ABI isolation without ML imports, sockets or devices."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import model_service


class TransportPathTests(unittest.TestCase):
    def test_selects_only_the_running_python_version(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for suffix in ('310', '311', '312'):
                (root / ('transport-py' + suffix)).mkdir()
            for minor in (10, 11, 12):
                with self.subTest(minor=minor), \
                        patch.object(model_service, '__file__', str(root / 'model_service.py')), \
                        patch.object(model_service.sys, 'version_info', (3, minor, 0)), \
                        patch.object(model_service.sys, 'path', ['existing-path']):
                    model_service.configure_transport_path()
                    expected = str(root.resolve() / f'transport-py3{minor}')
                    self.assertEqual(model_service.sys.path, [expected, 'existing-path'])
                    model_service.configure_transport_path()
                    self.assertEqual(model_service.sys.path, [expected, 'existing-path'])

    def test_absent_matching_directory_preserves_existing_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'transport-py311').mkdir()
            with patch.object(model_service, '__file__', str(root / 'model_service.py')), \
                    patch.object(model_service.sys, 'version_info', (3, 10, 0)), \
                    patch.object(model_service.sys, 'path', ['existing-path']):
                model_service.configure_transport_path()
                self.assertEqual(model_service.sys.path, ['existing-path'])


if __name__ == '__main__':
    unittest.main()
