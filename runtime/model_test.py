"""Unified independent server/client launcher. Models are rows in models.json."""
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from model_catalog import DEFAULT_CATALOG, contract, inspect_checkpoint, read_catalog, select_profile, validate_runtime_profile


def environment(profile):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               NO_ALBUMENTATIONS_UPDATE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    paths = [str(Path(__file__).resolve().parent)]
    transport = Path(__file__).resolve().parent / 'transport-py311'
    if profile['backend'] == 'molmoact2' and transport.is_dir():
        paths.append(str(transport))
    if profile.get('deps'):
        paths.append(profile['deps'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['list', 'inspect', 'infer', 'serve', 'client'])
    parser.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    parser.add_argument('--model')
    parser.add_argument('--control-hz', type=int)
    parser.add_argument('--bind')
    parser.add_argument('--url')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--instruction')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--execution-log', type=Path)
    parser.add_argument('--shadow-observation', type=Path)
    parser.add_argument('--shadow-report', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='print selection; do not launch')
    parser.add_argument('--in-runtime', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    catalog = read_catalog(args.catalog)
    if args.mode == 'list':
        for name in catalog['models']:
            profile = select_profile(catalog, name)
            print(json.dumps(contract(profile), ensure_ascii=False))
        return
    if not args.model:
        parser.error('--model is required')
    profile = select_profile(catalog, args.model, control_hz=args.control_hz)
    if args.mode == 'inspect':
        print(json.dumps(inspect_checkpoint(profile), indent=2))
        return
    address = (args.bind if args.mode == 'serve' else args.url) or profile['bind']
    parts = urlsplit(address)
    if parts.scheme != 'tcp' or not parts.hostname or not parts.port:
        parser.error('Use tcp://HOST:PORT for the independent test service')
    if args.mode in ('serve', 'infer'):
        if args.output is None:
            parser.error('serve/infer requires --output for a new result directory')
        if args.mode == 'infer' and not args.shadow_observation:
            parser.error('infer requires --shadow-observation')
        if profile['backend'] == 'g05' and profile['control_hz'] is None:
            parser.error('G05 frequency is unverified; supply --control-hz or set its profile explicitly')
    else:
        if args.shadow_observation:
            if args.live or not args.shadow_report:
                parser.error('Shadow mode requires --shadow-report and excludes --live')
        elif not args.live or not args.execution_log:
            parser.error('client requires shadow input, or --live with --execution-log')
        if not args.shadow_observation and profile['control_hz'] is None:
            parser.error('Live control_hz must be explicitly configured')
        if args.steps <= 0:
            parser.error('--steps must be positive')
    executable = profile['python'] if args.mode in ('serve', 'infer') else sys.executable
    if args.dry_run:
        print(json.dumps({'mode': args.mode, 'python': executable, 'address': address,
                          'contract': contract(profile), 'steps': args.steps,
                          'started': False}, indent=2))
        return
    if not args.in_runtime:
        validate_runtime_profile(profile)
        if not Path(executable).is_file():
            raise FileNotFoundError(f'Existing environment is missing: {executable}')
        arguments = list(sys.argv[1:] if argv is None else argv)
        os.execve(executable, [executable, '-B', '-u', str(Path(__file__).resolve()),
                              *arguments, '--in-runtime'], environment(profile))
    else:
        os.environ.update(environment(profile))
        if args.mode == 'infer':
            from model_service import infer_saved
            infer_saved(profile, args.shadow_observation, args.output,
                        args.instruction or profile['instruction'])
        elif args.mode == 'serve':
            from model_service import serve
            serve(profile, address, args.output)
        else:
            from model_client import run_client
            run_client(profile, url=address, instruction=args.instruction or profile['instruction'],
                       steps=args.steps, control_hz=profile['control_hz'],
                       execution_log=args.execution_log, shadow_observation=args.shadow_observation,
                       shadow_report=args.shadow_report)


if __name__ == '__main__':
    main()
