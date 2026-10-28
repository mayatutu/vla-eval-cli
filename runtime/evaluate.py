"""Independent command-line evaluation; importing this never opens devices or SSH."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import sys
import uuid

from evaluation_limits import action_seconds_for_profile

from evaluation_session import export_eligible, select_task_model_ids

def task_profiles(catalog, task, model_ids=None):
    from model_catalog import select_profile
    selected = select_task_model_ids(task, model_ids)
    if model_ids is not None:
        missing = [model for model in selected if model not in catalog['models']]
        if missing:
            raise ValueError('Selected checkpoints unavailable: ' + ', '.join(missing))
    profiles = [select_profile(catalog, model) for model in selected
                if model in catalog['models']]
    if not profiles:
        raise ValueError('No checkpoints available for task: ' + task)
    return profiles


def resume_profiles(catalog, root, saved):
    """Preserve a session's original ordered models when the catalog grows."""
    from model_catalog import select_profile
    if not isinstance(saved.get('task'), str) or not saved['task'].strip():
        raise RuntimeError('未知的存档任务，未修改会话。')
    model_ids = saved.get('model_ids')
    if (not isinstance(model_ids, list) or not model_ids
            or any(not isinstance(model, str) or not model for model in model_ids)
            or len(set(model_ids)) != len(model_ids)):
        raise RuntimeError('存档模型顺序无效，未修改会话。')
    missing = [model for model in model_ids if model not in catalog['models']]
    if missing:
        raise RuntimeError('存档模型不在当前目录中，未修改会话: ' + ', '.join(missing))
    profiles = [select_profile(catalog, model) for model in model_ids]
    if json.loads((Path(root) / 'profiles.json').read_text(encoding='utf-8')) != {'profiles': profiles}:
        raise RuntimeError('模型配置与会话存档不同；保留原会话，不自动替换。')
    return profiles


def client_idle(*, cameras=False):
    """OS process and camera ownership checks only; never opens a camera or robot connection."""
    import subprocess
    if sys.platform != 'linux':
        raise RuntimeError('实机入口只允许在 MiniPC Linux 上运行；本机可使用 --plan。')
    markers = ('minipc_local.py', 'robot_cli', 'policy_cli', 'run-open-local-eval')
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal() or int(process.name) == os.getpid():
            continue
        try:
            args = (process / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError('无法核对进程占用: ' + process.name) from error
        if (any(any(marker in arg for marker in markers) for arg in args)
                or any(Path(arg).name == 'model_test.py' for arg in args) and 'client' in args
                or any(Path(arg).name in ('evaluate.py', 'evaluate_web.py') for arg in args) and '--run' in args):
            raise RuntimeError('检测到其他 Client/评测进程: ' + process.name)
    if cameras:
        devices = sorted(str(p) for p in Path('/dev').glob('video*'))
        if not devices:
            raise RuntimeError('未发现相机视频设备。')
        owners = subprocess.run(['fuser', *devices], capture_output=True, text=True, timeout=5)
        if owners.returncode != 1 or owners.stdout.strip() or owners.stderr.strip():
            raise RuntimeError('相机被占用或无法确认空闲: ' + owners.stdout + owners.stderr)


class Evaluation:
    """UI-independent operations reused by the future webpage."""
    def __init__(self, session, profiles, remote, *, trial=None, exporter=None,
                 guard=client_idle, emit=print):
        self.session = session
        self.profiles = {p['model_id']: p for p in profiles}
        self.remote = remote
        self.guard, self.emit = guard, emit
        self.trial, self.exporter = trial, exporter
        self.ready_model = None
        self.active = None
        self.stop_reason = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='suite-trial')

    def _idle(self):
        self.poll()
        if self.active is not None:
            raise RuntimeError('测试尚在运行；可输入 finish 正常提前结束，或 pause 中断。')

    def _pending_score(self):
        return next((a for a in reversed(self.session.state['attempts'])
                     if a['status'] == 'awaiting_score'), None)

    def status(self):
        self.poll()
        state = self.session.snapshot()
        self.emit(f"任务 {state['task']} | 每次上限 {state['steps']} 步 / 首条关节动作起{action_seconds_for_profile(self.profiles[self.session.current_model or state['model_ids'][-1]]):g}秒")
        for name in state['model_ids']:
            row = state['progress'][name]
            marker = '>' if name == self.session.current_model else ' '
            self.emit(f"{marker} {name}: {row['completed']}/{row['target']} 次，尝试 {row['attempts']} 次")
        self.emit('当前状态: ' + ('运行中' if self.active else
                  '待评分' if self._pending_score() else
                  '全部试验已完成' if state['complete'] else
                  '就绪，等待 start' if self.ready_model == self.session.current_model and self.ready_model
                  else '等待 load / next'))
        for attempt in state['attempts']:
            if attempt['export']['status'] in ('pending', 'failed', 'in_progress'):
                self.emit(f"导出待处理 {attempt['id']}: {attempt['export']['status']}")
        return state

    def load(self):
        self._idle()
        if self._pending_score():
            raise RuntimeError('请先为上一试验评分。')
        model = self.session.current_model
        if model is None:
            raise RuntimeError('所有模型已完成。')
        from model_catalog import contract
        expected = contract(self.profiles[model])
        self.guard()
        self.ready_model = None
        self.emit('加载 ' + model + '；完成后等待 start，不自动运行机器人。')
        result = self.remote.load(model, expected_contract=expected)
        if result.get('contract') != expected or result.get('state') != 'ready':
            raise RuntimeError('服务身份/就绪状态不匹配，禁止开始。')
        self.ready_model = model
        self.emit(model + ' 已就绪。恢复场景后输入 start。')
        return result

    def start(self):
        self._idle()
        if self._pending_score():
            raise RuntimeError('请先评分。')
        model = self.session.current_model
        if not model or self.ready_model != model:
            raise RuntimeError('请先 load 并等待模型就绪。')
        profile = self.profiles[model]
        from model_catalog import contract
        expected = contract(profile)
        status = self.remote.status()
        if status.get('state') != 'ready' or status.get('contract') != expected:
            self.ready_model = None
            raise RuntimeError('模型已退出或身份改变，请重新 load。')
        self.guard(cameras=True)
        if self.trial is None:
            from evaluation_trial import run_trial
            self.trial = run_trial
        attempt = self.session.begin_attempt(instruction=profile['instruction'], contract=expected)
        self.stop_reason = None
        try:
            future = self.pool.submit(self.trial, profile, url=profile['bind'],
                                      steps=attempt['steps'], instruction=attempt['instruction'],
                                      output=Path(attempt['path']),
                                      stop_requested=lambda: self.stop_reason)
        except BaseException:
            self.session.finish_attempt(attempt['id'], {
                'end_reason': 'error', 'recording_complete': False,
                'steps_sent': 0, 'error': 'Trial worker could not start'})
            raise
        self.active = (attempt, future)
        self.emit('开始 ' + attempt['id'] + '；finish=正常提前结束并评分，pause=中断不计次。')
        return attempt

    def stop_trial(self, reason):
        if reason not in ('early_finish', 'paused'):
            raise ValueError('Invalid stop reason')
        self.poll()
        if self.active is None:
            raise RuntimeError('当前没有运行中的测试。')
        # An interruption cannot later be relabelled as a normal finish.
        if self.stop_reason != 'paused':
            self.stop_reason = reason
        self.emit('已请求停止发送后续动作，等待当前调用返回并收尾；这不是硬件急停。')

    def poll(self):
        if self.active is None or not self.active[1].done():
            return
        attempt, future = self.active
        try:
            result = future.result()
        except BaseException as error:
            result = {'end_reason': 'error', 'recording_complete': False,
                      'steps_sent': 0, 'error': f'{type(error).__name__}: {error}'}
        self.session.finish_attempt(attempt['id'], result)
        self.active = None
        self.emit(f"本次结束: {result['end_reason']}，已发送 {result.get('steps_sent', 0)} 步。")
        if result['end_reason'] == 'time_limit':
            seconds = result.get('max_action_seconds', 30)
            self.emit(f'已到{seconds:g}秒上限并停止；请人工判断成功/失败并评分，导出成功后计次。')
        if result.get('error'):
            self.emit('错误: ' + result['error'])
        if self._pending_score():
            self.emit('输入 score 分数 success或failed 可选备注；结果由你判断，评分后导出，导出成功后计次。')
        else:
            self.emit('本次未计入目标次数，原始记录保留；排除原因、恢复场景后再 start。')

    def score(self, score, note='', *, outcome=None):
        self._idle()
        attempt = self._pending_score()
        if attempt is None:
            raise RuntimeError('没有待评分的正常试验。')
        self.session.score_attempt(attempt['id'], score, note, outcome=outcome)
        self.emit('评分已保存；开始导出三路 LeRobot，导出成功后才计次。')
        return self.export(attempt['id'])

    def export(self, attempt_id):
        self._idle()
        attempt = next(a for a in self.session.state['attempts'] if a['id'] == attempt_id)
        if not export_eligible(attempt):
            raise RuntimeError('中断或未评分试验不自动导出。')
        if attempt['export']['status'] == 'succeeded':
            self.emit('已经导出: ' + attempt['export']['path'])
            return Path(attempt['export']['path'])
        self.session.mark_export(attempt_id, 'in_progress')
        metadata = {'session_id': self.session.state['id'], 'task_id': self.session.state['task'],
                    'mode': self.session.state['mode'],
                    'model_id': attempt['model_id'], 'attempt_id': attempt_id,
                    'contract': attempt['contract'], 'instruction': attempt['instruction'],
                    'score': attempt['score'], 'note': attempt['note'], 'success': attempt['success'],
                    'end_reason': attempt['end_reason'], 'step_limit': attempt['steps'],
                    'steps_sent': attempt['steps_sent']}
        if attempt.get('outcome_source') == 'manual':
            metadata.update(outcome=attempt['outcome'], outcome_source='manual')
        elif attempt['end_reason'] == 'time_limit':
            metadata['outcome'] = 'failed'
        try:
            if self.exporter is None:
                from evaluation_trial import export_recording
                self.exporter = export_recording
            path = self.exporter(self.profiles[attempt['model_id']],
                                 attempt_path=Path(attempt['path']), metadata=metadata)
        except Exception as error:
            self.session.mark_export(attempt_id, 'failed', error=f'{type(error).__name__}: {error}')
            self.emit(f'导出失败，本次未计次，评分及原始记录保留。请排查后重试：retry-export {attempt_id}\n{error}')
            return None
        self.session.mark_export(attempt_id, 'succeeded', path=str(path))
        self.emit('LeRobot 已保存并计次: ' + str(path))
        return path

    def next_model(self):
        self._idle()
        self.session.advance_model()
        self.ready_model = None
        if self.session.current_model is None:
            self.remote.stop()
            self.emit('本任务全部模型已完成。')
            return
        return self.load()

    def configure(self, kind, value, model=None):
        self._idle()
        if kind == 'steps':
            return self.session.set_steps(value)
        return self.session.set_repeats(value, model_id=model)

    def close(self):
        try:
            if self.active is not None:
                self.stop_reason = 'paused'
                self.emit('等待试验停止并保存中断记录…')
                try:
                    self.active[1].result()
                except BaseException:
                    pass  # poll persists the failure before cleanup.
                self.poll()
        finally:
            try:
                self.pool.shutdown(wait=True)
            finally:
                self.remote.close()


HELP = '''命令：
  status                  查看每款进度、保存状态
  load                    加载当前模型，等待就绪（不启动设备）
  start                   确认场景已恢复，开始本次相机/机械臂测试
  finish                  主动提前结束，随后评分、导出，成功导出后计次
  pause                   中断本次；保留原始记录，不计次、不自动导出
  score 80 success 备注   人工标记成功并评分、导出（备注可省略）
  score 20 failed 备注    人工标记失败并评分、导出；0分也在导出成功后计次
  next                    当前模型凑满目标后加载下一模型，仍需 start
  repeats 25 [模型ID]     修改当前/后续目标次数，不低于已完成数
  steps 800               修改下一次步数上限
  retry-export 试验ID     只重试数据导出，不运行机器人
  quit                    退出并停止本程序加载的模型服务
  help                    显示帮助
首条模型关节命令起计时，时长由 max_action_seconds 配置；成功/失败与分数由操作者分别决定。
Ctrl+C 在运行中按中断处理，不能替代实体急停。'''


def interactive(controller, *, input_stream=None):
    stream = input_stream or sys.stdin
    # Read terminal input on this thread only. A background stdin reader would
    # race OpenSSH's /dev/tty password prompt during load(). Robot work already
    # runs independently; polling the terminal lets us announce completion.
    pending = ''
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError):
        descriptor = None  # In-memory streams used by offline interaction tests.
    controller.emit(HELP)
    controller.status()
    while True:
        try:
            controller.poll()
            if descriptor is None:
                line = stream.readline()
                if not line:
                    return
            else:
                import select
                if '\n' not in pending:
                    readable, _, _ = select.select([descriptor], [], [], .1)
                    if not readable:
                        continue
                    raw = os.read(descriptor, 4096)
                    if not raw:
                        return  # close() preserves a running trial as interrupted.
                    pending += raw.decode('utf-8', errors='replace')
                if '\n' not in pending:
                    continue
                line, pending = pending.split('\n', 1)
            line = line.strip()
            if not line:
                continue
            args = shlex.split(line)
            command, tail = args[0], args[1:]
            if command == 'help':
                controller.emit(HELP)
            elif command == 'status':
                controller.status()
            elif command == 'load':
                controller.load()
            elif command == 'start':
                controller.start()
            elif command in ('finish', 'pause'):
                controller.stop_trial('early_finish' if command == 'finish' else 'paused')
            elif command == 'score' and tail:
                if len(tail) < 2 or tail[1] not in ('success', 'failed'):
                    raise ValueError('请显式选择结果，例如 score 80 success 备注 或 score 20 failed 备注。')
                controller.score(float(tail[0]), ' '.join(tail[2:]), outcome=tail[1])
            elif command == 'next':
                controller.next_model()
            elif command in ('steps', 'repeats') and tail:
                controller.configure(command, int(tail[0]), tail[1] if len(tail) > 1 else None)
            elif command == 'retry-export' and len(tail) == 1:
                controller.export(tail[0])
            elif command == 'quit':
                controller._idle()
                return
            else:
                raise ValueError('未知命令或参数不全；输入 help。')
        except KeyboardInterrupt:
            if controller.active:
                controller.stop_trial('paused')
            else:
                return
        except Exception as error:
            controller.emit(f'未继续执行: {type(error).__name__}: {error}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--task', help='任意任务标识，不决定模型或控制参数')
    parser.add_argument('--models', nargs='+', help='按此顺序评测的模型配置ID')
    parser.add_argument('--repeats', type=int, default=25)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--session', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--ai-host', help='模型机 SSH 用户@主机')
    parser.add_argument('--ai-root', help='模型机上本仓库的绝对路径')
    parser.add_argument('--ai-catalog', help='模型机上同一份配置的绝对路径')
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--plan', action='store_true', help='只显示配置，不联网或启动设备')
    operation.add_argument('--run', action='store_true', help='进入交互模式；load/start 分别启动模型/设备')
    args = parser.parse_args(argv)
    from model_catalog import contract, read_catalog, validate_runtime_profile
    catalog = read_catalog(args.catalog)
    if args.repeats <= 0 or args.steps <= 0:
        parser.error('--repeats 和 --steps 必须大于0')
    if args.resume and args.session:
        parser.error('--resume 与 --session 不能同时使用')
    if args.resume and args.models:
        parser.error('恢复时使用存档中的模型顺序，不指定 --models')
    task = args.task
    if args.resume:
        saved = json.loads((args.resume / 'session.json').read_text(encoding='utf-8'))
        if saved.get('mode') != 'live':
            parser.error('模拟会话不能作为实机评测恢复')
        if task and task != saved['task']:
            parser.error('恢复会话时不能更换任务')
        task = saved['task']
        profiles = resume_profiles(catalog, args.resume, saved)
    else:
        if not task or not args.models:
            parser.error('新会话必须指定 --task 和 --models')
        profiles = task_profiles(catalog, task, args.models)
    if not args.run:
        targets = saved['targets'] if args.resume else {p['model_id']: args.repeats for p in profiles}
        print(json.dumps({'task': task, 'target_counts': targets,
                          'total_counted_trials': sum(targets.values()),
                          'steps': saved['steps'] if args.resume else args.steps,
                          'models': [{'contract': contract(p), 'instruction': p['instruction'],
                                      'max_action_seconds': action_seconds_for_profile(p)} for p in profiles],
                          'started': False, 'recording_cameras': 3}, ensure_ascii=False, indent=2))
        return
    if not all((args.ai_host, args.ai_root, args.ai_catalog)):
        parser.error('--run 需要 --ai-host、--ai-root、--ai-catalog')
    for profile in profiles:
        validate_runtime_profile(profile)
    from pathlib import PurePosixPath
    from evaluation_session import Session
    from evaluation_remote import RemoteModels
    ai_runtime = PurePosixPath(args.ai_root) / 'runtime'
    remote = RemoteModels(suite_root=str(ai_runtime), host_script=str(ai_runtime / 'evaluation_model_host.py'),
                          host=args.ai_host, catalog=args.ai_catalog)
    client_idle()
    root = args.resume or args.session or Path('runs') / uuid.uuid4().hex
    if args.resume:
        session = Session.open(root)
    else:
        session = Session.create(root, task, [p['model_id'] for p in profiles],
                                 repeats=args.repeats, steps=args.steps)
        (root / 'profiles.json').write_text(json.dumps({'profiles': profiles}, ensure_ascii=False, indent=2), encoding='utf-8')
    controller = Evaluation(session, profiles, remote)
    print('会话目录: ' + str(root))
    try:
        interactive(controller)
    finally:
        try:
            controller.close()
        finally:
            session.close()


if __name__ == '__main__':
    main()
