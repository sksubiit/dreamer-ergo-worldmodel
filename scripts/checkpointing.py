"""Local checkpoint I/O only; no learner, replay-sampling or optimizer calls."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

METADATA = ('config.yaml', 'reach_task.json', 'servo_settings.json', 'manifest.json',
            'dreamer.py', 'ergocub_env.py', 'ergocub_physics.py', 'checkpointing.py',
            'prior_prediction.py', 'runtime_environment.json')


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def file_hashes(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): sha256(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def verify_prepared(args, config):
    """Reject stale/changed inputs before a prepared directory can be launched."""
    directory = args.logdir.resolve()
    record = json.loads((directory/'resume_input.json').read_text())
    if args.checkpoint.resolve() != directory/'resume.ckpt':
        raise ValueError('Prepared run must load its verified resume.ckpt copy')
    if list(args.milestones) != record['milestones']:
        raise ValueError('Milestones differ from the prepared continuation')
    if args.profile != record['profile']:
        raise ValueError('Profile differs from the prepared continuation')
    for name, expected in record['input_hashes'].items():
        if sha256(directory/name) != expected:
            raise ValueError(f'Prepared input changed: {name}')
    from ruamel.yaml import YAML
    import embodied
    saved = embodied.Config(YAML(typ='safe').load((directory/'config.yaml').read_text()))
    if config.flat != saved.flat:
        raise ValueError('Resolved configuration differs from prepared config.yaml')
    for name in ('metrics.jsonl', 'scores.jsonl', 'checkpoint.ckpt', 'final.ckpt'):
        if (directory/name).exists():
            raise ValueError(f'Prepared directory has already been launched: {name}')
    for name in ('train_replay', 'eval_replay'):
        actual = file_hashes(directory/name)
        expected = {str(Path(p).relative_to(name)): h for p,h in record['input_hashes'].items()
                    if p.startswith(name+'/')}
        if actual != expected:
            raise ValueError(f'Prepared replay changed: {name}')
    for name in ('dreamer.py', 'ergocub_env.py', 'ergocub_physics.py', 'checkpointing.py'):
        if sha256(Path(__file__).parent/name) != record['input_hashes'][name]:
            raise ValueError(f'Local source changed since preparation: {name}')
    if 'prior_prediction.py' in record['input_hashes']:
        if sha256(Path(__file__).parent/'prior_prediction.py') != record['input_hashes']['prior_prediction.py']:
            raise ValueError('Prior-prediction implementation changed since preparation')


def save_milestone(embodied, holder, directory, step):
    """Save a complete, immutable resume bundle while collection is idle."""
    directory = Path(directory)
    destination = directory/'milestones'/f'step_{step:06d}'
    temporary = destination.with_name(destination.name+'.tmp')
    temporary.mkdir(parents=True, exist_ok=False)
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = embodied.Checkpoint(str(temporary/'final.ckpt'), parallel=False)
    checkpoint.step = holder['logger'].step
    checkpoint.agent = holder['agent']
    for name in ('train_replay', 'eval_replay'):
        setattr(checkpoint, name, holder[name])
    checkpoint.save()  # Local factories use replay.save(wait=True).
    for name in METADATA:
        source = directory/name
        if source.is_file():
            shutil.copy2(source, temporary/name)
    for name in ('train_replay', 'eval_replay'):
        shutil.copytree(directory/name, temporary/name)
    (temporary/'snapshot.json').write_text(json.dumps(dict(
        driver_step=step, checkpoint='final.ckpt', includes_optimizer_state=True,
        replay_hashes={name: file_hashes(temporary/name) for name in ('train_replay','eval_replay')}
    ), indent=2)+'\n')
    temporary.rename(destination)
    print(f'Retained milestone at {step} total driver steps: {destination}', flush=True)


@contextmanager
def retain_milestones(embodied, holder, directory, milestones):
    """Observe completed Driver calls; leave upstream source and updates unchanged.

    train_eval advances the training counter only in its training Driver. The
    hook runs after that Driver's callbacks (including optimizer calls) finish.
    Evaluation Driver calls do not advance the training counter and are ignored.
    """
    original = embodied.Driver
    pending = set(milestones)
    class SnapshotDriver(original):
        def __call__(self, *args, **kwargs):
            before = int(holder['logger'].step)
            result = super().__call__(*args, **kwargs)
            after = int(holder['logger'].step)
            for target in sorted(pending):
                if before < target <= after:
                    if target != after:
                        raise ValueError('Milestone must coincide with a collection chunk boundary')
                    save_milestone(embodied, holder, directory, target)
                    pending.remove(target)
            return result
    embodied.Driver = SnapshotDriver
    try:
        yield
    finally:
        embodied.Driver = original
