import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
import run_training


def _flatten_cfg(cfg: DictConfig) -> dict:
    """Flatten nested Hydra cfg into a flat dict matching run_training.py expected arg names."""
    c = OmegaConf.to_container(cfg, resolve=True)
    flat = {}
    # general
    gen = c.get('general', {}) if 'general' in c else c
    flat.update({k: v for k, v in gen.items() if k in ['project_name', 'experiment_name', 'num_workers', 'precision', 'evaluate', 'ckpt_id', 'orig_sample_rate', 'subset']})

    # training
    train = c.get('training', {})
    flat.update({k: v for k, v in train.items() if k in ['n_epochs', 'batch_size', 'mixstyle_p', 'mixstyle_alpha', 'weight_decay', 'roll_sec', 'lr', 'warmup_steps', 'num_workers']})

    # preprocessing
    pre = c.get('preprocessing', {})
    flat.update({k: v for k, v in pre.items() if k in ['sample_rate', 'window_length', 'hop_length', 'n_fft', 'n_mels', 'freqm', 'timem', 'f_min', 'f_max']})

    # model
    import os
    import yaml
    model = c.get('model', {})
    flat.update({k: v for k, v in model.items() if k in ['n_classes', 'in_channels', 'base_channels', 'channels_multiplier', 'expansion_rate', 'n_blocks', 'strides']})

    # student/teacher yaml 경로 읽어서 dict로 로드
    student_cfg_path = model.get('student_cfg')
    teacher_cfg_path = model.get('teacher_cfg')
    if student_cfg_path:
        with open(os.path.join('configs', student_cfg_path), 'r', encoding='utf-8') as f:
            flat['student_cfg'] = yaml.safe_load(f)
    if teacher_cfg_path:
        with open(os.path.join('configs', teacher_cfg_path), 'r', encoding='utf-8') as f:
            flat['teacher_cfg'] = yaml.safe_load(f)

    # repconv / kd / prune: keep whole dicts
    flat['repconv'] = c.get('repconv', {})
    flat['kd'] = c.get('kd', {})
    flat['prune'] = c.get('prune', {})

    # allow top-level fallback values
    for k in ['project_name', 'experiment_name']:
        if k not in flat and k in c:
            flat[k] = c[k]

    return flat


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    flat = _flatten_cfg(cfg)
    args = SimpleNamespace(**flat)

    if getattr(args, 'evaluate', False):
        run_training.evaluate(args)
    else:
        run_training.train(args)


if __name__ == '__main__':
    pass  # hydra.main이 main()을 자동 호출함
