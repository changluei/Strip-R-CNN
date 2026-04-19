import argparse
from collections import OrderedDict
from pathlib import Path

import torch


AUTO_SOURCE_KEYS = ('state_dict_ema', 'model_ema', 'state_dict', 'model')
DROP_PREFIXES = ('head.',)
STRIP_PREFIXES = ('module.', 'model.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a timm StripNet-family training checkpoint to an MMRotate-compatible backbone checkpoint.')
    parser.add_argument('src', help='Path to the timm training checkpoint.')
    parser.add_argument('dst', help='Path to save the converted checkpoint.')
    parser.add_argument(
        '--source-key',
        default='auto',
        help='Checkpoint key to read. Use auto, or one of: state_dict_ema, model_ema, state_dict, model.')
    parser.add_argument(
        '--keep-head',
        action='store_true',
        help='Keep classifier head weights instead of dropping them.')
    return parser.parse_args()


def select_state_dict(checkpoint, source_key):
    if not isinstance(checkpoint, dict):
        return checkpoint, None

    if source_key != 'auto':
        if source_key not in checkpoint:
            raise KeyError(f'Source key "{source_key}" was not found in checkpoint.')
        return checkpoint[source_key], source_key

    for key in AUTO_SOURCE_KEYS:
        if key in checkpoint:
            return checkpoint[key], key

    # If it already looks like a plain state_dict, use it directly.
    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint, None

    raise KeyError(
        'Could not find a usable state_dict in checkpoint. '
        f'Tried keys: {", ".join(AUTO_SOURCE_KEYS)}')


def strip_known_prefixes(key):
    changed = True
    while changed:
        changed = False
        for prefix in STRIP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def should_drop(key, keep_head):
    if keep_head:
        return False
    return any(key.startswith(prefix) for prefix in DROP_PREFIXES)


def convert_state_dict(state_dict, keep_head=False):
    converted = OrderedDict()
    dropped = []

    for key, value in state_dict.items():
        new_key = strip_known_prefixes(key)
        if should_drop(new_key, keep_head):
            dropped.append(new_key)
            continue
        converted[new_key] = value

    return converted, dropped


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    checkpoint = torch.load(src, map_location='cpu')
    state_dict, used_key = select_state_dict(checkpoint, args.source_key)
    converted, dropped = convert_state_dict(state_dict, keep_head=args.keep_head)

    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'state_dict': converted,
        'meta': {
            'source': str(src),
            'source_key': used_key or 'plain_state_dict',
            'keep_head': args.keep_head,
            'num_tensors': len(converted),
            'dropped_keys': dropped,
        }
    }
    torch.save(payload, dst)

    print(f'Saved converted checkpoint to: {dst}')
    print(f'Loaded weights from key: {used_key or "plain_state_dict"}')
    print(f'Kept tensors: {len(converted)}')
    print(f'Dropped tensors: {len(dropped)}')
    if dropped:
        print('Dropped key prefixes: head.')


if __name__ == '__main__':
    main()
