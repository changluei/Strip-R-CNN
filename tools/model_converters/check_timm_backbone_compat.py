import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmrotate.models.backbones import StripNet, StripSMambaNet  # noqa: E402
from tools.model_converters.stripnet_timm_to_mmrotate import convert_state_dict  # noqa: E402


VARIANT_SPECS = {
    'stripnet_t': dict(
        timm_name='stripnet_t',
        mmrotate_class=StripNet,
        kwargs=dict(
            embed_dims=[32, 64, 160, 256],
            depths=[3, 3, 5, 2],
            mlp_ratios=[8, 8, 4, 4],
            k1s=[1, 1, 1, 1],
            k2s=[19, 19, 19, 19],
            norm_cfg=None,
        ),
    ),
    'stripnet_s': dict(
        timm_name='stripnet_s',
        mmrotate_class=StripNet,
        kwargs=dict(
            embed_dims=[64, 128, 320, 512],
            depths=[2, 2, 4, 2],
            mlp_ratios=[8, 8, 4, 4],
            k1s=[1, 1, 1, 1],
            k2s=[19, 19, 19, 19],
            norm_cfg=None,
        ),
    ),
    'stripnet_uni_cross_mamba_t': dict(
        timm_name='stripnet_uni_cross_mamba_t',
        mmrotate_class=StripSMambaNet,
        kwargs=dict(
            embed_dims=[32, 64, 160, 256],
            depths=[3, 3, 5, 2],
            mlp_ratios=[8, 8, 4, 4],
            k1s=[1, 1, 1, 1],
            k2s=[19, 19, 19, 19],
            norm_cfg=None,
        ),
    ),
    'stripnet_uni_cross_mamba_s': dict(
        timm_name='stripnet_uni_cross_mamba_s',
        mmrotate_class=StripSMambaNet,
        kwargs=dict(
            embed_dims=[64, 128, 320, 512],
            depths=[2, 2, 4, 2],
            mlp_ratios=[8, 8, 4, 4],
            k1s=[1, 1, 1, 1],
            k2s=[19, 19, 19, 19],
            norm_cfg=None,
        ),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check whether a timm StripNet-family model matches the MMRotate backbone state_dict.'
    )
    parser.add_argument(
        '--variant',
        required=True,
        choices=sorted(VARIANT_SPECS.keys()),
        help='Model variant to compare.',
    )
    parser.add_argument(
        '--timm-root',
        required=True,
        help='Path to the timm repository containing the custom model file.',
    )
    return parser.parse_args()


def compare_state_dicts(timm_state, mmrotate_state):
    timm_keys = set(timm_state.keys())
    mmrotate_keys = set(mmrotate_state.keys())

    missing_in_timm = sorted(mmrotate_keys - timm_keys)
    extra_in_timm = sorted(timm_keys - mmrotate_keys)
    mismatched_shapes = []

    for key in sorted(timm_keys & mmrotate_keys):
        if timm_state[key].shape != mmrotate_state[key].shape:
            mismatched_shapes.append(
                (key, tuple(timm_state[key].shape), tuple(mmrotate_state[key].shape))
            )

    return missing_in_timm, extra_in_timm, mismatched_shapes


def main():
    args = parse_args()
    timm_root = Path(args.timm_root).resolve()
    if not timm_root.is_dir():
        raise NotADirectoryError(f'timm root does not exist: {timm_root}')

    if str(timm_root) not in sys.path:
        sys.path.insert(0, str(timm_root))

    import timm  # noqa: WPS433

    spec = VARIANT_SPECS[args.variant]
    timm_model = timm.create_model(spec['timm_name'], pretrained=False)
    timm_state, dropped = convert_state_dict(timm_model.state_dict(), keep_head=False)

    mmrotate_model = spec['mmrotate_class'](**spec['kwargs'])
    mmrotate_state = mmrotate_model.state_dict()

    missing_in_timm, extra_in_timm, mismatched_shapes = compare_state_dicts(
        timm_state,
        mmrotate_state,
    )

    print(f'Variant: {args.variant}')
    print(f'Dropped timm keys: {len(dropped)}')
    print(f'MMRotate keys: {len(mmrotate_state)}')
    print(f'timm backbone keys: {len(timm_state)}')
    print(f'Missing in timm: {len(missing_in_timm)}')
    print(f'Extra in timm: {len(extra_in_timm)}')
    print(f'Shape mismatches: {len(mismatched_shapes)}')

    if missing_in_timm:
        print('First missing keys:')
        for key in missing_in_timm[:10]:
            print(f'  {key}')

    if extra_in_timm:
        print('First extra keys:')
        for key in extra_in_timm[:10]:
            print(f'  {key}')

    if mismatched_shapes:
        print('First shape mismatches:')
        for key, timm_shape, mmrotate_shape in mismatched_shapes[:10]:
            print(f'  {key}: timm={timm_shape}, mmrotate={mmrotate_shape}')

    if missing_in_timm or extra_in_timm or mismatched_shapes:
        raise SystemExit(1)

    print('State_dict compatibility check passed.')


if __name__ == '__main__':
    main()
