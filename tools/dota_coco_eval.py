import argparse
import os

import mmcv
from mmcv import Config, DictAction

from mmrotate.core import eval_rbbox_map
from mmrotate.datasets import build_dataset
from mmrotate.datasets.dota import DOTADataset
from mmrotate.utils import compat_cfg


IOU_THRS = [round(x / 100, 2) for x in range(50, 100, 5)]


def evaluate_dota_coco_ap(dataset,
                          results,
                          scale_ranges=None,
                          logger=None,
                          nproc=4):
    if not isinstance(dataset, DOTADataset):
        raise TypeError('`--eval-dota-coco-ap` only supports DOTADataset.')

    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    nproc = min(nproc, os.cpu_count())

    ap_results = {}
    for thr in IOU_THRS:
        mean_ap, _ = eval_rbbox_map(
            results,
            annotations,
            scale_ranges=scale_ranges,
            iou_thr=thr,
            dataset=dataset.CLASSES,
            logger=logger,
            nproc=nproc)
        ap_results[thr] = float(mean_ap)

    return {
        'AP50': ap_results[0.50],
        'AP75': ap_results[0.75],
        'AP': sum(ap_results.values()) / len(ap_results),
        'mAP': ap_results[0.50]
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate DOTA predictions with COCO-style AP metrics')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('result', help='result pickle file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--nproc',
        type=int,
        default=4,
        help='processes used for computing TP and FP')
    return parser.parse_args()


def _set_test_mode(test_cfg):
    if isinstance(test_cfg, dict):
        test_cfg.test_mode = True
    elif isinstance(test_cfg, list):
        for ds_cfg in test_cfg:
            ds_cfg.test_mode = True


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)

    _set_test_mode(cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)

    metric = evaluate_dota_coco_ap(dataset, results, nproc=args.nproc)
    print(metric)


if __name__ == '__main__':
    main()