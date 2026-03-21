from pathlib import Path

# 你的提交txt目录
res_dir = Path('dotav2_orign_Task1_results')
# 官方DOTA-v2原始test图片目录（不是split后的）
official_test_dir = Path('/path/to/DOTA-v2/test/images')

pred_ids = set()
for f in res_dir.glob('Task1_*.txt'):
    for line in f.open('r', encoding='utf-8', errors='replace'):
        s = line.strip()
        if s:
            pred_ids.add(s.split()[0])

official_ids = {p.stem for p in official_test_dir.glob('*.png')}
extra = pred_ids - official_ids

print('pred_ids:', len(pred_ids))
print('official_ids:', len(official_ids))
print('ids_not_in_official_test:', len(extra))
print('sample_extra:', sorted(list(extra))[:20])
