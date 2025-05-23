import sys
from pathlib import Path
import collections

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from PIE.utilities.pie_data import PIE

data_root = ROOT_DIR / 'data'
pie = PIE(data_path=str(data_root))  # Loads PIE dataset

action_counter = collections.Counter()

video_list = pie.get_all_video_names()

for video in video_list:
    anns = pie.get_annotations(video)
    for pid, ped_data in anns['ped_annotations'].items():
        actions = ped_data.get('actions', [])
        action_counter.update(actions)

print("Action label counts:")
print(action_counter)
