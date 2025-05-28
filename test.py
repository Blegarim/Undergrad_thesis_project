import pickle
from pathlib import Path

db_path = Path("data/data_cache/pie_database.pkl")  # Adjust path if needed
with open(db_path, "rb") as f:
    db = pickle.load(f)

print("✅ Annotation DB loaded")
print("Available sets:", list(db.keys()))

for set_id in db:
    print(f"\nSet: {set_id}")
    for video_id in db[set_id]:
        print(f"  Video: {video_id}")
        ped_annot = db[set_id][video_id].get("ped_annotations", {})
        for ped_id, ped_data in ped_annot.items():
            print(f"    Pedestrian ID: {ped_id}")
            frames = ped_data.get("frames", {})
            print(f"      Num frames: {len(frames)}")

        break  # just show one video for now
    break  # just show one set for now
