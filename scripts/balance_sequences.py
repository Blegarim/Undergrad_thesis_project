import argparse
import os
import pickle
import random
from collections import Counter


def _init_groups():
    return {
        0: {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []},
        1: {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []},
    }


def _group_indices(data):
    groups = _init_groups()
    for idx, item in enumerate(data):
        a = int(item["actions"])
        l = int(item["looks"])
        c = int(item["crosses"])
        if c not in (0, 1):
            c = 0
        groups[c][(a, l)].append(idx)
    return groups


def _solve_cross0_counts(n1, a1_pos, l1_pos, c00, c01, c10, c11):
    n0 = n1
    a0_target = n1 - a1_pos
    l0_target = n1 - l1_pos

    lower = max(0, a0_target + l0_target - n0, a0_target - c10, l0_target - c01)
    upper = min(a0_target, l0_target, c11, c00 + a0_target + l0_target - n0)
    if lower > upper:
        return None

    x11 = upper
    x10 = a0_target - x11
    x01 = l0_target - x11
    x00 = n0 - x10 - x01 - x11
    return x00, x01, x10, x11


def _counts_for_indices(data, indices):
    cnt = Counter()
    for idx in indices:
        item = data[idx]
        cnt["actions"] += int(item["actions"])
        cnt["looks"] += int(item["looks"])
        cnt["crosses"] += int(item["crosses"])
    return cnt


def balance_dataset(data, seed=0):
    rand = random.Random(seed)
    groups = _group_indices(data)

    cross1_indices = []
    for combo in groups[1].values():
        cross1_indices.extend(combo)

    n1 = len(cross1_indices)
    if n1 == 0:
        return []

    a1_pos = len(groups[1][(1, 0)]) + len(groups[1][(1, 1)])
    l1_pos = len(groups[1][(0, 1)]) + len(groups[1][(1, 1)])

    c00 = len(groups[0][(0, 0)])
    c01 = len(groups[0][(0, 1)])
    c10 = len(groups[0][(1, 0)])
    c11 = len(groups[0][(1, 1)])

    solution = _solve_cross0_counts(n1, a1_pos, l1_pos, c00, c01, c10, c11)
    if solution is None:
        raise RuntimeError("No feasible balanced subset found with current data.")

    x00, x01, x10, x11 = solution

    def pick(indices, k):
        if k == 0:
            return []
        if k > len(indices):
            raise RuntimeError("Insufficient samples for requested balance.")
        return indices if k == len(indices) else rand.sample(indices, k)

    cross0_indices = []
    cross0_indices += pick(groups[0][(0, 0)], x00)
    cross0_indices += pick(groups[0][(0, 1)], x01)
    cross0_indices += pick(groups[0][(1, 0)], x10)
    cross0_indices += pick(groups[0][(1, 1)], x11)

    selected = cross1_indices + cross0_indices
    rand.shuffle(selected)
    return selected


def summarize(data, indices):
    total = len(indices)
    cnt = _counts_for_indices(data, indices)
    if total == 0:
        return {}
    return {
        "total": total,
        "actions_pos_rate": cnt["actions"] / total,
        "looks_pos_rate": cnt["looks"] / total,
        "crosses_pos_rate": cnt["crosses"] / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inputs", nargs="+", default=[
        "sequences_train.pkl",
        "sequences_val.pkl",
        "sequences_test.pkl",
    ])
    parser.add_argument("--suffix", default="_balanced")
    args = parser.parse_args()

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"{path}: MISSING")
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        indices = balance_dataset(data, seed=args.seed)
        summary = summarize(data, indices)

        out_path = os.path.splitext(path)[0] + args.suffix + ".pkl"
        subset = [data[i] for i in indices]
        with open(out_path, "wb") as f:
            pickle.dump(subset, f)

        print(f"{path} -> {out_path}")
        print(
            "  total={total} | actions_pos={actions_pos_rate:.4f} | "
            "looks_pos={looks_pos_rate:.4f} | crosses_pos={crosses_pos_rate:.4f}".format(**summary)
        )


if __name__ == "__main__":
    main()
