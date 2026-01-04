import argparse
import os
import pickle
import random
from collections import Counter


def clamp_cross(value):
    return 1 if value == 1 else 0


def split_indices(n, train_pct, val_pct, seed):
    rand = random.Random(seed)
    indices = list(range(n))
    rand.shuffle(indices)
    train_n = int(n * train_pct)
    val_n = int(n * val_pct)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:train_n + val_n]
    test_idx = indices[train_n + val_n:]
    return train_idx, val_idx, test_idx


def group_indices(data, indices):
    groups = {
        0: {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []},
        1: {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []},
    }
    for idx in indices:
        item = data[idx]
        a = int(item["actions"])
        l = int(item["looks"])
        c = clamp_cross(int(item["crosses"]))
        groups[c][(a, l)].append(idx)
    return groups


def choose_cross1(groups, n1, rand):
    selected = []
    priority = [(0, 0), (0, 1), (1, 0), (1, 1)]
    remaining = n1
    for combo in priority:
        pool = groups[1][combo]
        if remaining <= 0:
            break
        if len(pool) <= remaining:
            selected.extend(pool)
            remaining -= len(pool)
        else:
            selected.extend(rand.sample(pool, remaining))
            remaining = 0
    return selected


def solve_exact(n0, a_target, l_target, c00, c01, c10, c11):
    lower = max(0, a_target - c10, l_target - c01, n0 - a_target - l_target)
    upper = min(c11, a_target, l_target, n0 - a_target - l_target + c00)
    if lower > upper:
        return None
    x11 = lower
    x10 = a_target - x11
    x01 = l_target - x11
    x00 = n0 - x11 - x10 - x01
    return x00, x01, x10, x11


def solve_approx(n0, a_target, l_target, c00, c01, c10, c11):
    best = None
    best_error = None
    max_x11 = min(c11, n0)
    for x11 in range(max_x11 + 1):
        remaining = n0 - x11
        x10_min = max(0, remaining - (c01 + c00))
        x10_max = min(c10, remaining)
        if x10_min > x10_max:
            continue
        target_x10 = a_target - x11
        x10 = min(max(target_x10, x10_min), x10_max)

        remaining2 = remaining - x10
        x01_min = max(0, remaining2 - c00)
        x01_max = min(c01, remaining2)
        if x01_min > x01_max:
            continue
        target_x01 = l_target - x11
        x01 = min(max(target_x01, x01_min), x01_max)
        x00 = remaining2 - x01

        action_count = x10 + x11
        look_count = x01 + x11
        error = abs(action_count - a_target) + abs(look_count - l_target)
        if best_error is None or error < best_error:
            best_error = error
            best = (x00, x01, x10, x11)
            if error == 0:
                break
    return best


def balance_split(data, indices, cross_pos_ratio, seed):
    rand = random.Random(seed)
    groups = group_indices(data, indices)

    c1_total = sum(len(v) for v in groups[1].values())
    c0_total = sum(len(v) for v in groups[0].values())
    if c1_total == 0 or c0_total == 0:
        return []

    n1 = min(c1_total, int(c0_total * cross_pos_ratio / (1.0 - cross_pos_ratio)))
    n0 = int(round(n1 * (1.0 - cross_pos_ratio) / cross_pos_ratio))
    if n0 > c0_total:
        n0 = c0_total
        n1 = int(round(n0 * cross_pos_ratio / (1.0 - cross_pos_ratio)))

    selected1 = choose_cross1(groups, n1, rand)
    a1 = sum(int(data[i]["actions"]) for i in selected1)
    l1 = sum(int(data[i]["looks"]) for i in selected1)

    total_n = n0 + n1
    a_target_total = int(round(0.5 * total_n))
    l_target_total = int(round(0.5 * total_n))
    a_target0 = max(0, min(n0, a_target_total - a1))
    l_target0 = max(0, min(n0, l_target_total - l1))

    c00 = len(groups[0][(0, 0)])
    c01 = len(groups[0][(0, 1)])
    c10 = len(groups[0][(1, 0)])
    c11 = len(groups[0][(1, 1)])

    solved = solve_exact(n0, a_target0, l_target0, c00, c01, c10, c11)
    if solved is None:
        solved = solve_approx(n0, a_target0, l_target0, c00, c01, c10, c11)
    if solved is None:
        return []

    x00, x01, x10, x11 = solved

    def pick(pool, k):
        if k <= 0:
            return []
        if k >= len(pool):
            return list(pool)
        return rand.sample(pool, k)

    selected0 = []
    selected0.extend(pick(groups[0][(0, 0)], x00))
    selected0.extend(pick(groups[0][(0, 1)], x01))
    selected0.extend(pick(groups[0][(1, 0)], x10))
    selected0.extend(pick(groups[0][(1, 1)], x11))

    selected = selected1 + selected0
    rand.shuffle(selected)
    return selected


def summarize(data, indices):
    total = len(indices)
    if total == 0:
        return {}
    counts = Counter()
    for idx in indices:
        item = data[idx]
        counts["actions"] += int(item["actions"])
        counts["looks"] += int(item["looks"])
        counts["crosses"] += clamp_cross(int(item["crosses"]))
    return {
        "total": total,
        "actions_pos_rate": counts["actions"] / total,
        "looks_pos_rate": counts["looks"] / total,
        "crosses_pos_rate": counts["crosses"] / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sequences_all.pkl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-pct", type=float, default=0.75)
    parser.add_argument("--val-pct", type=float, default=0.10)
    parser.add_argument("--test-pct", type=float, default=0.15)
    parser.add_argument("--cross-pos-ratio", type=float, default=0.30)
    parser.add_argument("--out-prefix", default="sequences_all")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Missing {args.input}")

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    total = len(data)
    train_idx, val_idx, test_idx = split_indices(
        total, args.train_pct, args.val_pct, args.seed
    )

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    for name, idxs in splits.items():
        out_path = f"{args.out_prefix}_{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump([data[i] for i in idxs], f)

        summary = summarize(data, idxs)
        print(f"{out_path}")
        print(
            "  total={total} | actions_pos={actions_pos_rate:.4f} | "
            "looks_pos={looks_pos_rate:.4f} | crosses_pos={crosses_pos_rate:.4f}".format(**summary)
        )

        balanced_idx = balance_split(data, idxs, args.cross_pos_ratio, args.seed)
        balanced_out = f"{args.out_prefix}_{name}_balanced_30_70.pkl"
        with open(balanced_out, "wb") as f:
            pickle.dump([data[i] for i in balanced_idx], f)

        summary_bal = summarize(data, balanced_idx)
        print(f"{balanced_out}")
        print(
            "  total={total} | actions_pos={actions_pos_rate:.4f} | "
            "looks_pos={looks_pos_rate:.4f} | crosses_pos={crosses_pos_rate:.4f}".format(**summary_bal)
        )


if __name__ == "__main__":
    main()
