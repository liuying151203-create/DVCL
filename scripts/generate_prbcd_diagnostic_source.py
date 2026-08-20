import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import file_sha256, load_split_artifact


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PRBCD/HetePRBCD sources with auditable optimization diagnostics."
    )
    parser.add_argument("--dataset", required=True, choices=["acm", "dblp", "aminer"])
    parser.add_argument("--attack", required=True, choices=["prbcd", "heteprbcd"])
    parser.add_argument("--rate", required=True, type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--legacy-root", default=str(ROOT.parent / "HSeCo"))
    parser.add_argument("--heteroguard-root", default=str(ROOT.parent / "Hetero-Guard"))
    parser.add_argument("--data-root")
    parser.add_argument("--split-path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--victim-epochs", type=int, default=200)
    parser.add_argument("--attack-epochs", type=int, default=200)
    parser.add_argument("--fine-tune-epochs", type=int, default=50)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-hidden", type=int)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--loss-type", default="CE", choices=["CE", "CW"])
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--biased", action="store_true")
    parser.add_argument("--lamb", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    heteroguard_root = Path(args.heteroguard_root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else legacy_root / "data"
    generator = _load_generator(legacy_root)
    values = argparse.Namespace(
        dataname=args.dataset,
        atk_name="PRBCD" if args.attack == "prbcd" else "HetePRBCD",
        atk_rate=args.rate,
        data_root=str(data_root),
        heteroguard_root=str(heteroguard_root),
        output=str(Path(args.output).resolve()),
        seed=args.seed,
        cuda=args.cuda,
        victim_epochs=args.victim_epochs,
        attack_epochs=args.attack_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        block_size=args.block_size,
        num_layers=args.num_layers,
        num_hidden=args.num_hidden,
        dropout=args.dropout,
        loss_type=args.loss_type,
        fill_type="zero",
        constrained=args.constrained,
        biased=args.biased,
        lamb=args.lamb,
        dry_run=False,
    )
    values = generator.resolve_defaults(values)
    generator.set_random_seed(values.seed)
    generator.add_heteroguard_path(str(heteroguard_root))
    generator.ensure_deeprobust_compat()
    if torch.cuda.is_available():
        torch.cuda.set_device(values.cuda)
    data, num_classes, head_node = generator.build_hseco_hdata(
        values.dataname, values.data_root, values.fill_type
    )
    split_path = (
        Path(args.split_path).resolve()
        if args.split_path
        else ROOT / "data" / "splits" / args.dataset / "paper_seed_1.pt"
    )
    split = load_split_artifact(split_path)
    _bind_frozen_split(data, head_node, split)
    budget, symmetric = generator.select_budget(
        data, values.dataname, values.constrained
    )
    expected = int((values.atk_rate / 100.0) * (data.num_edges / 2))
    if values.atk_name == "PRBCD":
        modified, before, after, history = _run_prbcd(
            generator, values, data, num_classes, head_node, budget, expected
        )
    else:
        modified, before, after, history = _run_heteprbcd(
            generator, values, data, num_classes, head_node, budget, symmetric, expected
        )
    actual = generator.count_budget_changes(data.cpu(), modified.cpu(), budget)
    tolerance = max(1, int(np.ceil(expected * 0.02)))
    if actual > expected or expected - actual > tolerance:
        raise RuntimeError(
            f"Attack budget was not realized: expected={expected}, actual={actual}"
        )
    modified.attack_metadata = {
        "generator": "scripts/generate_prbcd_diagnostic_source.py",
        "external_generator": str((legacy_root / "scripts" / "gen_heteroguard_prbcd.py").resolve()),
        "external_revisions": {
            "hseco": _git_revision(legacy_root),
            "heteroguard": _git_revision(heteroguard_root),
        },
        "split_artifact": str(split_path),
        "split_sha256": file_sha256(split_path),
        "dataset": values.dataname,
        "attack": values.atk_name,
        "rate": values.atk_rate,
        "seed": values.seed,
        "constrained": values.constrained,
        "biased": values.biased,
        "lambda": values.lamb if values.biased else None,
        "budget": [list(edge_type) for edge_type in budget],
        "victim_model": "GCN" if values.atk_name == "PRBCD" else "HeteroSAGE",
        "victim_epochs": values.victim_epochs,
        "attack_epochs": values.attack_epochs,
        "fine_tune_epochs": values.fine_tune_epochs,
        "block_size": values.block_size,
        "expected_perturbations": expected,
        "actual_perturbations": actual,
        "budget_tolerance": tolerance,
        "surrogate_before": before,
        "surrogate_after": after,
        "optimization_history": [float(value) for value in history],
    }
    output = Path(values.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modified.cpu(), output)
    print(f"Saved {output}")
    print(f"surrogate_before={before}")
    print(f"surrogate_after={after}")
    return 0


def _bind_frozen_split(data, head_node, split):
    store = data[head_node]
    if len(store.y) != len(split.train_mask):
        raise ValueError("Attack generator and frozen split node counts differ")
    store.train_mask = split.train_mask.clone()
    store.val_mask = split.val_mask.clone()
    store.test_mask = split.test_mask.clone()


def _run_heteprbcd(generator, args, data, classes, head, budget, symmetric, count):
    from libs.attack.const_hete_prbcd import ConstHetePRBCD
    from libs.gnn.graphsage import HeteroSAGE

    victim = HeteroSAGE(
        data, args.num_layers, args.num_hidden, classes, head, dropout=args.dropout
    )
    victim.fit(epochs=args.victim_epochs)
    before = _hetero_metrics(victim, data, head)
    attacker = ConstHetePRBCD(
        victim, data, args.block_size, head_node=head, budget=budget,
        hete_symmetric=symmetric, epochs=args.attack_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        lamb=args.lamb if args.biased else None, loss_type=args.loss_type,
    )
    history = attacker.attack(count, check_modified=False)
    modified = attacker.modified
    after = _hetero_metrics(victim, modified, head)
    return modified.cpu(), before, after, history


def _run_prbcd(generator, args, data, classes, head, budget, count):
    from libs.attack.const_prbcd import ConstPRBCD
    from libs.gnn.gcn import GCN
    from torch_geometric import utils as pyg_utils

    oriented, symmetric = generator.orient_budget_for_homo(data, budget)
    subset = generator.make_relation_subset(data, oriented)
    homogeneous = generator.to_homogeneous_for_prbcd(subset, head_node=head)
    if getattr(homogeneous, "edge_weight", None) is None:
        homogeneous.edge_weight = torch.ones(homogeneous.edge_index.size(1))
    homogeneous.edge_index, homogeneous.edge_weight = pyg_utils.to_undirected(
        homogeneous.edge_index, homogeneous.edge_weight,
        num_nodes=homogeneous.num_nodes,
    )
    victim = GCN(
        homogeneous, args.num_layers, args.num_hidden, classes, dropout=args.dropout
    )
    victim.fit(epochs=args.victim_epochs)
    before = _homo_metrics(victim, homogeneous)
    attacker = ConstPRBCD(
        victim, homogeneous, args.block_size, constraints=[],
        epochs=args.attack_epochs, fine_tune_epochs=args.fine_tune_epochs,
        loss_type=args.loss_type,
    )
    history = []
    attacker.attack(count, loss_callback=lambda loss, _: history.append(float(loss)))
    after = _homo_metrics(victim, attacker.modified)
    replaced = list(oriented) + [
        edge_type for edge_type in symmetric.values() if edge_type in data.edge_types
    ]
    modified = generator.apply_homo_edges_to_hetero(
        data.cpu(), attacker.modified.cpu(), replaced
    )
    return modified, before, after, history


def _hetero_metrics(model, data, head):
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict, data.edge_weight_dict)
    labels = data[head].y
    return _split_metrics(logits, labels, data[head])


def _homo_metrics(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_weight)
    return _split_metrics(logits, data.y, data)


def _split_metrics(logits, labels, store):
    result = {}
    for split in ("train", "val", "test"):
        mask = store[f"{split}_mask"].bool()
        prediction = logits[mask].argmax(dim=1)
        result[f"{split}_micro_f1"] = float((prediction == labels[mask]).float().mean())
    result["micro_f1"] = result["test_micro_f1"]
    return result


def _load_generator(legacy_root):
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    path = legacy_root / "scripts" / "gen_heteroguard_prbcd.py"
    spec = importlib.util.spec_from_file_location("dvcl_external_prbcd_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
