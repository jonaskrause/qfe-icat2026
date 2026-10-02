"""
QFE - Quantum Functional Evolution

Modes:
- RANDOM: random control.
- FIDELITY_ONLY: classical genetic baseline using fidelity as score.
- QFE_SCORE: fidelity plus functional information in selection.
- QFE_UNITARY_FLOW: proposed method. Uses functional information to generate a variational unitary-flow update through parameter-shift gradients.

The ansatz is explicitly GHZ-reachable: Ry(pi/2) on q0, followed by CRY(pi) along the chain.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import json
import platform
import time
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, state_fidelity
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: qiskit. Install it with: python -m pip install qiskit qiskit-aer matplotlib numpy"
    ) from exc


Individual = Dict[str, np.ndarray]

EVAL_COUNTER = {"circuit_evaluations": 0}



@dataclass
class Config:
    n_qubits: int = 3
    layers: int = 2
    population: int = 40
    generations: int = 80
    seed: int = 0
    output_dir: str = "qfe_outputs"
    modes: Tuple[str, ...] = ("FIDELITY_ONLY", "QFE_SCORE", "FLOW_ONLY", "GRADIENT_ONLY", "QFE_UNITARY_FLOW", "RANDOM")

    # GA baseline
    elite_fraction: float = 0.25
    mutation_initial: float = 0.25
    mutation_final: float = 0.015

    # Functional information
    fi_epsilon: float = 1e-12

    # QFE score selection
    qfe_score_fidelity_weight: float = 0.90
    qfe_score_fi_weight: float = 0.10

    # QFE unitary-flow dynamics
    flow_elite_fraction: float = 0.30
    flow_lr_initial: float = 0.55
    flow_lr_final: float = 0.10
    flow_mutation_initial: float = 0.10
    flow_mutation_final: float = 0.000
    flow_fi_weight_initial: float = 0.25
    flow_fi_weight_final: float = 0.03
    flow_grad_epsilon: float = 1e-8
    flow_refine_best_copies: int = 2

    # Ablation modes for paper-ready experiments
    gradient_only_elite_fraction: float = 0.30
    gradient_only_lr_initial: float = 0.45
    gradient_only_lr_final: float = 0.08

    # Parameter-shift cost control
    # Each individual has layers * n_qubits root/local angles plus layers*(n-1) ent angles.
    fast_gradient: bool = False
    max_gradient_parameters: int = 24

    # Reporting thresholds
    thresholds: Tuple[float, ...] = (0.80, 0.90, 0.95, 0.99)
    save_every: int = 10


def schedule(initial: float, final: float, gen: int, generations: int) -> float:
    if generations <= 1:
        return final
    t = gen / (generations - 1)
    return float((1 - t) * initial + t * final)


def rng_from_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def random_individual(cfg: Config, rng: np.random.Generator) -> Individual:
    return {
        "root": rng.uniform(0, 2 * np.pi, size=(cfg.layers,)),
        "ent": rng.uniform(0, 2 * np.pi, size=(cfg.layers, cfg.n_qubits - 1)),
        "local": rng.uniform(0, 2 * np.pi, size=(cfg.layers, cfg.n_qubits)),
    }


def ghz_template(cfg: Config) -> Individual:
    ind = {
        "root": np.zeros((cfg.layers,), dtype=float),
        "ent": np.zeros((cfg.layers, cfg.n_qubits - 1), dtype=float),
        "local": np.zeros((cfg.layers, cfg.n_qubits), dtype=float),
    }
    ind["root"][0] = np.pi / 2
    ind["ent"][0, :] = np.pi
    return ind


def copy_ind(ind: Individual) -> Individual:
    return {k: np.array(v, copy=True) for k, v in ind.items()}


def wrap_ind(ind: Individual) -> Individual:
    return {k: np.mod(v, 2 * np.pi) for k, v in ind.items()}


def build_candidate_circuit(ind: Individual, cfg: Config) -> QuantumCircuit:
    qc = QuantumCircuit(cfg.n_qubits)
    for layer in range(cfg.layers):
        # Root superposition control. This is enough to reach GHZ when root=pi/2.
        qc.ry(float(ind["root"][layer]), 0)

        # Chain of controlled rotations. CRY(pi) maps |10> -> |11>, generating GHZ.
        for q in range(cfg.n_qubits - 1):
            qc.cry(float(ind["ent"][layer, q]), q, q + 1)

        # Local phase/control rotations. They enlarge the variational manifold.
        # For the exact GHZ template these are zero.
        for q in range(cfg.n_qubits):
            qc.rz(float(ind["local"][layer, q]), q)
    return qc


def build_target_circuit(cfg: Config) -> QuantumCircuit:
    qc = QuantumCircuit(cfg.n_qubits)
    qc.h(0)
    for q in range(cfg.n_qubits - 1):
        qc.cx(q, q + 1)
    return qc


def fidelity(ind: Individual, target: Statevector, cfg: Config) -> float:
    EVAL_COUNTER["circuit_evaluations"] += 1
    sv = Statevector.from_instruction(build_candidate_circuit(ind, cfg))
    return float(np.clip(state_fidelity(sv, target), 0.0, 1.0))


def functional_information(f: float, eps: float) -> float:
    # Functional information increases as the miss probability goes to zero.
    return float(-math.log2(max(1.0 - f, eps)))


def normalize(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if abs(hi - lo) < 1e-15:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def evaluate(pop: List[Individual], mode: str, target: Statevector, cfg: Config, rng: np.random.Generator) -> List[dict]:
    fs = np.array([fidelity(ind, target, cfg) for ind in pop], dtype=float)
    fis = np.array([functional_information(f, cfg.fi_epsilon) for f in fs], dtype=float)
    fi_norm = normalize(fis)

    rows = []
    for i, ind in enumerate(pop):
        if mode == "FIDELITY_ONLY":
            score = fs[i]
        elif mode == "QFE_SCORE":
            score = cfg.qfe_score_fidelity_weight * fs[i] + cfg.qfe_score_fi_weight * fi_norm[i] * fs[i]
        elif mode == "FLOW_ONLY":
            # Ablation: FI-shaped selection plus contraction, without parameter-shift refinement.
            score = 0.92 * fs[i] + 0.08 * fi_norm[i] * fs[i]
        elif mode == "GRADIENT_ONLY":
            # Ablation: fidelity objective with local parameter-shift refinement, without FI shaping.
            score = fs[i]
        elif mode == "QFE_UNITARY_FLOW":
            # Selection still mostly fidelity; dynamics is the real distinction.
            score = 0.92 * fs[i] + 0.08 * fi_norm[i] * fs[i]
        elif mode == "RANDOM":
            score = float(rng.random())
        else:
            raise ValueError(f"Unknown mode: {mode}")
        rows.append({"idx": i, "individual": ind, "fidelity": fs[i], "fi": fis[i], "fi_norm": fi_norm[i], "score": float(score)})
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def crossover_uniform(a: Individual, b: Individual, rng: np.random.Generator) -> Individual:
    child = {}
    for key in a:
        mask = rng.random(a[key].shape) < 0.5
        child[key] = np.where(mask, a[key], b[key])
    return child


def phase_average(a: Individual, b: Individual, wa: float, wb: float) -> Individual:
    child = {}
    for key in a:
        z = wa * np.exp(1j * a[key]) + wb * np.exp(1j * b[key])
        child[key] = np.mod(np.angle(z), 2 * np.pi)
    return child


def mutate(ind: Individual, sigma: float, rng: np.random.Generator) -> Individual:
    out = copy_ind(ind)
    if sigma <= 0:
        return out
    for k in out:
        out[k] = np.mod(out[k] + rng.normal(0.0, sigma, size=out[k].shape), 2 * np.pi)
    return out


def next_fidelity_only(scored: List[dict], gen: int, cfg: Config, rng: np.random.Generator) -> List[Individual]:
    n_elite = max(1, int(cfg.elite_fraction * cfg.population))
    elites = [copy_ind(r["individual"]) for r in scored[:n_elite]]
    sigma = schedule(cfg.mutation_initial, cfg.mutation_final, gen, cfg.generations)

    new = elites.copy()
    while len(new) < cfg.population:
        a = elites[int(rng.integers(0, len(elites)))]
        b = elites[int(rng.integers(0, len(elites)))]
        new.append(mutate(crossover_uniform(a, b, rng), sigma, rng))
    return new


def next_qfe_score(scored: List[dict], gen: int, cfg: Config, rng: np.random.Generator) -> List[Individual]:
    n_elite = max(1, int(cfg.elite_fraction * cfg.population))
    elites = [copy_ind(r["individual"]) for r in scored[:n_elite]]
    sigma = schedule(cfg.mutation_initial * 0.75, cfg.mutation_final * 0.5, gen, cfg.generations)

    scores = np.array([max(r["score"], 1e-12) for r in scored], dtype=float)
    probs = scores / np.sum(scores)

    new = elites.copy()
    while len(new) < cfg.population:
        ia = int(rng.choice(len(scored), p=probs))
        ib = int(rng.choice(len(scored), p=probs))
        a, b = scored[ia], scored[ib]
        total = a["score"] + b["score"] + 1e-12
        child = phase_average(a["individual"], b["individual"], a["score"] / total, b["score"] / total)
        new.append(mutate(child, sigma, rng))
    return new


def parameter_refs(ind: Individual) -> List[Tuple[str, Tuple[int, ...]]]:
    refs = []
    for key in ("root", "ent", "local"):
        for idx in np.ndindex(ind[key].shape):
            refs.append((key, idx))
    return refs


def objective_for_flow(f: float, gen: int, cfg: Config, fi_min: float, fi_max: float) -> float:
    """Local refinement objective used by QFE_UNITARY_FLOW.

    Parameters
    ----------
    f:
        Fidelity value of the candidate state.
    gen:
        Current generation.
    cfg:
        Experiment configuration.
    fi_min, fi_max:
        Minimum and maximum FI values observed in the current scored population.
    """
    w = schedule(cfg.flow_fi_weight_initial, cfg.flow_fi_weight_final, gen, cfg.generations)
    fi = functional_information(f, cfg.fi_epsilon)
    fi_norm = (fi - fi_min) / (fi_max - fi_min + 1e-12)
    fi_norm = float(np.clip(fi_norm, 0.0, 1.0))
    return float(f + w * fi_norm)


def parameter_shift_gradient(ind: Individual, target: Statevector, gen: int, cfg: Config, rng: np.random.Generator, fi_min: float, fi_max: float) -> Individual:
    refs = parameter_refs(ind)
    if cfg.fast_gradient and len(refs) > cfg.max_gradient_parameters:
        chosen = rng.choice(len(refs), size=cfg.max_gradient_parameters, replace=False)
        refs = [refs[int(i)] for i in chosen]

    grad = {k: np.zeros_like(v, dtype=float) for k, v in ind.items()}
    shift = np.pi / 2

    for key, idx in refs:
        plus = copy_ind(ind)
        minus = copy_ind(ind)
        plus[key][idx] = (plus[key][idx] + shift) % (2 * np.pi)
        minus[key][idx] = (minus[key][idx] - shift) % (2 * np.pi)

        f_plus = fidelity(plus, target, cfg)
        f_minus = fidelity(minus, target, cfg)
        o_plus = objective_for_flow(f_plus, gen, cfg, fi_min, fi_max)
        o_minus = objective_for_flow(f_minus, gen, cfg, fi_min, fi_max)
        grad[key][idx] = 0.5 * (o_plus - o_minus)

    return grad


def gradient_step(ind: Individual, grad: Individual, lr: float) -> Individual:
    out = copy_ind(ind)
    for k in out:
        out[k] = np.mod(out[k] + lr * grad[k], 2 * np.pi)
    return out


def contract_toward_best(ind: Individual, best: Individual, strength: float) -> Individual:
    out = copy_ind(ind)
    for k in out:
        # Shortest circular direction from ind to best.
        direction = np.angle(np.exp(1j * (best[k] - out[k])))
        out[k] = np.mod(out[k] + strength * direction, 2 * np.pi)
    return out



def objective_fidelity_only(f: float, gen: int, cfg: Config, fi_min: float, fi_max: float) -> float:
    """Pure-fidelity objective for the GRADIENT_ONLY ablation."""
    return float(f)


def parameter_shift_gradient_custom(
    ind: Individual,
    target: Statevector,
    gen: int,
    cfg: Config,
    rng: np.random.Generator,
    fi_min: float,
    fi_max: float,
    objective_fn,
) -> Individual:
    refs = parameter_refs(ind)
    if cfg.fast_gradient and len(refs) > cfg.max_gradient_parameters:
        chosen = rng.choice(len(refs), size=cfg.max_gradient_parameters, replace=False)
        refs = [refs[int(i)] for i in chosen]

    grad = {k: np.zeros_like(v, dtype=float) for k, v in ind.items()}
    shift = np.pi / 2

    for key, idx in refs:
        plus = copy_ind(ind)
        minus = copy_ind(ind)
        plus[key][idx] = (plus[key][idx] + shift) % (2 * np.pi)
        minus[key][idx] = (minus[key][idx] - shift) % (2 * np.pi)

        f_plus = fidelity(plus, target, cfg)
        f_minus = fidelity(minus, target, cfg)
        o_plus = objective_fn(f_plus, gen, cfg, fi_min, fi_max)
        o_minus = objective_fn(f_minus, gen, cfg, fi_min, fi_max)
        grad[key][idx] = 0.5 * (o_plus - o_minus)

    return grad


def next_gradient_only(scored: List[dict], gen: int, target: Statevector, cfg: Config, rng: np.random.Generator) -> List[Individual]:
    """Ablation mode: evolutionary selection plus parameter-shift refinement, but no FI term and no contraction."""
    n_elite = max(1, int(cfg.gradient_only_elite_fraction * cfg.population))
    elites = [copy_ind(r["individual"]) for r in scored[:n_elite]]
    lr = schedule(cfg.gradient_only_lr_initial, cfg.gradient_only_lr_final, gen, cfg.generations)
    sigma = schedule(cfg.mutation_initial * 0.50, cfg.mutation_final * 0.50, gen, cfg.generations)

    new: List[Individual] = []
    for e in elites:
        g = parameter_shift_gradient_custom(e, target, gen, cfg, rng, 0.0, 1.0, objective_fidelity_only)
        new.append(gradient_step(e, g, lr))

    while len(new) < cfg.population:
        a = elites[int(rng.integers(0, len(elites)))]
        b = elites[int(rng.integers(0, len(elites)))]
        child = crossover_uniform(a, b, rng)
        g = parameter_shift_gradient_custom(child, target, gen, cfg, rng, 0.0, 1.0, objective_fidelity_only)
        child = gradient_step(child, g, lr)
        child = mutate(child, sigma, rng)
        new.append(child)

    return new[:cfg.population]


def next_flow_only(scored: List[dict], gen: int, cfg: Config, rng: np.random.Generator) -> List[Individual]:
    """Ablation mode: FI-shaped selection plus adaptive contraction, but no parameter-shift refinement."""
    n_elite = max(1, int(cfg.flow_elite_fraction * cfg.population))
    elites = [copy_ind(r["individual"]) for r in scored[:n_elite]]
    best = copy_ind(scored[0]["individual"])

    t = gen / max(1, cfg.generations - 1)
    contraction = 0.05 + 0.55 * (t ** 2)
    sigma = schedule(cfg.flow_mutation_initial, cfg.flow_mutation_final, gen, cfg.generations)

    scores = np.array([max(r["score"], 1e-12) for r in scored], dtype=float)
    temp = schedule(0.80, 0.18, gen, cfg.generations)
    logits = normalize(scores) / max(temp, 1e-8)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)

    new: List[Individual] = elites.copy()
    while len(new) < cfg.population:
        ia = int(rng.choice(len(scored), p=probs))
        ib = int(rng.choice(len(scored), p=probs))
        a, b = scored[ia], scored[ib]
        total = a["score"] + b["score"] + 1e-12
        child = phase_average(a["individual"], b["individual"], a["score"] / total, b["score"] / total)
        child = contract_toward_best(child, best, contraction)
        child = mutate(child, sigma, rng)
        new.append(child)

    return new[:cfg.population]


def next_qfe_unitary_flow(scored: List[dict], gen: int, target: Statevector, cfg: Config, rng: np.random.Generator) -> List[Individual]:
    n_elite = max(1, int(cfg.flow_elite_fraction * cfg.population))
    elites = [copy_ind(r["individual"]) for r in scored[:n_elite]]
    best = copy_ind(scored[0]["individual"])

    # Adaptive FI normalization for the local unitary-flow objective.
    fi_values = np.array([r["fi"] for r in scored], dtype=float)
    fi_min = float(np.min(fi_values))
    fi_max = float(np.max(fi_values))

    lr = schedule(cfg.flow_lr_initial, cfg.flow_lr_final, gen, cfg.generations)
    sigma = schedule(cfg.flow_mutation_initial, cfg.flow_mutation_final, gen, cfg.generations)

    # Late-stage contraction: this is why mean fidelity also approaches 1.
    t = gen / max(1, cfg.generations - 1)
    contraction = 0.05 + 0.55 * (t ** 2)

    # Gradient-refine elites.
    new: List[Individual] = []
    for e in elites:
        g = parameter_shift_gradient(e, target, gen, cfg, rng, fi_min, fi_max)
        refined = gradient_step(e, g, lr)
        refined = contract_toward_best(refined, best, contraction * 0.25)
        new.append(refined)

    # Add a few copies/refinements of the best individual to stabilize convergence.
    for _ in range(max(0, cfg.flow_refine_best_copies)):
        if len(new) < cfg.population:
            g = parameter_shift_gradient(best, target, gen, cfg, rng, fi_min, fi_max)
            new.append(gradient_step(best, g, lr))

    # Parent probabilities biased by score.
    scores = np.array([max(r["score"], 1e-12) for r in scored], dtype=float)
    temp = schedule(0.80, 0.18, gen, cfg.generations)
    logits = normalize(scores) / max(temp, 1e-8)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)

    while len(new) < cfg.population:
        ia = int(rng.choice(len(scored), p=probs))
        ib = int(rng.choice(len(scored), p=probs))
        a, b = scored[ia], scored[ib]
        total = a["score"] + b["score"] + 1e-12
        child = phase_average(a["individual"], b["individual"], a["score"] / total, b["score"] / total)
        child = contract_toward_best(child, best, contraction)
        g = parameter_shift_gradient(child, target, gen, cfg, rng, fi_min, fi_max)
        child = gradient_step(child, g, lr)
        child = mutate(child, sigma, rng)
        new.append(child)

    return new[:cfg.population]


def save_plots(history: List[dict], out: Path) -> None:
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    modes = sorted(set(r["mode"] for r in history))
    for metric, ylabel, filename, title in [
        ("best_fidelity", "Best fidelity", "global_best_fidelity.png", "Best fidelity by mode"),
        ("mean_fidelity", "Mean fidelity", "global_mean_fidelity.png", "Mean fidelity by mode"),
        ("best_fi", "Best functional information", "global_best_fi.png", "Best functional information by mode"),
    ]:
        plt.figure(figsize=(11, 6))
        for m in modes:
            rows = sorted([r for r in history if r["mode"] == m], key=lambda r: r["generation"])
            plt.plot([r["generation"] for r in rows], [r[metric] for r in rows], marker="o", markersize=2.5, linewidth=1.5, label=m)
        plt.xlabel("Generation")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / filename, dpi=180)
        plt.close()

    # Cost-normalized convergence plot for paper reporting.
    plt.figure(figsize=(11, 6))
    for m in modes:
        rows = sorted([r for r in history if r["mode"] == m], key=lambda r: r["cumulative_circuit_evaluations"])
        if rows and "cumulative_circuit_evaluations" in rows[0]:
            plt.plot([r["cumulative_circuit_evaluations"] for r in rows], [r["best_fidelity"] for r in rows], marker="o", markersize=2.5, linewidth=1.5, label=m)
    plt.xlabel("Cumulative circuit evaluations")
    plt.ylabel("Best fidelity")
    plt.title("Cost-normalized best fidelity")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "cost_normalized_best_fidelity.png", dpi=180)
    plt.close()

    # Advantage plot.
    qfe = sorted([r for r in history if r["mode"] == "QFE_UNITARY_FLOW"], key=lambda r: r["generation"])
    base = sorted([r for r in history if r["mode"] == "FIDELITY_ONLY"], key=lambda r: r["generation"])
    if qfe and base:
        n = min(len(qfe), len(base))
        plt.figure(figsize=(11, 6))
        plt.axhline(0.0, linewidth=1.0)
        plt.plot([qfe[i]["generation"] for i in range(n)], [qfe[i]["best_fidelity"] - base[i]["best_fidelity"] for i in range(n)], marker="o", markersize=2.5, label="Best fidelity gain")
        plt.plot([qfe[i]["generation"] for i in range(n)], [qfe[i]["mean_fidelity"] - base[i]["mean_fidelity"] for i in range(n)], marker="o", markersize=2.5, label="Mean fidelity gain")
        plt.xlabel("Generation")
        plt.ylabel("QFE_UNITARY_FLOW - FIDELITY_ONLY")
        plt.title("QFE unitary-flow advantage")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / "ADVANTAGE_qfe_unitary_flow_minus_fidelity_only.png", dpi=180)
        plt.close()


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def summarize(history: List[dict], cfg: Config) -> List[dict]:
    summary = []
    for m in sorted(set(r["mode"] for r in history)):
        rows = sorted([r for r in history if r["mode"] == m], key=lambda r: r["generation"])
        gens = np.array([r["generation"] for r in rows], dtype=float)
        best = np.array([r["best_fidelity"] for r in rows], dtype=float)
        mean = np.array([r["mean_fidelity"] for r in rows], dtype=float)
        item = {
            "mode": m,
            "final_best_fidelity": float(best[-1]),
            "final_mean_fidelity": float(mean[-1]),
            "max_best_fidelity": float(np.max(best)),
            "auc_best_fidelity": float(np.trapz(best, gens) / max(1.0, gens[-1] - gens[0])),
            "auc_mean_fidelity": float(np.trapz(mean, gens) / max(1.0, gens[-1] - gens[0])),
            "final_circuit_evaluations": int(rows[-1].get("cumulative_circuit_evaluations", 0)),
            "evals_to_final_best": float(rows[-1].get("evaluations_per_best_fidelity", 0.0)),
        }
        for th in cfg.thresholds:
            hits = [r["generation"] for r in rows if r["best_fidelity"] >= th]
            item[f"first_gen_best_ge_{th}"] = hits[0] if hits else "not_reached"
        summary.append(item)
    return summary


def run_mode(mode: str, cfg: Config, target: Statevector, out: Path, history: List[dict]) -> None:
    rng = rng_from_seed(cfg.seed)
    pop = [random_individual(cfg, rng) for _ in range(cfg.population)]
    start = time.time()
    eval_start = EVAL_COUNTER["circuit_evaluations"]

    print("\n" + "=" * 70)
    print(f"Running mode: {mode}")
    print("=" * 70)

    for gen in range(cfg.generations):
        scored = evaluate(pop, mode, target, cfg, rng)
        fids = np.array([r["fidelity"] for r in scored], dtype=float)
        fis = np.array([r["fi"] for r in scored], dtype=float)
        row = {
            "mode": mode,
            "generation": gen,
            "n_qubits": cfg.n_qubits,
            "layers": cfg.layers,
            "population": cfg.population,
            "seed": cfg.seed,
            "best_fidelity": float(np.max(fids)),
            "mean_fidelity": float(np.mean(fids)),
            "std_fidelity": float(np.std(fids)),
            "best_score": float(scored[0]["score"]),
            "mean_score": float(np.mean([r["score"] for r in scored])),
            "best_fi": float(np.max(fis)),
            "mean_fi": float(np.mean(fis)),
            "elapsed_seconds": float(time.time() - start),
            "cumulative_circuit_evaluations": int(EVAL_COUNTER["circuit_evaluations"] - eval_start),
            "evaluations_per_best_fidelity": float((EVAL_COUNTER["circuit_evaluations"] - eval_start) / max(np.max(fids), 1e-12)),
        }
        history.append(row)
        print(f"{mode:18s} | Gen {gen:03d} | BestF={row['best_fidelity']:.6f} | MeanF={row['mean_fidelity']:.6f} | BestFI={row['best_fi']:.3f}")

        if mode == "RANDOM":
            pop = [random_individual(cfg, rng) for _ in range(cfg.population)]
        elif mode == "FIDELITY_ONLY":
            pop = next_fidelity_only(scored, gen, cfg, rng)
        elif mode == "QFE_SCORE":
            pop = next_qfe_score(scored, gen, cfg, rng)
        elif mode == "FLOW_ONLY":
            pop = next_flow_only(scored, gen, cfg, rng)
        elif mode == "GRADIENT_ONLY":
            pop = next_gradient_only(scored, gen, target, cfg, rng)
        elif mode == "QFE_UNITARY_FLOW":
            pop = next_qfe_unitary_flow(scored, gen, target, cfg, rng)
        else:
            raise ValueError(mode)

        row["cumulative_circuit_evaluations"] = int(EVAL_COUNTER["circuit_evaluations"] - eval_start)
        row["evaluations_per_best_fidelity"] = float(row["cumulative_circuit_evaluations"] / max(row["best_fidelity"], 1e-12))

        if gen % max(1, cfg.save_every) == 0 or gen == cfg.generations - 1:
            write_csv(out / "history.csv", history)
            save_plots(history, out)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--n-qubits", type=int, default=3)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--generations", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="qfe_aligned_v2_outputs")
    p.add_argument("--modes", type=str, default="FIDELITY_ONLY,QFE_SCORE,FLOW_ONLY,GRADIENT_ONLY,QFE_UNITARY_FLOW,RANDOM")
    p.add_argument("--fast-gradient", action="store_true")
    p.add_argument("--max-gradient-parameters", type=int, default=24)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--thresholds", type=str, default="0.80,0.90,0.95,0.99")
    args = p.parse_args()
    return Config(
        n_qubits=args.n_qubits,
        layers=args.layers,
        population=args.population,
        generations=args.generations,
        seed=args.seed,
        output_dir=args.output_dir,
        modes=tuple(x.strip() for x in args.modes.split(",") if x.strip()),
        fast_gradient=args.fast_gradient,
        max_gradient_parameters=args.max_gradient_parameters,
        save_every=args.save_every,
        thresholds=tuple(float(x.strip()) for x in args.thresholds.split(",") if x.strip()),
    )



def write_experiment_metadata(out: Path, cfg: Config, template_f: float) -> None:
    """Save reproducibility metadata required for paper-ready reporting."""
    payload = {
        "experiment": "QFE ICAT 2026 comparison",
        "python": sys.version,
        "platform": platform.platform(),
        "n_qubits": cfg.n_qubits,
        "layers": cfg.layers,
        "population": cfg.population,
        "generations": cfg.generations,
        "seed": cfg.seed,
        "modes": list(cfg.modes),
        "thresholds": list(cfg.thresholds),
        "fast_gradient": cfg.fast_gradient,
        "max_gradient_parameters": cfg.max_gradient_parameters,
        "template_ghz_fidelity": template_f,
        "objective": "QFE_UNITARY_FLOW uses fidelity plus adaptive min-max normalized functional information during parameter-shift refinement.",
    }
    with open(out / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_comparison_tables(history: List[dict], cfg: Config, out: Path) -> None:
    """Create direct comparison tables across all modes.

    Outputs:
    - final_comparison.csv: one row per mode, sorted by final best fidelity.
    - pairwise_advantage_vs_fidelity_only.csv: difference against FIDELITY_ONLY.
    - threshold_times.csv: generation and evaluation count when each threshold is first reached.
    """
    if not history:
        return

    modes = sorted(set(r["mode"] for r in history))
    final_rows = []
    threshold_rows = []

    for m in modes:
        rows = sorted([r for r in history if r["mode"] == m], key=lambda r: r["generation"])
        best_values = np.array([r["best_fidelity"] for r in rows], dtype=float)
        mean_values = np.array([r["mean_fidelity"] for r in rows], dtype=float)
        final = rows[-1]
        best_idx = int(np.argmax(best_values))
        final_rows.append({
            "rank_metric": "final_best_fidelity",
            "mode": m,
            "final_best_fidelity": float(final["best_fidelity"]),
            "final_mean_fidelity": float(final["mean_fidelity"]),
            "max_best_fidelity": float(np.max(best_values)),
            "generation_of_max_best": int(rows[best_idx]["generation"]),
            "final_best_fi": float(final["best_fi"]),
            "final_mean_fi": float(final["mean_fi"]),
            "final_circuit_evaluations": int(final["cumulative_circuit_evaluations"]),
            "elapsed_seconds": float(final["elapsed_seconds"]),
        })
        for th in cfg.thresholds:
            hit = next((r for r in rows if r["best_fidelity"] >= th), None)
            threshold_rows.append({
                "mode": m,
                "threshold": th,
                "first_generation": hit["generation"] if hit else "not_reached",
                "circuit_evaluations": hit["cumulative_circuit_evaluations"] if hit else "not_reached",
                "elapsed_seconds": hit["elapsed_seconds"] if hit else "not_reached",
            })

    final_rows = sorted(final_rows, key=lambda r: r["final_best_fidelity"], reverse=True)
    write_csv(out / "final_comparison.csv", final_rows)
    write_csv(out / "threshold_times.csv", threshold_rows)

    base = {r["generation"]: r for r in history if r["mode"] == "FIDELITY_ONLY"}
    pairwise = []
    if base:
        for r in sorted(history, key=lambda x: (x["mode"], x["generation"])):
            if r["mode"] == "FIDELITY_ONLY" or r["generation"] not in base:
                continue
            b = base[r["generation"]]
            pairwise.append({
                "mode": r["mode"],
                "generation": r["generation"],
                "delta_best_fidelity_vs_fidelity_only": float(r["best_fidelity"] - b["best_fidelity"]),
                "delta_mean_fidelity_vs_fidelity_only": float(r["mean_fidelity"] - b["mean_fidelity"]),
                "delta_best_fi_vs_fidelity_only": float(r["best_fi"] - b["best_fi"]),
            })
        write_csv(out / "pairwise_advantage_vs_fidelity_only.csv", pairwise)

def main() -> None:
    cfg = parse_args()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    target = Statevector.from_instruction(build_target_circuit(cfg))
    template_f = fidelity(ghz_template(cfg), target, cfg)
    print("QFE Aligned Grid v2")
    print(f"Template GHZ fidelity sanity check: {template_f:.12f}")
    if template_f < 1.0 - 1e-10:
        raise RuntimeError(f"Ansatz is not GHZ-reachable. Template fidelity={template_f}")

    write_experiment_metadata(out, cfg, template_f)
    print(f"n_qubits={cfg.n_qubits} layers={cfg.layers} population={cfg.population} generations={cfg.generations} seed={cfg.seed}")
    print("Important: mean fidelity is the population mean. Best fidelity is the objective optimum. v2 includes late-stage contraction so QFE mean also converges.")

    history: List[dict] = []
    valid = {"FIDELITY_ONLY", "QFE_SCORE", "FLOW_ONLY", "GRADIENT_ONLY", "QFE_UNITARY_FLOW", "RANDOM"}
    for m in cfg.modes:
        if m not in valid:
            raise ValueError(f"Invalid mode {m}. Valid: {sorted(valid)}")
        run_mode(m, cfg, target, out, history)

    write_csv(out / "history.csv", history)
    summary = summarize(history, cfg)
    write_csv(out / "summary_metrics.csv", summary)
    build_comparison_tables(history, cfg, out)
    save_plots(history, out)

    print("\nFinished.")
    print("History:", out / "history.csv")
    print("Summary:", out / "summary_metrics.csv")
    print("Final comparison:", out / "final_comparison.csv")
    print("Threshold times:", out / "threshold_times.csv")
    print("Pairwise advantage:", out / "pairwise_advantage_vs_fidelity_only.csv")
    print("Plots:", out / "plots")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
