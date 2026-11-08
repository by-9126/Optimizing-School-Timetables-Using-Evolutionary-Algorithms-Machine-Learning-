"""Lớp tiện ích gom 4 hàm GA + các biến thể thuật toán thành 1 lệnh `solve()`.

Dùng chung cho: main.py (CLI), app.py (giao diện) và experiments/benchmark_30_runs.py.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .chromosome import make_create
from .engine import RunResult, evolve, hill_climbing, run_to_completion
from .fitness import BASE_SCORE, FitnessEvaluator
from .operators import make_crossover, make_guided_mutate, make_neighbor, make_random_mutate
from .problem import Problem

MODES = {
    "ga": "GA Chuẩn (Lai ghép 1 điểm + Đột biến + Tinh hoa)",
    "ga_uniform": "GA + Lai ghép đồng nhất (Uniform Crossover)",
    "guided": "GA + Đột biến định hướng (Guided GA)",
    "ep": "EP (Chỉ Đột biến, không Lai ghép)",
    "hc": "Leo đồi (Hill Climbing - 1 trạng thái)",
}


@dataclass
class GAConfig:
    mode: str = "ga"
    pop_size: int = 200
    generations: int = 300
    mutation_rate: float = 0.05
    elitism: int = 2
    tournament_k: int = 3
    seed: int | None = None
    smart_init: bool = True          # create(): chỉ chọn phòng đủ sức chứa & đúng loại
    stop_on_perfect: bool = True     # dừng khi fitness = 1000 (0 lỗi cứng & 0 lỗi mềm)
    patience: int | None = None      # dừng nếu best không cải thiện sau N thế hệ
    guided_prob: float = 0.5
    guided_candidates: int = 8
    hc_patience: int | None = None   # Leo đồi: số láng giềng liên tiếp không tốt hơn -> coi là kẹt

    def to_dict(self):
        return asdict(self)


def build_components(problem: Problem, cfg: GAConfig):
    """Trả về (rng, evaluator, create, crossover, mutate) theo chế độ thuật toán."""
    rng = random.Random(cfg.seed)
    evaluator = FitnessEvaluator(problem)
    create = make_create(problem, rng, cfg.smart_init)
    if cfg.mode == "guided":
        crossover = make_crossover("single", rng)
        mutate = make_guided_mutate(problem, rng, cfg.guided_prob, cfg.guided_candidates, cfg.smart_init)
    elif cfg.mode == "ga_uniform":
        crossover = make_crossover("uniform", rng)
        mutate = make_random_mutate(problem, rng, cfg.smart_init)
    elif cfg.mode == "ep":
        crossover = make_crossover("none", rng)
        mutate = make_random_mutate(problem, rng, cfg.smart_init)
    else:  # "ga" & "hc"
        crossover = make_crossover("single", rng)
        mutate = make_random_mutate(problem, rng, cfg.smart_init)
    return rng, evaluator, create, crossover, mutate


def make_iterator(problem: Problem, cfg: GAConfig):
    """Tạo generator tiến hóa (F0, F1, ...) + evaluator. Dùng cho chế độ chạy từng bước."""
    if cfg.mode not in MODES:
        raise ValueError(f"Chế độ không hợp lệ: {cfg.mode}. Chọn một trong {list(MODES)}")
    rng, evaluator, create, crossover, mutate = build_components(problem, cfg)
    target = BASE_SCORE if cfg.stop_on_perfect else None

    if cfg.mode == "hc":
        neighbor = make_neighbor(problem, rng, cfg.smart_init)
        hc_patience = cfg.hc_patience if cfg.hc_patience is not None else 2
        it = hill_climbing(create, evaluator, neighbor, max_evals=cfg.pop_size * cfg.generations,
                           report_every=cfg.pop_size, patience=hc_patience,
                           evaluate=evaluator.evaluate, target_fitness=target, batch_k=14)
    else:
        elitism = 0 if cfg.mode == "ep" else cfg.elitism
        k = 2 if cfg.mode == "ep" else cfg.tournament_k
        it = evolve(create, evaluator, crossover, mutate, pop_size=cfg.pop_size,
                    generations=cfg.generations, mutation_rate=cfg.mutation_rate,
                    elitism=elitism, tournament_k=k, rng=rng,
                    evaluate=evaluator.evaluate, target_fitness=target, patience=cfg.patience,
                    eval_counter=lambda: evaluator.n_evals)
    return it, evaluator


def solve(problem: Problem, cfg: GAConfig, on_generation=None) -> tuple[RunResult, FitnessEvaluator]:
    it, evaluator = make_iterator(problem, cfg)
    result = run_to_completion(it, on_generation)
    return result, evaluator

