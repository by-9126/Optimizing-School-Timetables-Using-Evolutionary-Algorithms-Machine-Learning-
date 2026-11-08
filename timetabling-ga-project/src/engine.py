"""Khung Giải thuật Di truyền tổng quát (chuyển hóa từ Slide 6) & Leo đồi (Slide 2).

Khung gốc trên slide:

    def genetic_algorithm(create, fitness, crossover, mutate,
                          pop_size=100, generations=200, rate=0.02):
        pop = [create() for _ in range(pop_size)]
        for g in range(generations):
            pop.sort(key=fitness, reverse=True)
            new_pop = pop[:2]                        # Tinh hoa (Elitism)
            while len(new_pop) < pop_size:
                p1 = tournament(pop, fitness)        # Chọn lọc đấu giải
                p2 = tournament(pop, fitness)
                child = crossover(p1, p2)            # Lai ghép
                child = mutate(child, rate)          # Đột biến
                new_pop.append(child)
            pop = new_pop
        return max(pop, key=fitness)

Ở đây giữ nguyên cấu trúc đó, chỉ bổ sung: thống kê từng thế hệ, dừng sớm và dạng
generator (`evolve`) để giao diện có thể "chạy từng bước F0 -> F1 -> ...".
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator


def tournament(pop, fitness, k: int = 3, rng: random.Random = random):
    """Bốc ngẫu nhiên k cá thể, lấy cá thể có fitness cao nhất."""
    return max(rng.sample(pop, k), key=fitness)


@dataclass
class GenerationState:
    generation: int
    population: list           # đã sắp xếp giảm dần theo fitness
    best: tuple
    best_fitness: float
    avg_fitness: float
    valid_pct: float            # % cá thể không vi phạm ràng buộc cứng
    best_hard: int
    best_soft: int
    evaluations: int = 0

    def as_record(self) -> dict:
        return {
            "generation": self.generation, "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness, "valid_pct": self.valid_pct,
            "best_hard": self.best_hard, "best_soft": self.best_soft,
            "evaluations": self.evaluations,
        }


def _stats(gen, pop, fitness, evaluate, n_evals) -> GenerationState:
    fits = [fitness(ind) for ind in pop]
    best = pop[0]
    if evaluate is not None:
        evs = [evaluate(ind) for ind in pop]
        valid = 100.0 * sum(1 for e in evs if e.hard == 0) / len(pop)
        bh, bs = evs[0].hard, evs[0].soft
    else:
        valid, bh, bs = float("nan"), -1, -1
    return GenerationState(gen, pop, best, fits[0], sum(fits) / len(fits), valid, bh, bs, n_evals)


def evolve(create: Callable, fitness: Callable, crossover: Callable, mutate: Callable,
           pop_size: int = 200, generations: int = 300, mutation_rate: float = 0.05,
           elitism: int = 2, tournament_k: int = 3, rng: random.Random | None = None,
           evaluate: Callable | None = None, target_fitness: float | None = None,
           patience: int | None = None, eval_counter: Callable[[], int] | None = None,
           initial_population: list | None = None) -> Iterator[GenerationState]:
    """Generator: yield trạng thái F0, F1, ..., cho tới khi hết `generations` hoặc dừng sớm."""
    rng = rng or random.Random()
    tournament_k = max(1, min(tournament_k, pop_size))
    pop = list(initial_population) if initial_population else [create() for _ in range(pop_size)]
    best_so_far, last_improve = float("-inf"), 0
    gen = 0
    while True:
        pop.sort(key=fitness, reverse=True)
        state = _stats(gen, pop, fitness, evaluate, eval_counter() if eval_counter else 0)
        yield state

        if state.best_fitness > best_so_far + 1e-12:
            best_so_far, last_improve = state.best_fitness, gen
        if gen >= generations:
            return
        if target_fitness is not None and state.best_fitness >= target_fitness - 1e-9:
            return
        if patience is not None and gen - last_improve >= patience:
            return

        new_pop = pop[:elitism]                                   # Tinh hoa
        while len(new_pop) < pop_size:
            p1 = tournament(pop, fitness, tournament_k, rng)      # Chọn lọc
            p2 = tournament(pop, fitness, tournament_k, rng)
            child = crossover(p1, p2)                             # Lai ghép
            child = mutate(child, mutation_rate)                  # Đột biến
            new_pop.append(child)
        pop = new_pop
        gen += 1


@dataclass
class RunResult:
    best: tuple
    best_fitness: float
    history: list = field(default_factory=list)
    generation_found: int = 0             # thế hệ đầu tiên đạt fitness tốt nhất cuối cùng
    first_valid_generation: int | None = None   # thế hệ đầu tiên có TKB 0 lỗi cứng
    generations_run: int = 0
    initial_population: list = field(default_factory=list)
    final_population: list = field(default_factory=list)
    elapsed: float = 0.0
    evaluations: int = 0


def run_to_completion(iterator: Iterator[GenerationState],
                      on_generation: Callable[[GenerationState], None] | None = None) -> RunResult:
    t0 = time.perf_counter()
    history, f0, last = [], None, None
    best_fit, found, first_valid = float("-inf"), 0, None
    best_ind = None
    for st in iterator:
        if f0 is None:
            f0 = list(st.population)
        history.append(st.as_record())
        if st.best_fitness > best_fit + 1e-12:
            best_fit, found, best_ind = st.best_fitness, st.generation, st.best
        if first_valid is None and st.best_hard == 0:
            first_valid = st.generation
        if on_generation:
            on_generation(st)
        last = st
    return RunResult(best=best_ind, best_fitness=best_fit, history=history, generation_found=found,
                     first_valid_generation=first_valid, generations_run=last.generation,
                     initial_population=f0, final_population=list(last.population),
                     elapsed=time.perf_counter() - t0, evaluations=last.evaluations)


def genetic_algorithm(create, fitness, crossover, mutate, pop_size=200, generations=300,
                      mutation_rate=0.05, elitism=2, on_generation=None, **kw) -> RunResult:
    """Phiên bản 'một lệnh' giống Slide 6: trả về RunResult (best = max(pop, key=fitness))."""
    it = evolve(create, fitness, crossover, mutate, pop_size=pop_size, generations=generations,
                mutation_rate=mutation_rate, elitism=elitism, **kw)
    return run_to_completion(it, on_generation)


def hill_climbing(create: Callable, fitness: Callable, neighbor: Callable, max_evals: int,
                  report_every: int = 200, patience: int | None = None,
                  evaluate: Callable | None = None, target_fitness: float | None = None,
                  batch_k: int = 12) -> Iterator[GenerationState]:
    """Leo đồi 1 trạng thái (Slide 2): mỗi bước sinh `batch_k` láng giềng ngẫu nhiên,
    chọn láng giềng tốt nhất; chỉ di chuyển nếu điểm CAO HƠN trạng thái hiện tại, nếu không
    thì đã kẹt ở cực trị địa phương (plateau / đỉnh cục bộ) và dừng.
    """
    current = create()
    f_cur = fitness(current)
    evals, no_improve = 1, 0

    def state(gen):
        e = evaluate(current) if evaluate else None
        return GenerationState(gen, [current], current, f_cur, f_cur,
                               (100.0 if e and e.hard == 0 else 0.0) if e else float("nan"),
                               e.hard if e else -1, e.soft if e else -1, evals)

    gen = 0
    yield state(gen)
    while evals < max_evals:
        best_c, best_f = None, float("-inf")
        for _ in range(batch_k):
            cand = neighbor(current)
            f_c = fitness(cand)
            evals += 1
            if f_c > best_f:
                best_c, best_f = cand, f_c
            if evals >= max_evals:
                break
        if best_f > f_cur:
            current, f_cur, no_improve = best_c, best_f, 0
        else:
            no_improve += 1
        stuck = patience is not None and no_improve >= patience
        done = target_fitness is not None and f_cur >= target_fitness - 1e-9
        gen += 1
        yield state(gen)
        if stuck or done:
            return

