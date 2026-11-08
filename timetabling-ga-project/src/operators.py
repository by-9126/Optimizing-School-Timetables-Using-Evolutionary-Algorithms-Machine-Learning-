"""Toán tử di truyền: crossover() & mutate() (kèm Guided Mutation / Repair Heuristic)."""
from __future__ import annotations

import random

from .chromosome import random_gene
from .fitness import SOFT_WEIGHTS, W_HARD, find_conflicts, find_soft_issues
from .problem import Problem


# ============================================================== LAI GHÉP (CROSSOVER)
def single_point_crossover(p1, p2, rng: random.Random, return_cut: bool = False):
    """Cắt tại 1 điểm: lịch các buổi [0, cut) lấy từ Cha, [cut, n) lấy từ Mẹ."""
    n = len(p1)
    cut = rng.randint(1, n - 1)
    child = p1[:cut] + p2[cut:]
    return (child, cut) if return_cut else child


def two_point_crossover(p1, p2, rng: random.Random):
    n = len(p1)
    a, b = sorted(rng.sample(range(1, n), 2))
    return p1[:a] + p2[a:b] + p1[b:]


def uniform_crossover(p1, p2, rng: random.Random):
    """Mỗi gen chọn từ Cha hoặc Mẹ với xác suất 50/50."""
    return tuple(g1 if rng.random() < 0.5 else g2 for g1, g2 in zip(p1, p2))


def no_crossover(p1, p2, rng: random.Random):
    """Dùng cho biến thể EP (chỉ đột biến): con = bản sao của Cha."""
    return p1


CROSSOVERS = {
    "single": single_point_crossover,
    "two_point": two_point_crossover,
    "uniform": uniform_crossover,
    "none": no_crossover,
}


def make_crossover(kind: str, rng: random.Random):
    fn = CROSSOVERS[kind]
    return lambda p1, p2: fn(p1, p2, rng)


# ============================================================== ĐỘT BIẾN (MUTATION)
def mutate_gene(problem: Problem, e: int, gene: tuple, rng: random.Random, smart: bool = True) -> tuple:
    """Đổi ngẫu nhiên Ca học (Thứ/Tiết) HOẶC Phòng học của 1 buổi sang giá trị hợp lệ khác."""
    r, t = gene
    rooms = problem.events[e].suitable_rooms if smart else range(problem.n_rooms)
    if len(rooms) > 1 and rng.random() < 0.5:
        nr = rng.choice(rooms)
        while nr == r:
            nr = rng.choice(rooms)
        return (nr, t)
    nt = rng.randrange(problem.n_slots - 1)
    if nt >= t:
        nt += 1
    return (r, nt)


def make_random_mutate(problem: Problem, rng: random.Random, smart: bool = True):
    """mutate(ind, rate): với xác suất rate trên từng gen, đổi phòng hoặc ca học."""
    n = problem.n_events

    def mutate(ind, rate):
        child = None
        for i in range(n):
            if rng.random() < rate:
                if child is None:
                    child = list(ind)
                child[i] = mutate_gene(problem, i, child[i], rng, smart)
        return ind if child is None else tuple(child)

    return mutate


def make_neighbor(problem: Problem, rng: random.Random, smart: bool = True):
    """Láng giềng cho Leo đồi: đổi đúng 1 gen ngẫu nhiên."""
    n = problem.n_events

    def neighbor(ind):
        i = rng.randrange(n)
        child = list(ind)
        child[i] = mutate_gene(problem, i, child[i], rng, smart)
        return tuple(child)

    return neighbor


class _Occupancy:
    """Bảng đếm chiếm dụng (phòng/GV/lớp SV/môn-ngày) để tính chi phí cục bộ khi sửa 1 gen."""

    def __init__(self, problem: Problem, ind):
        P = self.P = problem
        T = P.n_slots
        self.room = [0] * (P.n_rooms * T)
        self.teacher = [0] * (P.n_teachers * T)
        self.group = [0] * (P.n_groups * T)
        self.course_day = [0] * (len(P.courses) * P.n_days)
        for e, g in enumerate(ind):
            self.update(e, g, +1)

    def update(self, e, gene, delta):
        P = self.P
        T = P.n_slots
        ev = P.events[e]
        r, t = gene
        self.room[r * T + t] += delta
        self.teacher[ev.teacher_idx * T + t] += delta
        for g in ev.group_idxs:
            self.group[g * T + t] += delta
        self.course_day[ev.course_idx * P.n_days + t // P.n_shifts] += delta

    def cost(self, e, gene) -> int:
        """Chi phí khi đặt buổi e vào gene (giả định e đã được gỡ khỏi bảng)."""
        P = self.P
        T = P.n_slots
        ev = P.events[e]
        r, t = gene
        clashes = self.room[r * T + t] + self.teacher[ev.teacher_idx * T + t]
        for g in ev.group_idxs:
            clashes += self.group[g * T + t]
        if r not in ev.suitable_set:
            clashes += 1
        c = W_HARD * clashes
        if t in P.teachers[ev.teacher_idx].busy_slots:
            c += SOFT_WEIGHTS["teacher_busy"]
        c += SOFT_WEIGHTS["same_day"] * self.course_day[ev.course_idx * P.n_days + t // P.n_shifts]
        return c


def make_guided_mutate(problem: Problem, rng: random.Random, guided_prob: float = 0.5,
                       n_candidates: int = 8, smart: bool = True):
    """Đột biến thông minh (Guided Mutation) + sửa lỗi nhẹ (Repair Heuristic).

    1. Đột biến ngẫu nhiên thông thường với xác suất `rate` trên từng gen.
    2. Tìm các buổi đang dính xung đột CỨNG (trùng phòng / GV / lớp SV / sức chứa).
       Nếu không còn lỗi cứng -> nhắm vào các buổi vi phạm mềm (giờ bận GV, dồn cùng ngày).
    3. Mỗi buổi mục tiêu (xác suất guided_prob) được thử `n_candidates` cặp (phòng, ca) ngẫu nhiên
       và giữ lại phương án có chi phí cục bộ thấp nhất (luôn so với gen hiện tại -> không làm tệ đi).
    """
    random_mutate = make_random_mutate(problem, rng, smart)

    def mutate(ind, rate):
        child = random_mutate(ind, rate)
        targets = find_conflicts(problem, child)
        if not targets:
            targets = find_soft_issues(problem, child)
            if not targets:
                return child
        targets = [e for e in targets if rng.random() < guided_prob]
        if not targets:
            return child
        rng.shuffle(targets)
        genes = list(child)
        occ = _Occupancy(problem, genes)
        for e in targets:
            cur = genes[e]
            occ.update(e, cur, -1)
            best, best_cost = cur, occ.cost(e, cur)
            for _ in range(n_candidates):
                cand = random_gene(problem, e, rng, smart)
                c = occ.cost(e, cand)
                if c < best_cost:
                    best, best_cost = cand, c
            genes[e] = best
            occ.update(e, best, +1)
        return tuple(genes)

    return mutate

