"""Minh họa chi tiết 1 vòng tiến hóa F0 -> F1 (theo phong cách Slide 14-23).

4 cá thể TKB ở F0 -> chọn lọc đấu giải (k=3) -> lai ghép cắt 1 điểm -> đột biến -> F1
(2 cá thể tinh hoa + 2 cá thể con).
"""
from __future__ import annotations

import random

from .chromosome import make_create
from .fitness import BASE_SCORE, FitnessEvaluator, conflict_details, find_conflicts
from .operators import mutate_gene
from .problem import Problem

NAMES = ["A", "B", "C", "D"]


def gene_str(problem: Problem, gene) -> str:
    r, t = gene
    return f"{problem.rooms[r].name}@{problem.slot_label(t, short=True)}"


def build_f0_f1_trace(problem: Problem, seed: int = 11, mutation_rate: float = 0.05,
                      tournament_k: int = 3, max_attempts: int = 500) -> dict:
    """Tìm 1 ví dụ minh họa (điểm cắt vẫn ngẫu nhiên) trong đó Con sau lai ghép tốt hơn cả Cha lẫn Mẹ."""
    P = problem
    chosen = None
    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        ev = FitnessEvaluator(P)
        create = make_create(P, rng)
        f0 = [create() for _ in range(4)]
        fit = [ev(x) for x in f0]

        t1 = sorted(rng.sample(range(4), tournament_k))
        p1 = max(t1, key=lambda i: fit[i])
        t2 = sorted(rng.sample(range(4), tournament_k))
        p2 = max(t2, key=lambda i: fit[i])
        if p1 == p2:
            continue
        n = P.n_events
        cut = rng.randint(1, n - 1)
        child = f0[p1][:cut] + f0[p2][cut:]
        child2 = f0[p2][:cut] + f0[p1][cut:]
        ok = ev(child) > max(fit[p1], fit[p2])
        if not ok and attempt < max_attempts - 1:
            continue

        # Đột biến từng gen với xác suất mutation_rate (đảm bảo có ít nhất 1 gen để minh họa)
        mutations = []
        genes = list(child)
        for i in range(n):
            if rng.random() < mutation_rate:
                new = mutate_gene(P, i, genes[i], rng)
                mutations.append((i, genes[i], new))
                genes[i] = new
        if not mutations:
            i = rng.randrange(n)
            new = mutate_gene(P, i, genes[i], rng)
            mutations.append((i, genes[i], new))
            genes[i] = new
        child_mut = tuple(genes)

        f0_sorted = sorted(range(4), key=lambda i: fit[i], reverse=True)
        elites = f0_sorted[:2]
        f1 = [("Tinh hoa " + NAMES[i], f0[i]) for i in elites] + [("Con 1 (Cha|Mẹ)", child_mut),
                                                                   ("Con 2 (Mẹ|Cha)", child2)]
        chosen = dict(
            attempt=attempt, seed=seed + attempt, cut=cut, f0=f0,
            f0_eval=[ev.evaluate(x) for x in f0],
            f0_conflicts=[find_conflicts(P, x) for x in f0],
            f0_details=[conflict_details(P, x) for x in f0],
            tournaments=[(t1, p1), (t2, p2)], p1=p1, p2=p2,
            child=child, child_eval=ev.evaluate(child), child_conflicts=find_conflicts(P, child),
            mutations=mutations, child_mut=child_mut, child_mut_eval=ev.evaluate(child_mut),
            child_mut_conflicts=find_conflicts(P, child_mut),
            child2=child2, child2_eval=ev.evaluate(child2),
            f1=[(name, ind, ev.evaluate(ind)) for name, ind in f1],
        )
        break
    return chosen


def trace_to_text(problem: Problem, tr: dict) -> str:
    P = problem
    L = []
    w = L.append
    w("=" * 100)
    w("MINH HỌA 1 VÒNG TIẾN HÓA F0 -> F1 CHO BÀI TOÁN XẾP THỜI KHÓA BIỂU (theo mẫu Slide 14-23)")
    w("=" * 100)
    w(f"Dữ liệu: {P.summary()}")
    w(f"Mỗi cá thể = {P.n_events} gen, gen i = (Phòng, Khung giờ) của buổi học thứ i. Ký hiệu: Phòng@Thứ-Ca, "
      f"vd V.A201@T2-C1 = phòng V.A201, Thứ 2, Ca 1.")
    w(f"Fitness = {BASE_SCORE:.0f} x {BASE_SCORE:.0f} / ({BASE_SCORE:.0f} + 100 x lỗi_cứng + điểm_phạt_mềm)")
    w("")
    w("BƯỚC 1 - KHỞI TẠO QUẦN THỂ F0 (4 cá thể ngẫu nhiên, đã lọc phòng đủ sức chứa)")
    w("-" * 100)
    for i, (ev, det) in enumerate(zip(tr["f0_eval"], tr["f0_details"])):
        w(f"  TKB {NAMES[i]}: Fitness = {ev.fitness:6.1f} | Lỗi cứng = {ev.hard:2d} "
          f"(phòng {ev.hard_detail['room']}, GV {ev.hard_detail['teacher']}, lớp SV {ev.hard_detail['group']}, "
          f"sức chứa {ev.hard_detail['capacity']}) | Lỗi mềm = {ev.soft} (-{ev.soft_penalty} điểm)")
        for m in det[:6]:
            w(f"      ⚠ {m}")
        if len(det) > 6:
            w(f"      ... và {len(det) - 6} xung đột khác")
    avg0 = sum(e.fitness for e in tr["f0_eval"]) / 4
    w(f"  => F0: Fitness tốt nhất = {max(e.fitness for e in tr['f0_eval']):.1f}, trung bình = {avg0:.1f}")
    w("")
    w("BƯỚC 2 - CHỌN LỌC ĐẤU GIẢI (tournament, k = 3)")
    w("-" * 100)
    for k, (t, win) in enumerate(tr["tournaments"], 1):
        cand = ", ".join(f"{NAMES[i]}({tr['f0_eval'][i].fitness:.1f})" for i in t)
        w(f"  Lượt {k}: bốc ngẫu nhiên [{cand}] -> thắng: TKB {NAMES[win]}")
    w(f"  => Cha = TKB {NAMES[tr['p1']]}, Mẹ = TKB {NAMES[tr['p2']]}")
    w("")
    cut = tr["cut"]
    w(f"BƯỚC 3 - LAI GHÉP CẮT 1 ĐIỂM tại vị trí cut = {cut}: Con = Cha[0:{cut}] + Mẹ[{cut}:{P.n_events}]")
    w("BƯỚC 4 - ĐỘT BIẾN (xác suất 5%/gen): đổi Phòng hoặc Ca học của gen được chọn")
    w("-" * 100)
    father, mother = tr["f0"][tr["p1"]], tr["f0"][tr["p2"]]
    cf, cm = tr["f0_conflicts"][tr["p1"]], tr["f0_conflicts"][tr["p2"]]
    cc, ccm = tr["child_conflicts"], tr["child_mut_conflicts"]
    mut_pos = {i: (old, new) for i, old, new in tr["mutations"]}
    hdr = f"  {'#':>3} {'Buổi học':<13} {'Cha (' + NAMES[tr['p1']] + ')':<20} {'Mẹ (' + NAMES[tr['p2']] + ')':<20} " \
          f"{'Con (sau lai ghép)':<22} {'Con (sau đột biến)':<22}"
    w(hdr)
    for i in range(P.n_events):
        if i == cut:
            w("  " + "-" * 40 + f" ✂ ĐIỂM CẮT (cut = {cut}) " + "-" * 36)

        def cell(ind, conf, mark=""):
            return (("⚠" if i in conf else " ") + gene_str(P, ind[i]) + mark)

        src = "←Cha" if i < cut else "←Mẹ"
        mut = " ★" if i in mut_pos else ""
        w(f"  {i:>3} {P.event_code(i):<13} {cell(father, cf):<20} {cell(mother, cm):<20} "
          f"{(cell(tr['child'], cc) + ' ' + src):<22} {cell(tr['child_mut'], ccm, mut):<22}")
    w("  (⚠ = gen đang dính xung đột cứng; ★ = gen bị đột biến)")
    for i, old, new in tr["mutations"]:
        w(f"  Đột biến gen {i} ({P.event_code(i)}): {gene_str(P, old)} -> {gene_str(P, new)}")
    w("")
    ef, em = tr["f0_eval"][tr["p1"]], tr["f0_eval"][tr["p2"]]
    ec, ecm = tr["child_eval"], tr["child_mut_eval"]
    w("BƯỚC 5 - SO SÁNH CHA, MẸ VÀ CON")
    w("-" * 100)
    w(f"  Cha {NAMES[tr['p1']]}       : Fitness {ef.fitness:6.1f} | lỗi cứng {ef.hard:2d} | lỗi mềm {ef.soft}")
    w(f"  Mẹ  {NAMES[tr['p2']]}       : Fitness {em.fitness:6.1f} | lỗi cứng {em.hard:2d} | lỗi mềm {em.soft}")
    w(f"  Con (lai ghép) : Fitness {ec.fitness:6.1f} | lỗi cứng {ec.hard:2d} | lỗi mềm {ec.soft}"
      + ("   <-- TỐT HƠN CẢ CHA LẪN MẸ!" if ec.fitness > max(ef.fitness, em.fitness) else ""))
    w(f"  Con (đột biến) : Fitness {ecm.fitness:6.1f} | lỗi cứng {ecm.hard:2d} | lỗi mềm {ecm.soft}")
    w("  Giải thích: nửa đầu của Cha và nửa sau của Mẹ là các 'khối gen' ít xung đột; ghép lại loại bỏ "
      "được các cặp trùng lịch nằm ở nửa kém của mỗi bên.")
    w("")
    w("BƯỚC 6 - QUẦN THỂ F1 (Tinh hoa giữ 2 cá thể tốt nhất F0 + 2 cá thể con)")
    w("-" * 100)
    for name, _, ev in tr["f1"]:
        w(f"  {name:<18}: Fitness {ev.fitness:6.1f} | lỗi cứng {ev.hard:2d} | lỗi mềm {ev.soft}")
    avg1 = sum(e.fitness for _, _, e in tr["f1"]) / len(tr["f1"])
    best1 = max(e.fitness for _, _, e in tr["f1"])
    w(f"  => F1: Fitness tốt nhất = {best1:.1f}, trung bình = {avg1:.1f}  "
      f"(F0: {max(e.fitness for e in tr['f0_eval']):.1f} / {avg0:.1f})")
    w("=" * 100)
    return "\n".join(L)

