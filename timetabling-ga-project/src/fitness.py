"""Hàm đánh giá độ thích nghi (fitness) & cơ chế phạt.

Ràng buộc CỨNG (W_HARD = 100 điểm / lỗi):
    1. Trùng phòng học      : 2 buổi khác nhau cùng phòng r, cùng khung giờ t.
    2. Trùng giảng viên     : 1 giảng viên dạy 2 buổi cùng khung giờ t.
    3. Trùng lớp sinh viên  : 1 lớp sinh viên học 2 buổi cùng khung giờ t.
    4. Sức chứa / loại phòng: sĩ số > sức chứa hoặc môn thực hành xếp vào phòng lý thuyết (và ngược lại).
    (Một ô có k buổi chồng nhau được tính k-1 lỗi.)

Ràng buộc MỀM (W_SOFT = 5 - 10 điểm / lỗi):
    1. consecutive  (10): giảng viên / lớp SV học-dạy QUÁ 3 ca liên tiếp trong 1 ngày.
    2. same_day     (10): các buổi của cùng 1 lớp học phần bị dồn vào cùng 1 ngày.
    3. teacher_busy (10): buổi học rơi vào khung giờ giảng viên đăng ký bận.
    4. student_gap  ( 5): lớp SV bị trống ca (lủng ca) giữa các buổi học trong cùng 1 ngày.

Fitness (thang 1000, luôn > 0 để giữ hướng tiến hóa - "phạt nhẹ thay vì cho 0 điểm", Slide 13):

    Fitness(ind) = 1000 * 1000 / (1000 + 100 * số_lỗi_cứng + tổng_điểm_phạt_mềm)

    -> 1000 điểm khi hoàn hảo; ví dụ 0 lỗi cứng + 15 điểm phạt mềm = 985.2 điểm.
"""
from __future__ import annotations

from typing import NamedTuple

from .problem import Problem

BASE_SCORE = 1000.0
W_HARD = 100
SOFT_WEIGHTS = {"consecutive": 10, "same_day": 10, "teacher_busy": 10, "student_gap": 5}
MAX_CONSECUTIVE = 3

HARD_LABELS = {
    "room": "Trùng phòng học cùng ca",
    "teacher": "Trùng giảng viên cùng ca",
    "group": "Trùng lớp sinh viên cùng ca",
    "capacity": "Vượt sức chứa / Sai loại phòng học",
}
SOFT_LABELS = {
    "consecutive": f"GV/SV học quá {MAX_CONSECUTIVE} ca liên tiếp",
    "same_day": "Các buổi cùng môn dồn chung 1 ngày",
    "teacher_busy": "Xếp vào giờ giảng viên đăng ký bận",
    "student_gap": "Sinh viên bị lủng ca trong ngày",
}


class Evaluation(NamedTuple):
    fitness: float
    penalty: int
    hard: int            # tổng số lỗi cứng
    soft: int            # tổng số lỗi mềm (đếm số lần vi phạm)
    soft_penalty: int    # tổng điểm phạt mềm
    hard_detail: dict
    soft_detail: dict

    @property
    def is_valid(self) -> bool:
        return self.hard == 0


def penalty_to_fitness(penalty: float) -> float:
    return BASE_SCORE * BASE_SCORE / (BASE_SCORE + penalty)


def _build_day_pattern_table(n_shifts: int):
    """Tra cứu nhanh (số đoạn > MAX_CONSECUTIVE ca liên tiếp, số ca bị lủng) theo bitmask 1 ngày."""
    table = []
    for mask in range(1 << n_shifts):
        occ = [(mask >> s) & 1 for s in range(n_shifts)]
        consec_viol, run = 0, 0
        for o in occ + [0]:
            if o:
                run += 1
            else:
                if run > MAX_CONSECUTIVE:
                    consec_viol += 1
                run = 0
        gaps = 0
        if mask:
            first = occ.index(1)
            last = n_shifts - 1 - occ[::-1].index(1)
            gaps = sum(1 for s in range(first, last + 1) if not occ[s])
        table.append((consec_viol, gaps))
    return table


def evaluate(problem: Problem, ind) -> Evaluation:
    """Đánh giá chi tiết 1 cá thể ind = [(room, slot), ...] (độ dài = số buổi học)."""
    P = problem
    T, S, D = P.n_slots, P.n_shifts, P.n_days
    table = _pattern_table(P)

    room_cnt = [0] * (P.n_rooms * T)
    teacher_cnt = [0] * (P.n_teachers * T)
    group_cnt = [0] * (P.n_groups * T)
    h_room = h_teacher = h_group = h_cap = 0
    s_busy = 0
    teachers = P.teachers

    for ev, (r, t) in zip(P.events, ind):
        k = r * T + t
        if room_cnt[k]:
            h_room += 1
        room_cnt[k] += 1

        k = ev.teacher_idx * T + t
        if teacher_cnt[k]:
            h_teacher += 1
        teacher_cnt[k] += 1

        for g in ev.group_idxs:
            k = g * T + t
            if group_cnt[k]:
                h_group += 1
            group_cnt[k] += 1

        if r not in ev.suitable_set:
            h_cap += 1
        if t in teachers[ev.teacher_idx].busy_slots:
            s_busy += 1

    # Ràng buộc mềm 2: dồn cùng ngày
    s_same = 0
    for evs in P.multi_session_courses:
        days = [ind[i][1] // S for i in evs]
        s_same += len(days) - len(set(days))

    # Ràng buộc mềm 1 & 4: quá nhiều ca liên tiếp / lủng ca
    s_consec = s_gap = 0
    for g in range(P.n_groups):
        base = g * T
        for d in range(D):
            mask = 0
            off = base + d * S
            for s in range(S):
                if group_cnt[off + s]:
                    mask |= 1 << s
            if mask:
                c, gp = table[mask]
                s_consec += c
                s_gap += gp
    for tch in range(P.n_teachers):
        base = tch * T
        for d in range(D):
            mask = 0
            off = base + d * S
            for s in range(S):
                if teacher_cnt[off + s]:
                    mask |= 1 << s
            if mask:
                s_consec += table[mask][0]

    hard = h_room + h_teacher + h_group + h_cap
    soft_detail = {"consecutive": s_consec, "same_day": s_same, "teacher_busy": s_busy, "student_gap": s_gap}
    soft_pen = sum(SOFT_WEIGHTS[k] * v for k, v in soft_detail.items())
    penalty = W_HARD * hard + soft_pen
    return Evaluation(
        fitness=penalty_to_fitness(penalty), penalty=penalty, hard=hard,
        soft=sum(soft_detail.values()), soft_penalty=soft_pen,
        hard_detail={"room": h_room, "teacher": h_teacher, "group": h_group, "capacity": h_cap},
        soft_detail=soft_detail,
    )


_TABLE_CACHE: dict = {}


def _pattern_table(P: Problem):
    tb = _TABLE_CACHE.get(P.n_shifts)
    if tb is None:
        tb = _TABLE_CACHE[P.n_shifts] = _build_day_pattern_table(P.n_shifts)
    return tb


def find_conflicts(problem: Problem, ind) -> set:
    """Tập chỉ số các buổi học đang dính ràng buộc CỨNG (dùng cho Guided Mutation & tô đỏ giao diện)."""
    P = problem
    cells: dict = {}
    bad = set()
    for ev, (r, t) in zip(P.events, ind):
        cells.setdefault(("r", r, t), []).append(ev.idx)
        cells.setdefault(("t", ev.teacher_idx, t), []).append(ev.idx)
        for g in ev.group_idxs:
            cells.setdefault(("g", g, t), []).append(ev.idx)
        if r not in ev.suitable_set:
            bad.add(ev.idx)
    for evs in cells.values():
        if len(evs) > 1:
            bad.update(evs)
    return bad


def find_soft_issues(problem: Problem, ind) -> set:
    """Tập buổi học vi phạm ràng buộc mềm có thể sửa cục bộ (giờ bận GV, dồn cùng ngày)."""
    P = problem
    S = P.n_shifts
    out = set()
    for ev, (r, t) in zip(P.events, ind):
        if t in P.teachers[ev.teacher_idx].busy_slots:
            out.add(ev.idx)
    for evs in P.multi_session_courses:
        seen = {}
        for i in evs:
            d = ind[i][1] // S
            if d in seen:
                out.add(i)
                out.add(seen[d])
            seen[d] = i
    return out


def conflict_details(problem: Problem, ind) -> list[str]:
    """Mô tả bằng lời từng xung đột cứng (phục vụ báo cáo & trace F0 -> F1)."""
    P = problem
    msgs = []
    cells: dict = {}
    for ev, (r, t) in zip(P.events, ind):
        cells.setdefault(("Phòng", P.rooms[r].name, t), []).append(ev.idx)
        cells.setdefault(("GV", P.teachers[ev.teacher_idx].name, t), []).append(ev.idx)
        for g in ev.group_idxs:
            cells.setdefault(("Lớp SV", P.groups[g].name, t), []).append(ev.idx)
        if r not in ev.suitable_set:
            room = P.rooms[r]
            msgs.append(f"Sức chứa/loại phòng: {P.event_code(ev.idx)} ({ev.students} SV, {ev.room_type}) "
                        f"-> {room.name} ({room.capacity} chỗ, {room.type})")
    for (kind, name, t), evs in cells.items():
        if len(evs) > 1:
            codes = ", ".join(P.event_code(e) for e in evs)
            msgs.append(f"Trùng {kind} {name} lúc {P.slot_label(t, short=True)}: {codes}")
    return msgs


class FitnessEvaluator:
    """Bọc hàm evaluate() kèm bộ nhớ đệm (cache) để không tính lại cá thể trùng (tinh hoa, đấu giải)."""

    def __init__(self, problem: Problem, max_cache: int = 200_000):
        self.problem = problem
        self.max_cache = max_cache
        self._cache: dict = {}
        self.n_evals = 0  # số lần đánh giá thực sự (không tính cache hit)

    def evaluate(self, ind) -> Evaluation:
        key = tuple(ind)
        res = self._cache.get(key)
        if res is None:
            if len(self._cache) >= self.max_cache:
                self._cache.clear()
            res = evaluate(self.problem, key)
            self._cache[key] = res
            self.n_evals += 1
        return res

    def __call__(self, ind) -> float:
        """fitness(ind) đúng chữ ký trong khung giải thuật Slide 6."""
        return self.evaluate(ind).fitness


def validation_report(problem: Problem, ind, evaluation: Evaluation | None = None) -> str:
    ev = evaluation or evaluate(problem, ind)
    lines = ["=" * 80, "BÁO CÁO KIỂM TRA CHẤT LƯỢNG THỜI KHÓA BIỂU TỐI ƯU:"]
    for k, label in HARD_LABELS.items():
        v = ev.hard_detail[k]
        status = "ĐẠT ✓" if v == 0 else "KHÔNG ĐẠT ✗"
        lines.append(f"- {label:<36}: {v} lỗi ({status})")
    for k, label in SOFT_LABELS.items():
        v = ev.soft_detail[k]
        lines.append(f"- [Mềm] {label:<36}: {v} lỗi nhỏ (-{v * SOFT_WEIGHTS[k]} điểm)")
    lines.append(f"- Tổng điểm phạt: {ev.penalty}  =>  Fitness = {ev.fitness:.1f} / {BASE_SCORE:.0f}")
    if ev.hard == 0:
        lines.append("=> KẾT LUẬN: Thời khóa biểu hoàn toàn khả thi để áp dụng thực tế!")
    else:
        lines.append(f"=> KẾT LUẬN: Còn {ev.hard} xung đột cứng, cần chạy thêm thế hệ / tăng quần thể.")
    lines.append("=" * 80)
    return "\n".join(lines)

