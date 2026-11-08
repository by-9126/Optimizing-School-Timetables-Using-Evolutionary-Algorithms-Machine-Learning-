"""Mã hóa nhiễm sắc thể & hàm khởi tạo create().

Cá thể (Individual) = 1 phương án thời khóa biểu hoàn chỉnh trong 1 tuần:

    ind = ((room_1, slot_1), (room_2, slot_2), ..., (room_n, slot_n))

Gen thứ i = buổi học thứ i (lớp học phần + giảng viên + lớp SV đã cố định) được gán
(phòng học, khung giờ). Dùng tuple để cá thể bất biến & băm được (phục vụ cache fitness).
"""
from __future__ import annotations

import random

from .problem import Problem


def random_gene(problem: Problem, e: int, rng: random.Random, smart: bool = True) -> tuple:
    """Sinh 1 gen ngẫu nhiên cho buổi học e. smart=True: chỉ chọn phòng đúng loại & đủ sức chứa."""
    rooms = problem.events[e].suitable_rooms if smart else range(problem.n_rooms)
    return (rng.choice(rooms), rng.randrange(problem.n_slots))


def make_create(problem: Problem, rng: random.Random, smart: bool = True):
    """Trả về hàm create() sinh ngẫu nhiên 1 cá thể (Slide 6)."""
    n = problem.n_events

    def create():
        return tuple(random_gene(problem, e, rng, smart) for e in range(n))

    return create


def decode(problem: Problem, ind) -> list[dict]:
    """Giải mã cá thể thành danh sách lịch học dễ đọc (Output 1)."""
    P = problem
    out = []
    for ev, (r, t) in zip(P.events, ind):
        c = P.courses[ev.course_idx]
        teacher = P.teachers[ev.teacher_idx]
        out.append({
            "course_id": c.id,
            "session": ev.session_no,
            "course_name": c.name,
            "student_group": P.group_names(ev.group_idxs),
            "students_count": ev.students,
            "teacher": teacher.name,
            "room": P.rooms[r].label,
            "timeslot": P.slot_label(t),
            "_room_idx": r,
            "_slot_idx": t,
        })
    out.sort(key=lambda x: (x["_slot_idx"], x["student_group"]))
    return out


def solution_to_dict(problem: Problem, ind, evaluation, generation_found: int | None = None,
                     extra: dict | None = None) -> dict:
    """Output 1: cấu trúc JSON của lời giải tốt nhất."""
    schedule = [{k: v for k, v in item.items() if not k.startswith("_")} for item in decode(problem, ind)]
    d = {
        "generation_found": generation_found,
        "fitness_score": round(evaluation.fitness, 1),
        "penalty": evaluation.penalty,
        "hard_violations": evaluation.hard,
        "soft_violations": evaluation.soft,
        "hard_detail": evaluation.hard_detail,
        "soft_detail": evaluation.soft_detail,
    }
    if extra:
        d.update(extra)
    d["chromosome"] = [list(g) for g in ind]
    d["schedule"] = schedule
    return d

