"""Mô hình dữ liệu bài toán Xếp Thời Khóa Biểu (School Timetabling Problem).

Đọc 5 file JSON trong thư mục ``data/`` và chuẩn hóa thành các chỉ số nguyên
để các hàm GA (create / fitness / crossover / mutate) chạy nhanh.

Khái niệm chính:
    * Lớp học phần (Course)  : môn học + giảng viên + (các) lớp sinh viên + số buổi/tuần.
    * Buổi học (Event)       : mỗi buổi của một lớp học phần = 1 GEN trong nhiễm sắc thể.
    * Khung giờ (Timeslot)   : t = day * n_shifts + shift  (6 ngày x 4 ca = 24 khung giờ).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ROOM_TYPE_LABEL = {"LT": "Phòng Lý thuyết", "TH": "Phòng Máy"}


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    capacity: int
    type: str  # "LT" (lý thuyết) | "TH" (thực hành / phòng máy)

    @property
    def label(self) -> str:
        return f"{self.name} (Sức chứa: {self.capacity} - {ROOM_TYPE_LABEL.get(self.type, self.type)})"


@dataclass(frozen=True)
class Teacher:
    id: str
    name: str
    short: str
    busy_slots: frozenset  # tập khung giờ t giảng viên đăng ký bận


@dataclass(frozen=True)
class StudentGroup:
    id: str
    name: str
    size: int


@dataclass(frozen=True)
class Course:
    id: str
    name: str
    teacher_idx: int
    group_idxs: tuple
    room_type: str
    sessions: int
    students: int


@dataclass(frozen=True)
class Event:
    """Một buổi học cần xếp lịch = 1 gen."""
    idx: int
    course_idx: int
    session_no: int          # buổi thứ mấy trong tuần của lớp học phần (1, 2, ...)
    teacher_idx: int
    group_idxs: tuple
    students: int
    room_type: str
    suitable_rooms: tuple    # phòng đúng loại & đủ sức chứa (dùng để khởi tạo thông minh)
    suitable_set: frozenset


class Problem:
    """Toàn bộ dữ liệu đầu vào đã được chỉ số hóa."""

    def __init__(self, rooms, teachers, groups, courses_raw, days, shifts, name="Bộ dữ liệu"):
        self.name = name
        self.rooms: list[Room] = rooms
        self.teachers: list[Teacher] = teachers
        self.groups: list[StudentGroup] = groups
        self.days: list[str] = days
        self.shifts: list[dict] = shifts

        self.n_days = len(days)
        self.n_shifts = len(shifts)
        self.n_slots = self.n_days * self.n_shifts
        self.n_rooms = len(rooms)
        self.n_teachers = len(teachers)
        self.n_groups = len(groups)

        t_index = {t.id: i for i, t in enumerate(teachers)}
        g_index = {g.id: i for i, g in enumerate(groups)}

        self.courses: list[Course] = []
        for c in courses_raw:
            if c["teacher"] not in t_index:
                raise ValueError(f"Lớp học phần {c['id']}: không tìm thấy giảng viên {c['teacher']}")
            g_idxs = []
            for gid in c["groups"]:
                if gid not in g_index:
                    raise ValueError(f"Lớp học phần {c['id']}: không tìm thấy lớp sinh viên {gid}")
                g_idxs.append(g_index[gid])
            students = c.get("students") or sum(groups[g].size for g in g_idxs)
            self.courses.append(Course(
                id=c["id"], name=c["name"], teacher_idx=t_index[c["teacher"]],
                group_idxs=tuple(g_idxs), room_type=c.get("room_type", "LT"),
                sessions=int(c.get("sessions_per_week", 1)), students=int(students),
            ))

        self.events: list[Event] = []
        self.warnings: list[str] = []
        for ci, c in enumerate(self.courses):
            suitable = [ri for ri, r in enumerate(rooms) if r.type == c.room_type and r.capacity >= c.students]
            if not suitable:
                self.warnings.append(
                    f"Không có phòng {c.room_type} nào đủ {c.students} chỗ cho {c.id} -> sẽ luôn bị phạt sức chứa.")
                suitable = [ri for ri, r in enumerate(rooms) if r.type == c.room_type] or list(range(len(rooms)))
            for s in range(c.sessions):
                self.events.append(Event(
                    idx=len(self.events), course_idx=ci, session_no=s + 1,
                    teacher_idx=c.teacher_idx, group_idxs=c.group_idxs, students=c.students,
                    room_type=c.room_type, suitable_rooms=tuple(suitable),
                    suitable_set=frozenset(suitable),
                ))
        self.n_events = len(self.events)

        # Danh sách buổi học theo từng lớp học phần (để kiểm tra "dồn cùng ngày").
        self.course_events: list[tuple] = [tuple(e.idx for e in self.events if e.course_idx == ci)
                                           for ci in range(len(self.courses))]
        self.multi_session_courses = [evs for evs in self.course_events if len(evs) > 1]

    # ------------------------------------------------------------------ helpers
    def day_of(self, t: int) -> int:
        return t // self.n_shifts

    def shift_of(self, t: int) -> int:
        return t % self.n_shifts

    def slot_index(self, day: int, shift: int) -> int:
        return day * self.n_shifts + shift

    def slot_label(self, t: int, short: bool = False) -> str:
        d, s = divmod(t, self.n_shifts)
        sh = self.shifts[s]
        if short:
            return f"{self.days[d].replace('Thứ ', 'T')}-C{s + 1}"
        return f"{self.days[d]} - {sh['name']} ({sh['periods']})"

    def event_course(self, e: int) -> Course:
        return self.courses[self.events[e].course_idx]

    def event_code(self, e: int) -> str:
        ev = self.events[e]
        c = self.courses[ev.course_idx]
        return f"{c.id}#{ev.session_no}" if c.sessions > 1 else c.id

    def group_names(self, group_idxs) -> str:
        return ", ".join(self.groups[g].name for g in group_idxs)

    def summary(self) -> str:
        n_lab = sum(1 for e in self.events if e.room_type == "TH")
        return (f"{len(self.courses)} lớp học phần ({self.n_events} buổi/tuần, {n_lab} buổi thực hành) | "
                f"{self.n_teachers} giảng viên | {self.n_rooms} phòng | {self.n_groups} lớp SV | "
                f"{self.n_slots} khung giờ ({self.n_days} ngày x {self.n_shifts} ca)")

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: dict, name: str = "Bộ dữ liệu") -> "Problem":
        ts = data.get("timeslots") or {}
        days = ts.get("days") or ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7"]
        shifts = ts.get("shifts") or [
            {"name": "Ca 1", "periods": "Tiết 1-3", "time": "07:00 - 09:30"},
            {"name": "Ca 2", "periods": "Tiết 4-6", "time": "09:40 - 12:10"},
            {"name": "Ca 3", "periods": "Tiết 7-9", "time": "13:00 - 15:30"},
            {"name": "Ca 4", "periods": "Tiết 10-12", "time": "15:40 - 18:10"},
        ]
        n_shifts = len(shifts)

        rooms = [Room(r["id"], r.get("name", r["id"]), int(r["capacity"]), r.get("type", "LT"))
                 for r in data["rooms"]]

        teachers = []
        for t in data["teachers"]:
            busy = set()
            for b in t.get("busy", []):
                # "day" theo cách gọi Việt Nam: Thứ 2 = 2 ... Thứ 7 = 7 ; "shifts": Ca 1..n
                d = int(b["day"]) - 2
                for s in b.get("shifts", range(1, n_shifts + 1)):
                    if 0 <= d < len(days) and 1 <= int(s) <= n_shifts:
                        busy.add(d * n_shifts + int(s) - 1)
            name = t.get("name", t["id"])
            teachers.append(Teacher(t["id"], name, t.get("short", name), frozenset(busy)))

        groups = [StudentGroup(g["id"], g.get("name", g["id"]), int(g["size"])) for g in data["student_groups"]]
        return cls(rooms, teachers, groups, data["courses"], days, shifts, name=name)

    @classmethod
    def from_dir(cls, data_dir: str | Path | None = None) -> "Problem":
        data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

        def load(fname):
            with open(data_dir / fname, encoding="utf-8") as f:
                return json.load(f)

        data = {
            "rooms": load("rooms.json"),
            "teachers": load("teachers.json"),
            "courses": load("courses.json"),
            "student_groups": load("student_groups.json"),
        }
        if (data_dir / "timeslots.json").exists():
            data["timeslots"] = load("timeslots.json")
        return cls.from_dict(data, name=f"Dữ liệu mẫu ({data_dir.name})")

