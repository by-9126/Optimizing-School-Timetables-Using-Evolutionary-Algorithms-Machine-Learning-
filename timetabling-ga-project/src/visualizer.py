"""Trực quan hóa: biểu đồ hội tụ (Slide 12), bảng Thời khóa biểu 3 góc nhìn, xuất HTML / Excel."""
from __future__ import annotations

import html
import io
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache_timetabling"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .fitness import (BASE_SCORE, HARD_LABELS, SOFT_LABELS, SOFT_WEIGHTS, Evaluation,  # noqa: E402
                      evaluate, find_conflicts)
from .problem import Problem  # noqa: E402

VIEWS = {"group": "Lớp sinh viên", "teacher": "Giảng viên", "room": "Phòng học"}

PALETTE = ["#DBEAFE", "#DCFCE7", "#FEF3C7", "#FCE7F3", "#E0E7FF", "#CCFBF1", "#FFE4E6", "#EDE9FE",
           "#FFEDD5", "#ECFCCB", "#CFFAFE", "#F5F5F4", "#FAE8FF", "#E0F2FE", "#FEF9C3", "#D1FAE5"]
PALETTE_BORDER = ["#3B82F6", "#22C55E", "#F59E0B", "#EC4899", "#6366F1", "#14B8A6", "#F43F5E", "#8B5CF6",
                  "#F97316", "#84CC16", "#06B6D4", "#78716C", "#D946EF", "#0EA5E9", "#EAB308", "#10B981"]


def course_color(ci: int) -> tuple[str, str]:
    return PALETTE[ci % len(PALETTE)], PALETTE_BORDER[ci % len(PALETTE_BORDER)]


# ===================================================================== LƯỚI TKB
def view_options(problem: Problem, view: str) -> list[tuple[int, str]]:
    P = problem
    if view == "group":
        return [(i, g.name) for i, g in enumerate(P.groups)]
    if view == "teacher":
        return [(i, t.name) for i, t in enumerate(P.teachers)]
    return [(i, r.name) for i, r in enumerate(P.rooms)]


def _belongs(problem: Problem, view: str, key: int, e: int, gene) -> bool:
    ev = problem.events[e]
    if view == "group":
        return key in ev.group_idxs
    if view == "teacher":
        return ev.teacher_idx == key
    return gene[0] == key


def build_grid(problem: Problem, ind, view: str, key: int) -> list[list[list[int]]]:
    """grid[ca][ngày] = danh sách buổi học (chỉ số event) của đối tượng `key` theo góc nhìn `view`."""
    P = problem
    grid = [[[] for _ in range(P.n_days)] for _ in range(P.n_shifts)]
    for e, gene in enumerate(ind):
        if _belongs(P, view, key, e, gene):
            d, s = divmod(gene[1], P.n_shifts)
            grid[s][d].append(e)
    return grid


def _cell_lines(problem: Problem, ind, view: str, e: int) -> tuple[str, list[str]]:
    P = problem
    ev = P.events[e]
    c = P.courses[ev.course_idx]
    r = P.rooms[ind[e][0]]
    t = P.teachers[ev.teacher_idx]
    title = c.name
    if view == "group":
        info = [f"P: {r.name} · GV: {t.short}"]
    elif view == "teacher":
        info = [f"P: {r.name} · Lớp: {P.group_names(ev.group_idxs)}"]
    else:
        info = [f"GV: {t.short} · Lớp: {P.group_names(ev.group_idxs)}"]
    info.append(f"Sĩ số: {ev.students} · Mã: {P.event_code(e)}")
    return title, info


TABLE_CSS = """
<style>
.tkb{border-collapse:collapse;width:100%;font-family:'Segoe UI',Roboto,Arial,sans-serif;font-size:12.5px;table-layout:fixed}
.tkb th{background:#1E3A8A;color:#fff;padding:8px 4px;border:1px solid #CBD5E1;font-weight:600}
.tkb td{border:1px solid #CBD5E1;vertical-align:top;padding:4px;height:78px}
.tkb td.shift{background:#F1F5F9;font-weight:600;color:#1E293B;width:110px;text-align:center;vertical-align:middle}
.tkb td.shift small{display:block;font-weight:400;color:#64748B}
.tkb .ev{border-left:4px solid #3B82F6;border-radius:6px;padding:4px 6px;margin-bottom:3px;color:#0F172A}
.tkb .ev b{display:block;margin-bottom:2px}
.tkb .ev span{display:block;font-size:11.5px;color:#334155}
.tkb .ev.bad{outline:2.5px solid #DC2626;background:#FEE2E2 !important}
.tkb td.badcell{background:#FEF2F2}
.tkb .empty{color:#94A3B8;font-style:italic;font-size:11.5px}
.tkb .busy{color:#B45309;font-size:11px}
</style>
"""


def timetable_html(problem: Problem, ind, view: str, key: int, conflicts: set | None = None,
                   include_css: bool = True) -> str:
    """Bảng TKB dạng lưới (Thứ 2 -> Thứ 7) x (Ca 1 -> Ca 4). Ô bị trùng lịch: viền đỏ + ⚠️."""
    P = problem
    if conflicts is None:
        conflicts = find_conflicts(P, ind)
    grid = build_grid(P, ind, view, key)
    busy = P.teachers[key].busy_slots if view == "teacher" else frozenset()
    out = [TABLE_CSS] if include_css else []
    out.append('<table class="tkb"><thead><tr><th style="width:110px">Ca học / Thứ</th>')
    out += [f"<th>{html.escape(d)}</th>" for d in P.days]
    out.append("</tr></thead><tbody>")
    for s, sh in enumerate(P.shifts):
        out.append(f'<tr><td class="shift">{html.escape(sh["name"])}<small>{html.escape(sh["periods"])}</small>'
                   f'<small>{html.escape(sh.get("time", ""))}</small></td>')
        for d in range(P.n_days):
            evs = grid[s][d]
            clash = len(evs) > 1
            bad_cell = clash or any(e in conflicts for e in evs)
            out.append(f'<td class="{"badcell" if bad_cell else ""}">')
            if not evs:
                if P.slot_index(d, s) in busy:
                    out.append('<div class="busy">⛔ GV đăng ký bận</div>')
                else:
                    out.append('<div class="empty">Trống</div>')
            for e in evs:
                ci = P.events[e].course_idx
                bg, bd = course_color(ci)
                title, info = _cell_lines(P, ind, view, e)
                is_bad = clash or e in conflicts
                warn = "⚠️ " if is_bad else ""
                out.append(f'<div class="ev{" bad" if is_bad else ""}" style="background:{bg};border-left-color:{bd}">'
                           f'<b>{warn}{html.escape(title)}</b>'
                           + "".join(f"<span>{html.escape(x)}</span>" for x in info) + "</div>")
            out.append("</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def validation_html(evaluation: Evaluation) -> str:
    rows = []
    for k, label in HARD_LABELS.items():
        v = evaluation.hard_detail[k]
        ok = v == 0
        rows.append(f"<tr><td>{label}</td><td>Cứng</td><td>{v}</td>"
                    f"<td style='color:{'#16A34A' if ok else '#DC2626'};font-weight:600'>"
                    f"{'ĐẠT ✓' if ok else 'KHÔNG ĐẠT ✗'}</td></tr>")
    for k, label in SOFT_LABELS.items():
        v = evaluation.soft_detail[k]
        rows.append(f"<tr><td>{label}</td><td>Mềm</td><td>{v}</td><td>-{v * SOFT_WEIGHTS[k]} điểm</td></tr>")
    return ("<table class='val'><thead><tr><th>Ràng buộc</th><th>Loại</th><th>Số lỗi</th><th>Kết quả</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def export_html(problem: Problem, ind, path: str | Path, evaluation: Evaluation | None = None,
                title: str = "Thời khóa biểu tối ưu bằng Giải thuật Di truyền") -> Path:
    """Xuất 1 file HTML độc lập chứa TKB theo cả 3 góc nhìn + báo cáo kiểm tra ràng buộc."""
    P = problem
    ev = evaluation or evaluate(P, ind)
    conflicts = find_conflicts(P, ind)
    parts = [f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:'Segoe UI',Roboto,Arial,sans-serif;margin:24px;background:#F8FAFC;color:#0F172A}}
h1{{color:#1E3A8A;margin-bottom:4px}} h2{{color:#1E3A8A;border-bottom:2px solid #1E3A8A;padding-bottom:4px;margin-top:36px}}
h3{{margin:22px 0 8px}} .meta{{color:#475569}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:12px 18px;min-width:170px}}
.card b{{display:block;font-size:22px;color:#1E3A8A}}
.val{{border-collapse:collapse;background:#fff}} .val td,.val th{{border:1px solid #CBD5E1;padding:6px 12px}}
.val th{{background:#E2E8F0}}
nav a{{margin-right:14px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">Nhóm 11 - VKU · {html.escape(P.summary())}</div>
<div class="cards">
<div class="card">Fitness<b>{ev.fitness:.1f} / {BASE_SCORE:.0f}</b></div>
<div class="card">Vi phạm cứng<b style="color:{'#16A34A' if ev.hard == 0 else '#DC2626'}">{ev.hard} lỗi</b></div>
<div class="card">Vi phạm mềm<b>{ev.soft} lỗi (-{ev.soft_penalty})</b></div>
</div>
<nav><a href="#group">Theo lớp sinh viên</a><a href="#teacher">Theo giảng viên</a><a href="#room">Theo phòng học</a>
<a href="#check">Kiểm tra ràng buộc</a></nav>
{TABLE_CSS}"""]
    for view, label in VIEWS.items():
        parts.append(f'<h2 id="{view}">TKB theo {label.lower()}</h2>')
        for key, name in view_options(P, view):
            parts.append(f"<h3>{html.escape(label)}: {html.escape(name)}</h3>")
            parts.append(timetable_html(P, ind, view, key, conflicts, include_css=False))
    parts.append('<h2 id="check">Báo cáo kiểm tra ràng buộc</h2>')
    parts.append(validation_html(ev))
    parts.append("</body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")
    return path


# ===================================================================== EXCEL
def export_excel(problem: Problem, ind, target=None, evaluation: Evaluation | None = None):
    """Xuất TKB ra .xlsx (mỗi lớp SV / GV / phòng 1 sheet + sheet danh sách & kiểm tra).

    target: đường dẫn file, hoặc None -> trả về bytes (dùng cho nút tải về trên giao diện).
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    P = problem
    ev = evaluation or evaluate(P, ind)
    conflicts = find_conflicts(P, ind)
    thin = Side(style="thin", color="CBD5E1")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor="1E3A8A")
    head_font = Font(bold=True, color="FFFFFF")
    wrap = Alignment(wrap_text=True, vertical="top")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    red_fill = PatternFill("solid", fgColor="FEE2E2")

    wb = Workbook()
    ws = wb.active
    ws.title = "Tổng quan"
    ws.append(["THỜI KHÓA BIỂU TỐI ƯU BẰNG GIẢI THUẬT DI TRUYỀN - NHÓM 11 VKU"])
    ws["A1"].font = Font(bold=True, size=14, color="1E3A8A")
    ws.append([P.summary()])
    ws.append([])
    ws.append(["Fitness", round(ev.fitness, 1), f"/ {BASE_SCORE:.0f}"])
    ws.append(["Vi phạm cứng", ev.hard])
    ws.append(["Vi phạm mềm", ev.soft, f"-{ev.soft_penalty} điểm"])
    ws.append([])
    ws.append(["Ràng buộc", "Loại", "Số lỗi", "Kết quả"])
    for c in ws[ws.max_row]:
        c.fill, c.font = head_fill, head_font
    for k, label in HARD_LABELS.items():
        v = ev.hard_detail[k]
        ws.append([label, "Cứng", v, "ĐẠT ✓" if v == 0 else "KHÔNG ĐẠT ✗"])
    for k, label in SOFT_LABELS.items():
        v = ev.soft_detail[k]
        ws.append([label, "Mềm", v, f"-{v * SOFT_WEIGHTS[k]} điểm"])
    ws.column_dimensions["A"].width = 42
    for col in "BCD":
        ws.column_dimensions[col].width = 16

    # Danh sách phẳng
    wl = wb.create_sheet("Danh sách buổi học")
    wl.append(["Mã HP", "Buổi", "Tên học phần", "Lớp SV", "Sĩ số", "Giảng viên", "Phòng", "Thứ", "Ca", "Xung đột"])
    for c in wl[1]:
        c.fill, c.font = head_fill, head_font
    rows = sorted(range(P.n_events), key=lambda e: (ind[e][1], P.events[e].group_idxs))
    for e in rows:
        evn = P.events[e]
        c = P.courses[evn.course_idx]
        r, t = ind[e]
        d, s = divmod(t, P.n_shifts)
        wl.append([c.id, evn.session_no, c.name, P.group_names(evn.group_idxs), evn.students,
                   P.teachers[evn.teacher_idx].name, P.rooms[r].name, P.days[d],
                   f"{P.shifts[s]['name']} ({P.shifts[s]['periods']})", "⚠️ Có" if e in conflicts else ""])
        if e in conflicts:
            for cell in wl[wl.max_row]:
                cell.fill = red_fill
    for i, w in enumerate([12, 6, 38, 14, 7, 22, 12, 8, 20, 10], 1):
        wl.column_dimensions[get_column_letter(i)].width = w

    # Lưới theo từng đối tượng
    prefix = {"group": "Lớp", "teacher": "GV", "room": "Phòng"}
    used_names = set(wb.sheetnames)
    for view in VIEWS:
        for key, name in view_options(P, view):
            sname = f"{prefix[view]} {name}"
            for ch in '[]:*?/\\':
                sname = sname.replace(ch, "-")
            sname = sname[:31]
            base, k = sname, 2
            while sname in used_names:
                sname = f"{base[:28]}_{k}"
                k += 1
            used_names.add(sname)
            wg = wb.create_sheet(sname)
            wg.append([f"TKB {VIEWS[view].lower()}: {name}"])
            wg["A1"].font = Font(bold=True, size=13, color="1E3A8A")
            wg.append(["Ca học / Thứ"] + P.days)
            for c in wg[2]:
                c.fill, c.font, c.alignment, c.border = head_fill, head_font, center, border
            grid = build_grid(P, ind, view, key)
            for s, sh in enumerate(P.shifts):
                row = [f"{sh['name']}\n{sh['periods']}\n{sh.get('time', '')}"]
                for d in range(P.n_days):
                    texts = []
                    for e in grid[s][d]:
                        title, info = _cell_lines(P, ind, view, e)
                        warn = "⚠️ " if (len(grid[s][d]) > 1 or e in conflicts) else ""
                        texts.append(warn + title + "\n" + "\n".join(info))
                    row.append("\n\n".join(texts))
                wg.append(row)
                rr = wg.max_row
                wg.row_dimensions[rr].height = 75 if max(len(grid[s][d]) for d in range(P.n_days)) <= 1 else 140
                wg.cell(rr, 1).alignment = center
                wg.cell(rr, 1).font = Font(bold=True)
                for d in range(P.n_days):
                    cell = wg.cell(rr, d + 2)
                    cell.alignment = wrap
                    evs = grid[s][d]
                    if evs:
                        bad = len(evs) > 1 or any(e in conflicts for e in evs)
                        color = "FEE2E2" if bad else course_color(P.events[evs[0]].course_idx)[0].lstrip("#")
                        cell.fill = PatternFill("solid", fgColor=color)
                for c in wg[rr]:
                    c.border = border
            wg.column_dimensions["A"].width = 14
            for d in range(P.n_days):
                wg.column_dimensions[get_column_letter(d + 2)].width = 30

    if target is None:
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    wb.save(target)
    return target


# ===================================================================== BIỂU ĐỒ
def plot_convergence(history: list[dict], path: str | Path | None = None, title: str | None = None):
    """Biểu đồ hội tụ kép (theo mẫu Slide 12): trái = Fitness tốt nhất & trung bình, phải = % cá thể hợp lệ."""
    gens = [h["generation"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.plot(gens, [h["best_fitness"] for h in history], color="#2563EB", lw=2.2, label="Fitness tốt nhất")
    ax1.plot(gens, [h["avg_fitness"] for h in history], color="#F97316", lw=2, label="Fitness trung bình")
    first_valid = next((h["generation"] for h in history if h["best_hard"] == 0), None)
    if first_valid is not None:
        ax1.axvline(first_valid, color="#16A34A", ls="--", lw=1.2)
        ax1.annotate(f"TKB hợp lệ đầu tiên\n(thế hệ {first_valid})", xy=(first_valid, ax1.get_ylim()[0]),
                     xytext=(6, 18), textcoords="offset points", color="#16A34A", fontsize=9)
    ax1.set_title("Fitness theo thế hệ")
    ax1.set_xlabel("Thế hệ")
    ax1.set_ylabel(f"Fitness (tối đa {BASE_SCORE:.0f})")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    valid = [h["valid_pct"] for h in history]
    ax2.plot(gens, valid, color="#16A34A", lw=2.2)
    ax2.fill_between(gens, valid, color="#16A34A", alpha=0.12)
    ax2.set_ylim(-3, 103)
    ax2.set_title("% cá thể hợp lệ (0 vi phạm ràng buộc cứng)")
    ax2.set_xlabel("Thế hệ")
    ax2.set_ylabel("% quần thể")
    ax2.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_benchmark(curves: dict[str, list[float]], path: str | Path | None = None,
                   success: dict[str, float] | None = None):
    """So sánh các biến thể: đường Fitness tốt nhất trung bình qua 30 lần chạy + cột tỉ lệ thành công."""
    fig, axes = plt.subplots(1, 2 if success else 1, figsize=(13 if success else 7.5, 4.6))
    ax1 = axes[0] if success else axes
    colors = ["#2563EB", "#9333EA", "#16A34A", "#F97316", "#DC2626", "#0891B2"]
    for i, (name, ys) in enumerate(curves.items()):
        ax1.plot(range(len(ys)), ys, lw=2, color=colors[i % len(colors)], label=name)
    ax1.set_title("Fitness tốt nhất (trung bình các lần chạy)")
    ax1.set_xlabel("Thế hệ (Leo đồi: mỗi pop_size lần đánh giá = 1 thế hệ)")
    ax1.set_ylabel("Fitness")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")
    if success:
        ax2 = axes[1]
        names = list(success)
        vals = [success[n] for n in names]
        bars = ax2.barh(range(len(names)), vals, color=[colors[i % len(colors)] for i in range(len(names))])
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels([n if len(n) < 34 else n[:32] + "…" for n in names], fontsize=8.5)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 105)
        ax2.set_xlabel("% lần chạy tìm được TKB 0 lỗi cứng")
        ax2.set_title("Tỉ lệ thành công")
        for b, v in zip(bars, vals):
            ax2.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
        ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def pad_curve(ys: list[float], length: int) -> list[float]:
    """Kéo dài đường cong (khi thuật toán dừng sớm) bằng giá trị cuối để lấy trung bình."""
    if not ys:
        return [math.nan] * length
    return list(ys[:length]) + [ys[-1]] * max(0, length - len(ys))

