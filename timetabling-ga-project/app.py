"""Giao diện Dashboard: Tối ưu Thời khóa biểu bằng Giải thuật Di truyền (Nhóm 11 - VKU).

Chạy:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chromosome import decode, solution_to_dict  # noqa: E402
from src.fitness import (BASE_SCORE, HARD_LABELS, SOFT_LABELS, SOFT_WEIGHTS, conflict_details,  # noqa: E402
                         find_conflicts, validation_report)
from src.problem import Problem  # noqa: E402
from src.solver import MODES, GAConfig, make_iterator, solve  # noqa: E402
from src.trace import NAMES, build_f0_f1_trace, gene_str, trace_to_text  # noqa: E402
from src.visualizer import (VIEWS, course_color, export_excel, export_html, plot_convergence,  # noqa: E402
                            timetable_html, view_options)

st.set_page_config(page_title="Tối ưu TKB bằng GA - Nhóm 11 VKU", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.6rem}
div[data-testid="stMetric"]{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:10px 14px}
.badge-ok{background:#DCFCE7;color:#166534;padding:2px 10px;border-radius:999px;font-weight:600}
.badge-bad{background:#FEE2E2;color:#991B1B;padding:2px 10px;border-radius:999px;font-weight:600}
.genes{border-collapse:collapse;font-size:12.5px;font-family:Consolas,'Segoe UI',monospace;width:100%}
.genes th{background:#1E3A8A;color:#fff;padding:6px;border:1px solid #CBD5E1;position:sticky;top:0}
.genes td{border:1px solid #E2E8F0;padding:3px 6px;white-space:nowrap}
.genes tr.cut td{background:#111827;color:#FDE047;text-align:center;font-weight:700}
.genes td.fa{background:#DBEAFE} .genes td.mo{background:#FCE7F3}
.genes td.mut{background:#FEF08A;font-weight:700}
.genes .bad{color:#DC2626;font-weight:700}
.f0card{border:1px solid #E2E8F0;border-radius:12px;padding:10px 14px;background:#fff;height:100%}
.f0card h4{margin:0 0 4px 0}
</style>
""", unsafe_allow_html=True)


# =============================================================== DỮ LIỆU
@st.cache_resource(show_spinner=False)
def load_sample(path: str) -> Problem:
    return Problem.from_dir(path)


def sample_datasets() -> dict[str, Path]:
    out = {"Bộ dữ liệu mẫu VKU (data/)": ROOT / "data"}
    for sub in sorted((ROOT / "data").iterdir()):
        if sub.is_dir() and (sub / "courses.json").exists():
            label = {"hard": "Kịch bản thiếu phòng - khó (data/hard/)"}.get(sub.name, f"Bộ dữ liệu {sub.name}")
            out[label] = sub
    return out


def problem_from_uploads(files) -> Problem | None:
    data = {}
    keymap = {"rooms": "rooms", "teachers": "teachers", "courses": "courses",
              "student_groups": "student_groups", "timeslots": "timeslots"}
    for f in files:
        stem = Path(f.name).stem
        if stem in keymap:
            data[keymap[stem]] = json.loads(f.getvalue().decode("utf-8"))
    missing = [k for k in ["rooms", "teachers", "courses", "student_groups"] if k not in data]
    if missing:
        st.sidebar.warning("Thiếu file: " + ", ".join(f"{m}.json" for m in missing))
        return None
    return Problem.from_dict(data, name="Dữ liệu tải lên")


# =============================================================== SIDEBAR
sb = st.sidebar
sb.title("🧬 Tham số GA")
datasets = sample_datasets()
src_options = list(datasets) + ["Tải lên file JSON"]
data_src = sb.selectbox("Dữ liệu đầu vào", src_options)
if data_src == "Tải lên file JSON":
    ups = sb.file_uploader("rooms.json, teachers.json, courses.json, student_groups.json (+ timeslots.json)",
                           type="json", accept_multiple_files=True)
    problem = problem_from_uploads(ups) if ups else None
    if problem is None:
        problem = load_sample(str(datasets[src_options[0]]))
        sb.caption("Đang dùng tạm bộ dữ liệu mẫu.")
else:
    problem = load_sample(str(datasets[data_src]))

mode = sb.selectbox("Chế độ thuật toán", list(MODES), format_func=lambda m: MODES[m])
pop_size = sb.slider("Kích thước quần thể (pop_size)", 50, 500, 200, step=10)
generations = sb.slider("Số thế hệ tối đa (generations)", 50, 1000, 300, step=10)
mutation_rate = sb.slider("Tỉ lệ đột biến (mutation_rate)", 0.01, 0.20, 0.05, step=0.01)
elitism = sb.slider("Số cá thể ưu tú giữ lại (elitism)", 0, 10, 2)
tournament_k = sb.slider("Kích thước đấu giải (tournament k)", 2, 7, 3)
seed = sb.number_input("Random seed", 0, 10_000, 11, step=1)
stop_perfect = sb.checkbox("Dừng sớm khi đạt 1000 điểm", value=True)

cfg = GAConfig(mode=mode, pop_size=pop_size, generations=generations, mutation_rate=mutation_rate,
               elitism=elitism, tournament_k=tournament_k, seed=int(seed), stop_on_perfect=stop_perfect)
cfg_key = (json.dumps(cfg.to_dict(), sort_keys=True), problem.name, problem.n_events)

sb.markdown("---")
c1, c2 = sb.columns(2)
btn_run = sb.button("🚀 Bắt đầu Tiến hóa", type="primary", use_container_width=True)
btn_step = c1.button("⏭️ Từng bước", use_container_width=True, help="F0 -> F1 -> F2 ... (mỗi lần bấm 1 thế hệ)")
btn_step10 = c2.button("⏩ +10 thế hệ", use_container_width=True)
btn_reset = sb.button("🔄 Đặt lại", use_container_width=True)

ss = st.session_state
if btn_reset:
    for k in ["it", "history", "state", "f0_best", "best", "cfg_key", "done", "evaluator"]:
        ss.pop(k, None)


# =============================================================== HEADER
st.title("📅 Tối ưu hóa Thời khóa biểu bằng Giải thuật Di truyền")
st.caption(f"Nhóm 11 - VKU · Trí tuệ nhân tạo - Chương 5 · {problem.summary()}")
for w in problem.warnings:
    st.warning(w)

metrics_ph = st.container()
live_ph = st.empty()


def render_metrics(container, state_gen, max_gen, best_eval, f0_fit, done=False):
    with container:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Thế hệ hiện tại", f"{state_gen} / {max_gen}", "Đã dừng" if done else None, delta_color="off")
        m2.metric("Fitness tốt nhất", f"{best_eval.fitness:.1f} / {BASE_SCORE:.0f}",
                  f"{best_eval.fitness - f0_fit:+.1f} điểm so với F0")
        m3.metric("Vi phạm ràng buộc cứng", f"{best_eval.hard} lỗi",
                  "✓ HỢP LỆ" if best_eval.hard == 0 else f"-{best_eval.hard} xung đột",
                  delta_color="normal")
        m4.metric("Vi phạm ràng buộc mềm", f"{best_eval.soft} lỗi", f"-{best_eval.soft_penalty} điểm",
                  delta_color="off")


def ensure_iterator(force_new=False):
    if force_new or "it" not in ss or ss.get("cfg_key") != cfg_key or ss.get("done"):
        it, evaluator = make_iterator(problem, cfg)
        ss.it, ss.evaluator, ss.cfg_key = it, evaluator, cfg_key
        ss.history, ss.state, ss.f0_best, ss.best, ss.done = [], None, None, None, False
        return True
    return False


def advance(n_steps: int, live: bool = False):
    """Chạy thêm n_steps thế hệ (hoặc tới khi dừng)."""
    ev = ss.evaluator
    every = max(1, cfg.generations // 60)
    for _ in range(n_steps):
        try:
            st_ = next(ss.it)
        except StopIteration:
            ss.done = True
            break
        ss.state = st_
        ss.history.append(st_.as_record())
        if ss.f0_best is None:
            ss.f0_best = st_.best
        if ss.best is None or ev(st_.best) > ev(ss.best):
            ss.best = st_.best
        if live and (st_.generation % every == 0):
            df = pd.DataFrame(ss.history).set_index("generation")
            with live_ph.container():
                st.progress(min(1.0, st_.generation / cfg.generations),
                            text=f"Đang tiến hóa... thế hệ {st_.generation}/{cfg.generations}")
                lc1, lc2 = st.columns(2)
                lc1.line_chart(df[["best_fitness", "avg_fitness"]].rename(
                    columns={"best_fitness": "Fitness tốt nhất", "avg_fitness": "Fitness trung bình"}),
                    color=["#2563EB", "#F97316"], height=230)
                lc2.area_chart(df[["valid_pct"]].rename(columns={"valid_pct": "% cá thể hợp lệ"}),
                               color=["#16A34A"], height=230)
                b = ev.evaluate(ss.best)
                st.caption(f"Best: {b.fitness:.1f} | lỗi cứng {b.hard} | lỗi mềm {b.soft}")
    # Kiểm tra iterator đã kết thúc chưa (đạt 1000 điểm / hết thế hệ)
    if ss.state is not None and not ss.done:
        if ss.state.generation >= cfg.generations or (
                cfg.stop_on_perfect and ss.state.best_fitness >= BASE_SCORE - 1e-9):
            ss.done = True


if btn_run:
    ensure_iterator(force_new=True)
    advance(cfg.generations + 1, live=True)
    live_ph.empty()
elif btn_step:
    ensure_iterator()
    advance(1)
elif btn_step10:
    ensure_iterator()
    advance(10)

has_run = ss.get("state") is not None
if has_run:
    ev = ss.evaluator
    best = ss.best
    best_eval = ev.evaluate(best)
    f0_eval = ev.evaluate(ss.f0_best)
    render_metrics(metrics_ph, ss.state.generation, cfg.generations, best_eval, f0_eval.fitness, ss.done)
    if ss.done:
        msg = (f"✅ Hoàn tất sau {ss.state.generation} thế hệ. " +
               ("Đã tìm được TKB **hợp lệ** (0 xung đột cứng)." if best_eval.hard == 0
                else f"Còn {best_eval.hard} xung đột cứng - thử tăng pop_size / số thế hệ hoặc dùng Guided GA."))
        (st.success if best_eval.hard == 0 else st.warning)(msg)
    # Nút xuất file
    sb.markdown("---")
    sb.download_button("📥 Xuất Thời khóa biểu (.xlsx)", data=export_excel(problem, best, None, best_eval),
                       file_name="thoi_khoa_bieu_GA.xlsx", use_container_width=True,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    sol = solution_to_dict(problem, best, best_eval, next(
        (h["generation"] for h in ss.history if abs(h["best_fitness"] - best_eval.fitness) < 1e-9), None),
        extra={"algorithm": MODES[cfg.mode], "config": cfg.to_dict()})
    sb.download_button("🧾 Tải lời giải (.json)", data=json.dumps(sol, ensure_ascii=False, indent=2),
                       file_name="best_solution.json", mime="application/json", use_container_width=True)
else:
    with metrics_ph:
        st.info("👈 Chỉnh tham số ở thanh bên trái rồi bấm **🚀 Bắt đầu Tiến hóa** "
                "(hoặc **⏭️ Từng bước** để xem F0 → F1 → F2 ...).")

tab1, tab2, tab3, tab4 = st.tabs(["📅 Bảng Thời khóa biểu", "📈 Biểu đồ tiến hóa & So sánh",
                                  "🧬 Minh họa F0 → F1", "📂 Dữ liệu đầu vào"])

# =============================================================== TAB 1
with tab1:
    if not has_run:
        st.write("Chưa có lời giải. Hãy chạy giải thuật trước.")
    else:
        cA, cB, cC = st.columns([1.3, 1, 1.4])
        which = cA.radio("Cá thể hiển thị", ["Tốt nhất hiện tại", "Tốt nhất ở F0 (trước tiến hóa)"],
                         horizontal=True)
        ind = best if which.startswith("Tốt nhất hiện") else ss.f0_best
        ind_eval = ev.evaluate(ind)
        view = cB.selectbox("Xem theo", list(VIEWS), format_func=lambda v: VIEWS[v])
        opts = view_options(problem, view)
        key = cC.selectbox(VIEWS[view], [o[0] for o in opts], format_func=lambda i: dict(opts)[i])
        conflicts = find_conflicts(problem, ind)
        badge = ("<span class='badge-ok'>HỢP LỆ - 0 xung đột cứng</span>" if ind_eval.hard == 0
                 else f"<span class='badge-bad'>⚠️ {ind_eval.hard} xung đột cứng</span>")
        st.markdown(f"**Fitness:** {ind_eval.fitness:.1f} / {BASE_SCORE:.0f} &nbsp; {badge} &nbsp; "
                    f"Lỗi mềm: {ind_eval.soft} (-{ind_eval.soft_penalty} điểm)", unsafe_allow_html=True)
        st.html(timetable_html(problem, ind, view, key, conflicts))

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("##### ✅ Bảng kiểm tra ràng buộc")
            rows = [{"Ràng buộc": lb, "Loại": "Cứng", "Số lỗi": ind_eval.hard_detail[k],
                     "Kết quả": "ĐẠT ✓" if ind_eval.hard_detail[k] == 0 else "KHÔNG ĐẠT ✗"}
                    for k, lb in HARD_LABELS.items()]
            rows += [{"Ràng buộc": lb, "Loại": "Mềm", "Số lỗi": ind_eval.soft_detail[k],
                      "Kết quả": f"-{ind_eval.soft_detail[k] * SOFT_WEIGHTS[k]} điểm"}
                     for k, lb in SOFT_LABELS.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with col2:
            st.markdown("##### ⚠️ Chi tiết xung đột cứng")
            det = conflict_details(problem, ind)
            if det:
                st.code("\n".join(det[:40]) + (f"\n... và {len(det) - 40} xung đột khác" if len(det) > 40 else ""),
                        language=None)
            else:
                st.success("Không có xung đột cứng nào.")
        with st.expander("📋 Danh sách toàn bộ buổi học (Output 1)"):
            df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in decode(problem, ind)])
            st.dataframe(df, hide_index=True, use_container_width=True)
        with st.expander("🖨️ Báo cáo kiểm tra dạng văn bản (Output 2)"):
            st.code(validation_report(problem, ind, ind_eval), language=None)
        html_path = ROOT / "outputs" / "_app_timetable.html"
        st.download_button("🌐 Tải TKB dạng HTML (cả 3 góc nhìn)",
                           data=export_html(problem, ind, html_path, ind_eval).read_text(encoding="utf-8"),
                           file_name="thoi_khoa_bieu.html", mime="text/html")

# =============================================================== TAB 2
with tab2:
    if has_run:
        df = pd.DataFrame(ss.history).set_index("generation")
        g1, g2 = st.columns(2)
        g1.markdown("**Fitness theo thế hệ** (xanh: tốt nhất · cam: trung bình)")
        g1.line_chart(df[["best_fitness", "avg_fitness"]].rename(
            columns={"best_fitness": "Fitness tốt nhất", "avg_fitness": "Fitness trung bình"}),
            color=["#2563EB", "#F97316"], height=300)
        g2.markdown("**% cá thể hợp lệ** (0 vi phạm ràng buộc cứng)")
        g2.area_chart(df[["valid_pct"]].rename(columns={"valid_pct": "% cá thể hợp lệ"}),
                      color=["#16A34A"], height=300)
        with st.expander("🖼️ Biểu đồ hội tụ kép kiểu Slide 12 (matplotlib - dùng cho báo cáo)"):
            st.pyplot(plot_convergence(ss.history, title=f"{MODES[cfg.mode]} - pop={cfg.pop_size}"))
        with st.expander("🔢 Số liệu từng thế hệ"):
            st.dataframe(df, use_container_width=True)
    else:
        st.write("Chạy giải thuật để xem biểu đồ hội tụ.")

    st.markdown("---")
    st.subheader("⚖️ So sánh nhanh các biến thể")
    st.caption("Chạy mỗi biến thể N lần với các seed khác nhau (cùng tham số ở thanh bên). "
               "Bản đầy đủ 30 lần: `python experiments/benchmark_30_runs.py`.")
    q1, q2, q3 = st.columns([2, 1, 1])
    chosen = q1.multiselect("Biến thể", list(MODES), default=["ga", "guided", "ep", "hc"],
                            format_func=lambda m: MODES[m])
    n_runs = q2.number_input("Số lần chạy / biến thể", 1, 30, 5)
    q3.write("")
    if q3.button("▶️ Chạy so sánh", use_container_width=True):
        prog = st.progress(0.0)
        rows, curves = [], {}
        total = len(chosen) * n_runs
        k = 0
        for m in chosen:
            cs = []
            for s in range(n_runs):
                c = GAConfig(**{**cfg.to_dict(), "mode": m, "seed": s, "patience": 150})
                r, e = solve(problem, c)
                b = e.evaluate(r.best)
                rows.append({"Biến thể": MODES[m], "seed": s, "0 lỗi cứng": b.hard == 0, "Fitness": b.fitness,
                             "Lỗi cứng": b.hard, "Lỗi mềm": b.soft, "Thế hệ hội tụ": r.generation_found,
                             "Thời gian (s)": r.elapsed})
                cs.append([h["best_fitness"] for h in r.history])
                k += 1
                prog.progress(k / total, text=f"{MODES[m]} - lần {s + 1}/{n_runs}")
            L = max(len(x) for x in cs)
            curves[MODES[m]] = [sum((x[i] if i < len(x) else x[-1]) for x in cs) / len(cs) for i in range(L)]
        raw = pd.DataFrame(rows)
        summ = raw.groupby("Biến thể", sort=False).agg(
            **{"Tỉ lệ 0 lỗi cứng (%)": ("0 lỗi cứng", lambda x: 100 * x.mean()),
               "Fitness TB": ("Fitness", "mean"), "Fitness tốt nhất": ("Fitness", "max"),
               "Thế hệ hội tụ TB": ("Thế hệ hội tụ", "mean"), "Thời gian TB (s)": ("Thời gian (s)", "mean")})
        st.dataframe(summ.round(2), use_container_width=True)
        L = max(len(v) for v in curves.values())
        st.line_chart(pd.DataFrame({k: v + [v[-1]] * (L - len(v)) for k, v in curves.items()}), height=300)
        with st.expander("Kết quả thô"):
            st.dataframe(raw, hide_index=True, use_container_width=True)

    bench = ROOT / "outputs" / "benchmark"
    if (bench / "benchmark_comparison.csv").exists():
        with st.expander("📊 Kết quả benchmark 30 lần chạy đã lưu (outputs/benchmark)", expanded=False):
            st.dataframe(pd.read_csv(bench / "benchmark_comparison.csv"), hide_index=True,
                         use_container_width=True)
            if (bench / "benchmark_chart.png").exists():
                st.image(str(bench / "benchmark_chart.png"))

# =============================================================== TAB 3
with tab3:
    st.markdown("Minh họa **một vòng tiến hóa** trên 4 thời khóa biểu mẫu (giống Slide 14-23): "
                "Khởi tạo F0 → Chọn lọc đấu giải → Lai ghép cắt 1 điểm → Đột biến → F1.")
    tcol1, tcol2 = st.columns([1, 4])
    if "trace_seed" not in ss:
        ss.trace_seed = int(seed)
    if tcol1.button("🎲 Tạo ví dụ khác"):
        ss.trace_seed += 1000
    tr = build_f0_f1_trace(problem, seed=ss.trace_seed, mutation_rate=max(cfg.mutation_rate, 0.03))
    if tr is None:
        st.warning("Không tạo được ví dụ minh họa.")
    else:
        st.markdown("#### Bước 1 · Quần thể F0 (4 cá thể ngẫu nhiên)")
        cols = st.columns(4)
        for i, c in enumerate(cols):
            e = tr["f0_eval"][i]
            role = " 👨 Cha" if i == tr["p1"] else (" 👩 Mẹ" if i == tr["p2"] else "")
            det = tr["f0_details"][i]
            items = "".join(f"<li>{html.escape(m)}</li>" for m in det[:4])
            more = f"<li>... +{len(det) - 4} xung đột</li>" if len(det) > 4 else ""
            c.markdown(f"<div class='f0card'><h4>TKB {NAMES[i]}{role}</h4>"
                       f"<b>Fitness: {e.fitness:.1f}</b><br>Lỗi cứng: <span class='bad' style='color:#DC2626'>"
                       f"{e.hard}</span> · Lỗi mềm: {e.soft}<ul style='font-size:11.5px;padding-left:16px'>"
                       f"{items}{more}</ul></div>", unsafe_allow_html=True)

        st.markdown("#### Bước 2 · Chọn lọc đấu giải (k = 3)")
        for k_, (t, win) in enumerate(tr["tournaments"], 1):
            cand = ", ".join(f"{NAMES[i]} ({tr['f0_eval'][i].fitness:.1f})" for i in t)
            st.markdown(f"- Lượt {k_}: bốc ngẫu nhiên **[{cand}]** → thắng: **TKB {NAMES[win]}**")

        cut = tr["cut"]
        st.markdown(f"#### Bước 3 & 4 · Lai ghép cắt tại `cut = {cut}` và đột biến")
        father, mother = tr["f0"][tr["p1"]], tr["f0"][tr["p2"]]
        cf, cm = tr["f0_conflicts"][tr["p1"]], tr["f0_conflicts"][tr["p2"]]
        cc, ccm = tr["child_conflicts"], tr["child_mut_conflicts"]
        mut_pos = {i for i, _, _ in tr["mutations"]}

        def gcell(ind_, i, conf, cls=""):
            txt = html.escape(gene_str(problem, ind_[i]))
            if i in conf:
                txt = f"<span class='bad'>⚠ {txt}</span>"
            return f"<td class='{cls}'>{txt}</td>"

        rows_html = []
        for i in range(problem.n_events):
            if i == cut:
                rows_html.append(f"<tr class='cut'><td colspan='6'>✂ ĐIỂM CẮT · cut = {cut} · "
                                 f"phía trên lấy từ Cha, phía dưới lấy từ Mẹ</td></tr>")
            src_cls = "fa" if i < cut else "mo"
            rows_html.append(
                f"<tr><td>{i}</td><td>{html.escape(problem.event_code(i))}</td>"
                + gcell(father, i, cf, "fa" if i < cut else "")
                + gcell(mother, i, cm, "mo" if i >= cut else "")
                + gcell(tr["child"], i, cc, src_cls)
                + gcell(tr["child_mut"], i, ccm, "mut" if i in mut_pos else src_cls) + "</tr>")
        st.html("<div style='max-height:520px;overflow:auto'><table class='genes'><thead><tr><th>#</th>"
                "<th>Buổi học</th>"
                f"<th>Cha (TKB {NAMES[tr['p1']]})</th><th>Mẹ (TKB {NAMES[tr['p2']]})</th>"
                "<th>Con sau lai ghép</th><th>Con sau đột biến ★</th></tr></thead><tbody>"
                + "".join(rows_html) + "</tbody></table></div>"
                + "<style>" + ".genes{border-collapse:collapse;font-size:12.5px;font-family:Consolas,monospace;"
                  "width:100%}.genes th{background:#1E3A8A;color:#fff;padding:6px;position:sticky;top:0}"
                  ".genes td{border:1px solid #E2E8F0;padding:3px 6px;white-space:nowrap}"
                  ".genes tr.cut td{background:#111827;color:#FDE047;text-align:center;font-weight:700}"
                  ".genes td.fa{background:#DBEAFE}.genes td.mo{background:#FCE7F3}"
                  ".genes td.mut{background:#FEF08A;font-weight:700}.genes .bad{color:#DC2626;font-weight:700}"
                + "</style>")
        st.caption("Ký hiệu gen: Phòng@Thứ-Ca (vd V.A201@T2-C1). Nền xanh = gen từ Cha, hồng = gen từ Mẹ, "
                   "vàng = gen bị đột biến, ⚠ đỏ = gen đang dính xung đột cứng.")
        for i, old, new in tr["mutations"]:
            st.markdown(f"- ★ Đột biến gen **{i}** ({problem.event_code(i)}): "
                        f"`{gene_str(problem, old)}` → `{gene_str(problem, new)}`")

        st.markdown("#### Bước 5 · So sánh Cha - Mẹ - Con")
        ef, em = tr["f0_eval"][tr["p1"]], tr["f0_eval"][tr["p2"]]
        ec, ecm = tr["child_eval"], tr["child_mut_eval"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Cha (TKB {NAMES[tr['p1']]})", f"{ef.fitness:.1f}", f"{ef.hard} lỗi cứng", delta_color="off")
        k2.metric(f"Mẹ (TKB {NAMES[tr['p2']]})", f"{em.fitness:.1f}", f"{em.hard} lỗi cứng", delta_color="off")
        k3.metric("Con sau lai ghép", f"{ec.fitness:.1f}", f"{ec.fitness - max(ef.fitness, em.fitness):+.1f} "
                  "so với bố/mẹ tốt nhất")
        k4.metric("Con sau đột biến", f"{ecm.fitness:.1f}", f"{ecm.fitness - ec.fitness:+.1f} do đột biến")

        st.markdown("#### Bước 6 · Quần thể F1 (2 tinh hoa + 2 con)")
        f1df = pd.DataFrame([{"Cá thể": n, "Fitness": round(e.fitness, 1), "Lỗi cứng": e.hard, "Lỗi mềm": e.soft}
                             for n, _, e in tr["f1"]])
        avg0 = sum(e.fitness for e in tr["f0_eval"]) / 4
        avg1 = f1df["Fitness"].mean()
        st.dataframe(f1df, hide_index=True, use_container_width=False)
        st.markdown(f"Fitness trung bình: **F0 = {avg0:.1f} → F1 = {avg1:.1f}** · "
                    f"Fitness tốt nhất: **{max(e.fitness for e in tr['f0_eval']):.1f} → {f1df['Fitness'].max():.1f}**")
        st.download_button("📄 Tải bản trích xuất f0_to_f1_trace.txt", data=trace_to_text(problem, tr),
                           file_name="f0_to_f1_trace.txt")

# =============================================================== TAB 4
with tab4:
    st.markdown(f"**{problem.name}** · {problem.summary()}")
    d1, d2 = st.columns(2)
    d1.markdown("**Phòng học**")
    d1.dataframe(pd.DataFrame([{"Phòng": r.name, "Sức chứa": r.capacity, "Loại": r.type} for r in problem.rooms]),
                 hide_index=True, use_container_width=True)
    d2.markdown("**Lớp sinh viên**")
    d2.dataframe(pd.DataFrame([{"Lớp": g.name, "Sĩ số": g.size} for g in problem.groups]),
                 hide_index=True, use_container_width=True)
    st.markdown("**Giảng viên & khung giờ bận**")
    st.dataframe(pd.DataFrame([{
        "Mã": t.id, "Giảng viên": t.name,
        "Số buổi dạy/tuần": sum(1 for e in problem.events if e.teacher_idx == ti),
        "Khung giờ bận": ", ".join(problem.slot_label(s, short=True) for s in sorted(t.busy_slots))}
        for ti, t in enumerate(problem.teachers)]), hide_index=True, use_container_width=True)
    st.markdown("**Lớp học phần**")
    st.dataframe(pd.DataFrame([{
        "Mã HP": c.id, "Tên học phần": c.name, "Giảng viên": problem.teachers[c.teacher_idx].name,
        "Lớp SV": problem.group_names(c.group_idxs), "Sĩ số": c.students, "Loại phòng": c.room_type,
        "Số buổi/tuần": c.sessions,
        "Phòng phù hợp": ", ".join(problem.rooms[r].name for r in problem.events[
            problem.course_events[ci][0]].suitable_rooms)}
        for ci, c in enumerate(problem.courses)]), hide_index=True, use_container_width=True)

