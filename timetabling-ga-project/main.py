"""Chạy Giải thuật Di truyền xếp Thời khóa biểu từ dòng lệnh & xuất toàn bộ Output.

Ví dụ:
    python main.py                               # GA chuẩn, tham số mặc định
    python main.py --mode guided --pop 300 --gens 500 --seed 11
    python main.py --data data/hard --out outputs/hard

Output (thư mục --out, mặc định outputs/):
    best_solution.json      Output 1 - lời giải tốt nhất (JSON)
    evolution_log.txt       Output 2 - log tiến hóa + báo cáo kiểm tra ràng buộc
    timetable.html/.xlsx    Output 3 - TKB trực quan theo Lớp SV / Giảng viên / Phòng học
    convergence_chart.png   Output 4 - biểu đồ hội tụ kép (Slide 12)
    f0_to_f1_trace.txt      Output 4 - minh họa 1 vòng tiến hóa F0 -> F1 (Slide 14-23)
    history.csv             Số liệu từng thế hệ
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.chromosome import solution_to_dict
from src.fitness import BASE_SCORE, validation_report
from src.problem import Problem
from src.solver import MODES, GAConfig, solve
from src.trace import build_f0_f1_trace, trace_to_text
from src.visualizer import export_excel, export_html, plot_convergence

ROOT = Path(__file__).resolve().parent


def fmt_state(st, label: str) -> str:
    valid = f"{st.valid_pct:5.1f}%"
    return (f"[{label}] Best Fitness: {st.best_fitness:6.1f}/{BASE_SCORE:.0f} "
            f"(Lỗi cứng: {st.best_hard:2d}, Lỗi mềm: {st.best_soft:2d}) | "
            f"Avg Fitness: {st.avg_fitness:6.1f} | Hợp lệ: {valid}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Tối ưu Thời khóa biểu bằng Giải thuật Di truyền (Nhóm 11 - VKU)")
    ap.add_argument("--data", default=str(ROOT / "data"), help="Thư mục dữ liệu JSON")
    ap.add_argument("--out", default=str(ROOT / "outputs"), help="Thư mục xuất kết quả")
    ap.add_argument("--mode", default="ga", choices=list(MODES), help="Biến thể thuật toán")
    ap.add_argument("--pop", type=int, default=200, help="Kích thước quần thể (pop_size)")
    ap.add_argument("--gens", type=int, default=300, help="Số thế hệ tối đa")
    ap.add_argument("--mut", type=float, default=0.05, help="Tỉ lệ đột biến trên mỗi gen")
    ap.add_argument("--elitism", type=int, default=2, help="Số cá thể tinh hoa giữ lại")
    ap.add_argument("--k", type=int, default=3, help="Kích thước đấu giải (tournament)")
    ap.add_argument("--seed", type=int, default=11, help="Random seed")
    ap.add_argument("--log-every", type=int, default=25, help="In log sau mỗi N thế hệ")
    ap.add_argument("--no-stop", action="store_true", help="Không dừng sớm khi đạt 1000 điểm")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problem = Problem.from_dir(args.data)
    cfg = GAConfig(mode=args.mode, pop_size=args.pop, generations=args.gens, mutation_rate=args.mut,
                   elitism=args.elitism, tournament_k=args.k, seed=args.seed,
                   stop_on_perfect=not args.no_stop)

    log_lines: list[str] = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)

    log(f"Bài toán : {problem.summary()}")
    for w in problem.warnings:
        log(f"CẢNH BÁO : {w}")
    log(f"Thuật toán: {MODES[cfg.mode]}")
    log(f"Tham số  : pop_size={cfg.pop_size}, generations={cfg.generations}, mutation_rate={cfg.mutation_rate}, "
        f"elitism={cfg.elitism}, tournament_k={cfg.tournament_k}, seed={cfg.seed}")
    log("-" * 110)

    flags = {"valid": False}

    def on_gen(st):
        if st.generation == 0:
            log(fmt_state(st, f"Khởi tạo F0 | Pop Size = {cfg.pop_size}"))
            if st.best_hard == 0:
                flags["valid"] = True
            return
        newly_valid = not flags["valid"] and st.best_hard == 0
        if newly_valid:
            flags["valid"] = True
        if newly_valid or st.generation % args.log_every == 0:
            line = fmt_state(st, f"Thế hệ {st.generation:4d}")
            if newly_valid:
                line += "  <-- Đạt TKB hợp lệ đầu tiên!"
            log(line)
        elif st.best_fitness >= BASE_SCORE - 1e-9:
            log(fmt_state(st, f"Thế hệ {st.generation:4d}") + "  <-- Đạt 1000 điểm: 0 lỗi cứng & 0 lỗi mềm!")

    result, evaluator = solve(problem, cfg, on_gen)
    best_eval = evaluator.evaluate(result.best)
    last = result.history[-1]
    if last["generation"] % args.log_every != 0 and last["best_fitness"] < BASE_SCORE - 1e-9:
        log(f"[Thế hệ {last['generation']:4d}] Best Fitness: {last['best_fitness']:6.1f}/{BASE_SCORE:.0f} "
            f"(Lỗi cứng: {last['best_hard']:2d}, Lỗi mềm: {last['best_soft']:2d}) | "
            f"Avg Fitness: {last['avg_fitness']:6.1f} | Hợp lệ: {last['valid_pct']:5.1f}%  (kết thúc)")
    log(f"Hoàn thành {result.generations_run} thế hệ trong {result.elapsed:.1f}s | "
        f"{result.evaluations} lần đánh giá fitness | Lời giải tốt nhất tìm thấy ở thế hệ {result.generation_found}")
    log(validation_report(problem, result.best, best_eval))

    # ---------------- Output 1: JSON
    sol = solution_to_dict(problem, result.best, best_eval, result.generation_found, extra={
        "algorithm": MODES[cfg.mode], "config": cfg.to_dict(),
        "first_valid_generation": result.first_valid_generation,
        "generations_run": result.generations_run, "elapsed_seconds": round(result.elapsed, 2),
    })
    (out / "best_solution.json").write_text(json.dumps(sol, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- Output 2: log
    (out / "evolution_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    with open(out / "history.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(result.history[0].keys()))
        wr.writeheader()
        wr.writerows(result.history)

    # ---------------- Output 3: TKB HTML + Excel
    export_html(problem, result.best, out / "timetable.html", best_eval)
    export_excel(problem, result.best, out / "timetable.xlsx", best_eval)

    # ---------------- Output 4: biểu đồ + trace F0 -> F1
    plot_convergence(result.history, out / "convergence_chart.png",
                     title=f"{MODES[cfg.mode]} - pop={cfg.pop_size}, mutation={cfg.mutation_rate}")
    tr = build_f0_f1_trace(problem, seed=cfg.seed or 11, mutation_rate=cfg.mutation_rate,
                           tournament_k=min(3, cfg.tournament_k))
    (out / "f0_to_f1_trace.txt").write_text(trace_to_text(problem, tr), encoding="utf-8")

    print(f"\nĐã xuất kết quả vào: {out}")
    for name in ["best_solution.json", "evolution_log.txt", "history.csv", "timetable.html", "timetable.xlsx",
                 "convergence_chart.png", "f0_to_f1_trace.txt"]:
        print(f"  - {name}")
    return result


if __name__ == "__main__":
    main()

