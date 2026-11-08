"""Thực nghiệm 30 lần chạy độc lập (30 random seed) để so sánh các biến thể (theo mẫu Slide 27).

    python experiments/benchmark_30_runs.py                      # 30 lần x 5 biến thể, dữ liệu data/
    python experiments/benchmark_30_runs.py --runs 10 --gens 200
    python experiments/benchmark_30_runs.py --data data/hard --out outputs/benchmark_hard

Xuất ra (thư mục --out, mặc định outputs/benchmark):
    benchmark_comparison.csv   Bảng tổng hợp (tỉ lệ thành công, fitness TB, số thế hệ hội tụ TB, ...)
    benchmark_runs.csv         Kết quả thô từng lần chạy
    benchmark_chart.png        Đường hội tụ trung bình + biểu đồ tỉ lệ thành công
    benchmark_summary.md       Bảng Markdown để dán vào báo cáo
"""
from __future__ import annotations

import argparse
import csv
import statistics as stats
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.problem import Problem  # noqa: E402
from src.solver import MODES, GAConfig, solve  # noqa: E402

NOTES = {
    "ga": "Lai ghép 1 điểm ghép các cụm lịch tốt của Cha và Mẹ; tinh hoa giữ không tụt fitness.",
    "ga_uniform": "Trộn gen 50/50 khám phá rộng nhưng dễ phá vỡ các 'khối gen' tốt nằm liền nhau.",
    "guided": "Đột biến nhắm thẳng vào các buổi đang trùng lịch + chọn phương án ít xung đột nhất.",
    "ep": "Chỉ đột biến, không ghép được các nhánh lịch tốt từ hai cá thể khác nhau.",
    "hc": "1 trạng thái, chỉ nhận bước tốt hơn: nhanh nhưng dễ kẹt cực trị địa phương.",
}


def run_one(data_dir: str, mode: str, seed: int, pop: int, gens: int, mut: float, elitism: int,
            patience: int | None) -> dict:
    problem = Problem.from_dir(data_dir)
    cfg = GAConfig(mode=mode, pop_size=pop, generations=gens, mutation_rate=mut, elitism=elitism,
                   seed=seed, patience=patience)
    res, ev = solve(problem, cfg)
    best = ev.evaluate(res.best)
    return {
        "variant": mode, "seed": seed, "success": int(best.hard == 0),
        "best_fitness": round(best.fitness, 2), "hard": best.hard, "soft": best.soft,
        "penalty": best.penalty, "generation_found": res.generation_found,
        "first_valid_generation": res.first_valid_generation if res.first_valid_generation is not None else "",
        "generations_run": res.generations_run, "elapsed": round(res.elapsed, 2),
        "curve": [h["best_fitness"] for h in res.history],
    }


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "benchmark"))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--mut", type=float, default=0.05)
    ap.add_argument("--elitism", type=int, default=2)
    ap.add_argument("--patience", type=int, default=150, help="Dừng nếu best không cải thiện sau N thế hệ")
    ap.add_argument("--variants", default="ga,ga_uniform,guided,ep,hc")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    from src.visualizer import pad_curve, plot_benchmark

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in MODES:
            raise SystemExit(f"Biến thể không hợp lệ: {v}. Chọn trong {list(MODES)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problem = Problem.from_dir(args.data)
    print(f"Bài toán: {problem.summary()}")
    print(f"Thực nghiệm: {args.runs} lần chạy x {len(variants)} biến thể | pop={args.pop}, gens={args.gens}, "
          f"mutation={args.mut}, elitism={args.elitism}, patience={args.patience}")

    t0 = time.perf_counter()
    rows = []
    jobs = [(args.data, v, seed, args.pop, args.gens, args.mut, args.elitism, args.patience)
            for v in variants for seed in range(args.runs)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i:3d}/{len(jobs)}] {r['variant']:<10} seed={r['seed']:<3} fitness={r['best_fitness']:7.2f} "
                  f"hard={r['hard']} soft={r['soft']} gen_found={r['generation_found']} ({r['elapsed']}s)", flush=True)
    print(f"Tổng thời gian: {time.perf_counter() - t0:.1f}s")

    rows.sort(key=lambda r: (variants.index(r["variant"]), r["seed"]))
    with open(out / "benchmark_runs.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = [k for k in rows[0] if k != "curve"]
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    summary, curves, success_rates = [], {}, {}
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        n = len(rs)
        succ = sum(r["success"] for r in rs)
        fits = [r["best_fitness"] for r in rs]
        perfect = sum(1 for r in rs if r["penalty"] == 0)
        fv = [r["first_valid_generation"] for r in rs if r["first_valid_generation"] != ""]
        gf = [r["generation_found"] for r in rs]
        summary.append({
            "Biến thể / Thuật toán": MODES[v],
            "Tỉ lệ tìm ra TKB 0 lỗi cứng": f"{succ} / {n} ({100 * succ / n:.1f}%)",
            "Tỉ lệ đạt 1000 điểm (0 lỗi mềm)": f"{perfect} / {n} ({100 * perfect / n:.1f}%)",
            "Fitness TB đạt được": f"{stats.mean(fits):.1f} ± {stats.pstdev(fits):.1f}",
            "Fitness tốt nhất / kém nhất": f"{max(fits):.1f} / {min(fits):.1f}",
            "Thế hệ đầu tiên có TKB hợp lệ (TB)": f"{stats.mean(fv):.1f}" if fv else "-",
            "Số thế hệ hội tụ TB": f"{stats.mean(gf):.1f}",
            "Thời gian TB (s)": f"{stats.mean(r['elapsed'] for r in rs):.2f}",
            "Nhận xét đặc trưng": NOTES.get(v, ""),
        })
        success_rates[MODES[v]] = 100 * succ / n
        L = args.gens + 1
        padded = [pad_curve(r["curve"], L) for r in rs]
        curves[MODES[v]] = [stats.mean(col) for col in zip(*padded)]

    with open(out / "benchmark_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        wr.writerows(summary)

    md = [f"# Kết quả thực nghiệm {args.runs} lần chạy độc lập", "",
          f"- Bài toán: {problem.summary()}",
          f"- Tham số: pop_size={args.pop}, generations={args.gens}, mutation_rate={args.mut}, "
          f"elitism={args.elitism}, tournament k=3, dừng sớm khi đạt 1000 điểm hoặc {args.patience} thế hệ không cải thiện",
          "- Leo đồi: ngân sách đánh giá bằng GA (pop_size x generations); mỗi pop_size lần đánh giá = 1 'thế hệ'.", "",
          "| " + " | ".join(summary[0].keys()) + " |",
          "|" + "---|" * len(summary[0])]
    for s in summary:
        md.append("| " + " | ".join(str(x) for x in s.values()) + " |")
    md += ["", "![Biểu đồ so sánh](benchmark_chart.png)"]
    (out / "benchmark_summary.md").write_text("\n".join(md), encoding="utf-8")
    plot_benchmark(curves, out / "benchmark_chart.png", success_rates)

    print("\n" + "\n".join(md[:-2]))
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    main()

