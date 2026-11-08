# 🧬 Tối Ưu Hóa Thời Khóa Biểu Trường Học Áp Dụng Giải Thuật Tiến Hóa (EA/GA)

**Môn học:** Trí tuệ nhân tạo — Chương 5: Tìm kiếm trong môi trường phức tạp  
**Nhóm thực hiện:** Nhóm 11 (VKU) — Phan Nhật Tân (22IT260), Nguyễn Văn Tấn (24IT236), Lê Đức Bảo Minh (24IT161), Lê Văn Phước  
**Phương pháp cốt lõi:** Giải thuật Di truyền (Genetic Algorithm - GA) thuộc họ Giải thuật Tiến hóa (Evolutionary Algorithm - EA), kế thừa khung tổng quát `genetic_algorithm(create, fitness, crossover, mutate)` từ bài thuyết trình.

---

## 1. Cấu Trúc Thư Mục Mã Nguồn

```text
timetabling-ga-project/
├── data/
│   ├── rooms.json              # Danh sách phòng học & sức chứa (LT / TH)
│   ├── teachers.json           # Danh sách giảng viên & khung giờ bận
│   ├── courses.json            # Danh sách lớp học phần, sĩ số, số ca/tuần
│   ├── student_groups.json     # Danh sách lớp sinh viên biên chế
│   └── timeslots.json          # Khung giờ trong tuần (Thứ 2 -> Thứ 7 x 4 ca/ngày)
├── src/
│   ├── problem.py              # Mô hình hóa & chỉ số hóa dữ liệu bài toán
│   ├── engine.py               # Khung genetic_algorithm(), evolve(), tournament(), hill_climbing()
│   ├── chromosome.py           # Hàm create() và giải mã cá thể ra JSON
│   ├── fitness.py              # Hàm fitness() tách riêng lỗi Cứng & Mềm + báo cáo kiểm tra
│   ├── operators.py            # Các hàm crossover() và mutate() (kèm Guided Mutation)
│   ├── solver.py               # Lớp tiện ích tích hợp 5 chế độ thuật toán (GA, Uniform, Guided, EP, HC)
│   ├── trace.py                # Trích xuất minh họa 1 vòng lặp F0 -> F1 (theo mẫu Slide 14-23)
│   └── visualizer.py           # Vẽ biểu đồ hội tụ (Slide 12) & xuất bảng TKB (.html / .xlsx)
├── experiments/
│   └── benchmark_30_runs.py    # Script chạy 30 lần độc lập để lập bảng so sánh (Slide 27)
├── outputs/                    # Thư mục lưu kết quả tự động (JSON, Log, HTML, Excel, PNG, CSV)
├── main.py                     # Chạy thuật toán trên Terminal & xuất toàn bộ 4 nhóm Output
├── app.py                      # Giao diện Web Dashboard tương tác bằng Streamlit
└── README.md                   # Hướng dẫn cài đặt và chạy demo
```

---

## 2. Hướng Dẫn Cài Đặt & Chạy Chương Trình

### 2.1. Cài đặt thư viện
Yêu cầu **Python 3.10+**:

```bash
pip install streamlit matplotlib pandas openpyxl
```

### 2.2. Chạy trên Terminal và xuất toàn bộ 4 nhóm Output (`main.py`)
```bash
# Chạy GA chuẩn (Single-point Crossover + Mutation + Elitism = 2)
python main.py --mode ga --pop 200 --gens 300 --mut 0.05 --seed 11

# Chạy GA + Đột biến định hướng (Guided GA - hội tụ cực nhanh)
python main.py --mode guided --pop 200 --gens 300 --seed 11 --out outputs/guided
```

Sau khi chạy xong, thư mục `outputs/` sẽ có đầy đủ:
- `best_solution.json` *(Output 1: Cấu trúc JSON lời giải tốt nhất)*
- `evolution_log.txt` *(Output 2: Log tiến hóa qua các thế hệ & Báo cáo kiểm tra 0 xung đột cứng)*
- `timetable.html` & `timetable.xlsx` *(Output 3: Bảng Thời khóa biểu trực quan theo Lớp SV / Giảng viên / Phòng học)*
- `convergence_chart.png` *(Output 4.1: Biểu đồ hội tụ kép theo mẫu Slide 12)*
- `f0_to_f1_trace.txt` *(Output 4.2: Minh họa chi tiết 1 vòng lặp tiến hóa `F0 -> F1` theo mẫu Slide 14–23)*

### 2.3. Chạy thực nghiệm 30 lần độc lập (`experiments/benchmark_30_runs.py`)
So sánh 5 biến thể (`ga`, `ga_uniform`, `guided`, `ep`, `hc`) trên nhiều `random seed` độc lập (mẫu Slide 27):

```bash
python experiments/benchmark_30_runs.py --runs 30 --pop 200 --gens 300
```

Kết quả được lưu tại `outputs/benchmark/`:
- `benchmark_comparison.csv` & `benchmark_summary.md` *(Bảng thống kê tỉ lệ thành công, điểm Fitness TB, số thế hệ hội tụ TB)*
- `benchmark_chart.png` *(Biểu đồ đường hội tụ trung bình & cột tỉ lệ thành công)*
- `benchmark_runs.csv` *(Dữ liệu chi tiết của từng lần chạy)*

### 2.4. Khởi chạy Giao diện Web Dashboard (`app.py`)
```bash
streamlit run app.py
```

Giao diện gồm:
1. **Sidebar (Tinh chỉnh tham số GA):** Chọn bộ dữ liệu hoặc tải file `.json`, chỉnh `pop_size`, `generations`, `mutation_rate`, `elitism`, chế độ thuật toán, chạy toàn bộ hoặc chạy từng bước `F0 -> F1`, tải file `.xlsx` / `.json`.
2. **Dải thẻ chỉ số (Top Metric Cards):** Thế hệ hiện tại, Fitness tốt nhất, Số lỗi cứng (nhãn xanh `HỢP LỆ` khi về `0`), Số lỗi mềm.
3. **3 Tab chức năng chính:**
   - **📅 Bảng Thời khóa biểu:** Lọc theo *Lớp sinh viên*, *Giảng viên*, hoặc *Phòng học*; tô viền đỏ + icon `⚠️` tại các ô trùng lịch khi xem lại thế hệ `F0`.
   - **📈 Biểu đồ tiến hóa & So sánh:** Vẽ trực tiếp đường `Fitness tốt nhất` vs `Fitness trung bình` và `% Cá thể hợp lệ`.
   - **🧬 Minh họa `F0 -> F1`:** Hiển thị 4 cá thể `F0`, đấu giải chọn Cha/Mẹ, vị trí cắt `Cut point`, cá thể Con sau lai ghép & đột biến.

