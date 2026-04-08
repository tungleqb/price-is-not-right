# Tóm Tắt Luồng Quy Trình Huấn Luyện

## Tổng Quan Dự Án

Dự án **Price is Not Right** nghiên cứu và so sánh hai phương pháp điều khiển robot trong các nhiệm vụ thao tác phức tạp dài hạn:

1. **Phương pháp Neuro-Symbolic (Ký hiệu-Thần kinh)**: Kết hợp lập kế hoạch PDDL ký hiệu với các chính sách học sâu (Diffusion Policy).
2. **Phương pháp VLA (Vision-Language Action)**: Sử dụng mô hình ngôn ngữ-thị giác-hành động đầu cuối (pi0).

---

## Sơ Đồ Luồng Tổng Thể

```
Môi trường Robosuite
        │
        ▼
[Bước 1] Thu thập Dữ liệu Demo
  (dataset_making/)
        │
        ▼
[Bước 2] Xử lý & Chuyển đổi Dữ liệu
  (neuro_symbolic_method/data_processing/)
        │
   ┌────┴────────────────────────────────┐
   ▼                                     ▼
[Bước 3a] Huấn luyện              [Bước 3b] Huấn luyện
  Diffusion Policy                   YOLO + Regressor
  (chính sách kỹ năng)               (nhận diện vật thể)
        │                                     │
        └────────────┬────────────────────────┘
                     ▼
[Bước 4] Xây dựng Bộ Lập Kế Hoạch PDDL
  (planning/PDDL/)
                     │
                     ▼
[Bước 5] Chạy Thử Nghiệm & Đánh Giá
  (neuro_symbolic_method/experiments_neurosymbolic.py)
```

---

## Chi Tiết Từng Bước

---

### Bước 1: Thu Thập Dữ Liệu Demo

**Thư mục:** `dataset_making/`

**Mục đích:** Sử dụng bộ lập kế hoạch ký hiệu (PDDL) và bộ thực thi oracle (dẫn đường bởi bộ dò trạng thái) để robot tự động hoàn thành nhiệm vụ trong môi trường mô phỏng, ghi lại toàn bộ dữ liệu quỹ đạo.

**Các thành phần chính:**

| File | Vai trò |
|------|---------|
| `main.py` | Điểm khởi đầu – khởi tạo môi trường, detector, bộ ghi và vòng lặp episode |
| `record_demos.py` | Lớp `RecordDemos` – bao bọc env, gọi planner, thực thi hành động, lưu quỹ đạo |
| `tasks.py` | Định nghĩa các thao tác cơ bản: `PickOperation`, `PlaceOperation`, `TurnOnOperation`, `TurnOffOperation` |
| `auto_demonstration.py` | Tự động tạo demo |
| `panda_hanoi_detector.py` | Bộ dò trạng thái ký hiệu cho robot Panda trong bài toán Hanoi |
| `graph_learner.py` | Học biểu đồ quan hệ từ dữ liệu |
| `utils.py` | Hàm tiện ích |

**Luồng chi tiết:**

```
1. Khởi tạo môi trường Robosuite (Panda robot, camera agentview + wrist)
          │
          ▼
2. Khởi tạo Detector (HanoiDetector / KitchenDetector / NutAssemblyDetector)
   → Bộ dò trạng thái ký hiệu từ quan sát thực tế
          │
          ▼
3. Mỗi episode:
   a. Reset môi trường (có thể thêm nhiễu vị trí vật thể)
   b. Gọi bộ lập kế hoạch PDDL:
      - Detector → trạng thái ký hiệu (predicates: on, clear, grasped…)
      - add_predicates_to_pddl() → ghi trạng thái vào file PDDL
      - define_goal_in_pddl() → ghi mục tiêu vào file PDDL
      - call_planner() → gọi Metric-FF → chuỗi hành động (plan)
   c. Thực thi từng bước trong kế hoạch:
      - TaskOperation (Pick/Place/TurnOn/TurnOff) dẫn robot đến mục tiêu
      - Có thể thêm nhiễu ngẫu nhiên vào hành động (noisy fraction)
   d. Lưu quỹ đạo (observations, actions, rewards) vào file .zip (pickle)
```

**Tham số quan trọng khi chạy:**
```bash
python -m dataset_making.main \
  --env Hanoi \
  --episodes 150 \
  --noise-std 0.03 \
  --noisy-fraction 0.30 \
  --random-block-placement \
  --random-block-selection
```

---

### Bước 2: Xử Lý & Chuyển Đổi Dữ Liệu

**Thư mục:** `neuro_symbolic_method/data_processing/`

**Mục đích:** Chuyển đổi dữ liệu thô (pickle/zip) sang định dạng **Zarr** tối ưu cho việc huấn luyện Diffusion Policy.

**File chính:** `data_to_zarr.py`

**Luồng chi tiết:**

```
Thư mục datasets/ (file .zip chứa .pkl)
          │
          ▼
load_data_from_zip() / load_buffer()
→ Đọc danh sách episode: (episode_buffer, symbolic_buffer, task)
          │
          ▼
convert_to_dict_format()
→ Chuyển từng transition thành dict {obs, action, reward, done}
          │
          ▼
Phân tách theo kỹ năng (ReachPick / Pick / ReachDrop / Drop)
→ Mỗi kỹ năng tạo ra một tập dữ liệu riêng
          │
          ▼
Lưu thành file .zarr (dùng thư viện zarr + numcodecs)
→ Cấu trúc: data/obs, data/action, meta/episode_ends
```

**Cấu trúc dữ liệu Zarr đầu ra:**
```
dataset.zarr
├── data/
│   ├── obs          # Quan sát: proprio, ảnh camera
│   └── action       # Hành động robot (7-DOF: x,y,z,rx,ry,rz,gripper)
└── meta/
    └── episode_ends # Chỉ số kết thúc của từng episode
```

---

### Bước 3a: Huấn Luyện Diffusion Policy

**Thư mục:** `neuro_symbolic_method/diffusion_policy/`

**Mục đích:** Huấn luyện **4 chính sách kỹ năng** riêng biệt cho từng giai đoạn thao tác (dựa trên Diffusion Transformer):

| Kỹ năng | Vai trò |
|---------|---------|
| `reach_pick` | Di chuyển tay robot đến gần vật thể cần lấy |
| `grasp` (pick) | Cầm nắm vật thể |
| `reach_drop` | Di chuyển robot đến vị trí đặt vật thể |
| `drop` | Thả vật thể xuống đúng vị trí |

**Kiến trúc mô hình:**
- **Diffusion Transformer (lowdim)**: Học phân phối xác suất hành động điều kiện theo quan sát.
- Đầu vào: Chuỗi quan sát gần đây (proprio + có thể thêm ảnh).
- Đầu ra: Chuỗi hành động dự đoán (action chunk).

**Tham số cấu hình (YAML):**
```yaml
# Ví dụ từ config/hanoi.yaml
policies:
  grasp:      policies/noisy_30/grasp.ckpt
  drop:       policies/noisy_30/drop.ckpt
  reach_pick: policies/noisy_30/reach_pick.ckpt
  reach_place: policies/noisy_30/reach_drop.ckpt
```

**Đầu ra:** File checkpoint `.ckpt` (PyTorch) cho mỗi kỹ năng.

---

### Bước 3b: Huấn Luyện YOLO & Regressor

**Thư mục:** `neuro_symbolic_method/objects_detection/`

#### 3b-1: Huấn luyện YOLO (Nhận diện vật thể)

**Mục đích:** Phát hiện và theo dõi vật thể trong ảnh camera để ánh xạ từ ID ngữ nghĩa PDDL (cube1, peg2…) sang vị trí pixel.

**File notebook:** `train_yolov8_object_detection_on_custom_dataset.ipynb`

**Luồng:**
```
Ảnh từ camera (agentview)
          │
          ▼
Gán nhãn bounding box (YOLOv8 format)
          │
          ▼
Huấn luyện YOLOv8 (ultralytics)
          │
          ▼
Lưu model: models/yolo/hanoi_yolo.pt
```

#### 3b-2: Huấn luyện Regressor (Ước tính vị trí 3D)

**Mục đích:** Từ bounding box 2D trong ảnh → ước tính tọa độ 3D của vật thể trong không gian robot.

**Luồng:**
```
Bounding box 2D (x, y, w, h) + thông tin camera
          │
          ▼
Trích xuất đặc trưng (tọa độ pixel trung tâm, kích thước bbox)
          │
          ▼
Huấn luyện mô hình hồi quy (scikit-learn, lưu bằng joblib)
          │
          ▼
Lưu model: models/regressors/hanoi_regressor.pkl
```

---

### Bước 4: Xây Dựng Bộ Lập Kế Hoạch PDDL

**Thư mục:** `neuro_symbolic_method/planning/PDDL/`

**Mục đích:** Định nghĩa bài toán lập kế hoạch ký hiệu cho từng môi trường. Không cần huấn luyện – chỉ cần viết đúng file PDDL.

**Cấu trúc PDDL:**
```
planning/PDDL/<env>/
├── domain.pddl        # Định nghĩa loại vật, predicates, actions
├── problem_save.pddl  # Template bài toán (trạng thái ban đầu rỗng)
└── problem_dummy.pddl # File bài toán được cập nhật tại runtime
```

**Ví dụ Domain Hanoi:**
```pddl
(define (domain hanoi)
  (:predicates
    (on ?disk - disk ?location - location)
    (clear ?location - location)
    (grasped ?disk - disk)
    (smaller ?disk - disk ?location - location)
    (free-gripper))

  (:action pick ...)   ; Nhấc đĩa
  (:action place ...)  ; Đặt đĩa
)
```

**Bộ lập kế hoạch sử dụng:** **Metric-FF v2.1** (fast-forward heuristic planner)
```bash
./Metric-FF-v2.1/ff -o domain.pddl -f problem_dummy.pddl -s 0
```

---

### Bước 5: Chạy Thử Nghiệm & Đánh Giá

**File chính:** `neuro_symbolic_method/experiments_neurosymbolic.py`

**Mục đích:** Chạy hệ thống đầy đủ để đánh giá hiệu suất của phương pháp Neuro-Symbolic trên các môi trường mục tiêu.

**Luồng chi tiết tại runtime:**

```
1. Tải cấu hình YAML (config/hanoi.yaml)
          │
          ▼
2. Tải các mô hình đã huấn luyện:
   - YOLO (.pt) → nhận diện vật thể
   - Regressor (.pkl) → ước tính vị trí 3D
   - Diffusion Policy checkpoints (.ckpt) × 4 kỹ năng
          │
          ▼
3. Khởi tạo môi trường Robosuite + Detector
          │
          ▼
4. Mỗi episode:
   ┌─────────────────────────────────────────┐
   │ a. Reset môi trường                      │
   │ b. Quan sát trạng thái ban đầu           │
   │    → Detector → predicates ký hiệu       │
   │ c. Lập kế hoạch PDDL                    │
   │    add_predicates_to_pddl()              │
   │    define_goal_in_pddl()                 │
   │    call_planner() → [pick obj1 peg2, …]  │
   │                                          │
   │ d. Thực thi từng bước kế hoạch:          │
   │    Với mỗi cặp (pick obj, place peg):    │
   │    ① ReachPick (Diffusion Policy)        │
   │    ② Grasp     (Diffusion Policy)        │
   │    ③ ReachDrop (Diffusion Policy)        │
   │    ④ Drop      (Diffusion Policy)        │
   │    → YOLO tracking + Regressor           │
   │      cung cấp vị trí vật thể realtime    │
   │ e. Kiểm tra điều kiện kết thúc (Beta)    │
   │ f. Reset tay robot về vị trí gốc        │
   └─────────────────────────────────────────┘
          │
          ▼
5. Ghi kết quả vào file + W&B logging:
   - Success rate (tỉ lệ thành công toàn episode)
   - Pick-place success rate (tỉ lệ thành công từng thao tác)
   - Mean percentage advancement (% tiến độ trung bình)
```

**Tham số quan trọng khi chạy:**
```bash
python neuro_symbolic_method/experiments_neurosymbolic.py \
  --env Hanoi \
  --n_ep 100 \
  --n_act 4 \
  --seed 0
```

---

## Luồng Đầy Đủ Phương Pháp VLA (pi0)

Đây là phương pháp so sánh với Neuro-Symbolic. Thay vì chia nhỏ kỹ năng, pi0 học **end-to-end** toàn bộ nhiệm vụ.

### Huấn Luyện pi0

```
Thu thập Demo (dataset_making/)
          │
          ▼
Chuyển đổi sang định dạng RLDS
(rlds_dataset_builder/)
          │
          ▼
Fine-tune mô hình pi0
(openpi/ – dùng Docker)
          │
          ▼
Lưu checkpoint vào openpi/checkpoints/
```

### Đánh Giá pi0

```bash
# Từ thư mục openpi/
docker compose -f examples/robosuite/compose.yml up
# → Khởi động policy server pi0
# → Client kết nối WebSocket, gửi quan sát, nhận hành động
```

**Hai chế độ đánh giá:**

| Chế độ | Mô tả | use_sequential_tasks |
|--------|-------|----------------------|
| End-to-End | Một prompt duy nhất, robot tự quyết định toàn bộ | `False` |
| Planner-Guided | Planner PDDL cung cấp sub-goal từng bước | `True` |

---

## Sơ Đồ Quan Hệ Giữa Các Module

```
robosuite/ ──────────────────────► Môi trường mô phỏng
     │
     ├── dataset_making/ ──────────► Thu thập demo (oracle)
     │        │
     │        └── planning/ ──────► Lập kế hoạch PDDL + Metric-FF
     │
     ├── neuro_symbolic_method/
     │        ├── data_processing/ ► Chuyển đổi sang Zarr
     │        ├── diffusion_policy/► Huấn luyện chính sách kỹ năng
     │        ├── objects_detection► Huấn luyện YOLO + Regressor
     │        ├── models/ ─────────► Lưu trữ model (.pt, .pkl, .ckpt)
     │        ├── planning/ ───────► PDDL domain + executor runtime
     │        ├── config/ ─────────► Cấu hình YAML cho từng môi trường
     │        └── experiments_neurosymbolic.py ► Điểm chạy đánh giá
     │
     ├── openpi/ ─────────────────► VLA pipeline (pi0)
     │        └── checkpoints/ ───► Lưu model pi0 fine-tuned
     │
     └── analysis/ ───────────────► Phân tích kết quả thực nghiệm
```

---

## Tóm Tắt Nhanh – Checklist Huấn Luyện

```
□ 1. Cài đặt môi trường (git submodule update --init --recursive)
□ 2. Thu thập demo (python -m dataset_making.main --env Hanoi --episodes 150)
□ 3. Chuyển đổi dữ liệu sang Zarr (data_to_zarr.py)
□ 4a. Huấn luyện YOLO (notebook objects_detection/)
□ 4b. Huấn luyện Regressor
□ 5. Huấn luyện 4 Diffusion Policy (reach_pick, grasp, reach_drop, drop)
□ 6. Kiểm tra PDDL domain files
□ 7. Chạy đánh giá (experiments_neurosymbolic.py)
```

---

## Các Môi Trường Được Hỗ Trợ

| Môi trường | Mô tả | Detector | PDDL Domain |
|------------|-------|----------|-------------|
| `Hanoi` | Tháp Hà Nội 3 đĩa | `HanoiDetector` | `hanoi/` |
| `Hanoi4x3` | Tháp Hà Nội 4 đĩa | `HanoiDetector` | `hanoi4x3/` |
| `KitchenEnv` | Môi trường nhà bếp | `KitchenDetector` | `kitchen/` |
| `NutAssembly` | Lắp ráp đai ốc | `NutAssemblyDetector` | `nut_assembly/` |
| `CubeSorting` | Phân loại khối | `CubeSortingDetector` | `cubesorting/` |
| `HeightStacking` | Xếp chồng theo độ cao | `HeightStackingDetector` | `heightstacking/` |
| `AssemblyLineSorting` | Phân loại dây chuyền | `AssemblyLineSortingDetector` | `assemblylinesorting/` |
| `PatternReplication` | Sao chép mẫu | `PatternReplicationDetector` | `patternreplication/` |

---

## Ghi Chú Cho Người Học

- **Predicates PDDL** là ngôn ngữ ký hiệu dùng để mô tả trạng thái thế giới (ví dụ: `on(cube1, peg2)`, `clear(peg1)`).
- **Diffusion Policy** là mô hình sinh ra hành động bằng cách học từ dữ liệu demo qua quá trình khuếch tán ngược (reverse diffusion).
- **YOLO** (You Only Look Once) là mô hình nhận diện vật thể realtime theo bounding box.
- **Metric-FF** là bộ lập kế hoạch AI cổ điển, tìm chuỗi hành động tối ưu từ trạng thái hiện tại đến mục tiêu trong không gian ký hiệu.
- Hệ thống **Neuro-Symbolic** kết hợp: phần **ký hiệu** (PDDL planner) quyết định *làm gì*, phần **thần kinh** (Diffusion Policy) quyết định *làm như thế nào*.
