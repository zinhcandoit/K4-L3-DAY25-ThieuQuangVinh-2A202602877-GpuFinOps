# NimbusAI — Báo Cáo Kỹ Thuật Tối Ưu Hóa Chi Phí GPU (FinOps)

> **Vị trí:** Kỹ sư FinOps (FinOps Solution Engineer)  
> **Người nhận:** Ban Lãnh đạo & Đội ngũ Kỹ thuật NimbusAI / Giảng viên hướng dẫn AICB  
> **Học phần:** AICB · Phase 2 · Track 2 (Infrastructure) · Day 25 Lab Deliverable  

---

## 1. So Sánh Chi Phí: Baseline vs. Optimized

| Chỉ số kinh tế | Trước tối ưu (Baseline) | Sau tối ưu (Optimized) | Chênh lệch / Tiết kiệm |
|---|---|---|---|
| **Tổng chi phí hàng tháng** | **$27,133** | **$14,626** | **Tiết kiệm $12,507** (**46.1%**) |
| **Đơn giá suy luận ($/1M-token)** | **$6.488** / 1M token | **$1.126** / 1M token | **Giảm $5.362** (**82.6%**) |
| **Chi phí suy luận hàng ngày** | $48.87 / ngày | $8.48 / ngày | Tiết kiệm $40.39 / ngày |
| **Chi phí hạ tầng GPU hàng tháng** | $25,667 / tháng | $15,627 / tháng | Tiết kiệm $10,040 / tháng |

---

## 2. Phân Tích Chi Tiết Từng Đòn Bẩy Tiết Kiệm (Levers Breakdown)

```
Tổng tiết kiệm hàng tháng: $12,507
├── Chiến lược mua GPU (Spot + 3yr Reserved):   $10,040  (80.3%)
├── Đòn bẩy phần mềm Inference (Cascade + Cache + Batch): $1,212   (9.7%)
├── Right-sizing các GPU bị "Util-Lie":         $655     (5.2%)
└── Thu hồi GPU nhàn rỗi (Kill Idle):           $600     (4.8%)
```

### Đòn bẩy đóng góp nhiều nhất: Chiến lược mua sắm ($10,040/tháng — 80.3%)
- **Tại sao đóng góp lớn nhất?**  
  Bởi vì NimbusAI trước đây chạy 100% on-demand với giá niêm yết đầy đủ. Bằng cách phân loại workload theo tính chất:
  1. **Workload ổn định 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`):** Đạt duty cycle 100%, vượt xa ngưỡng hòa vốn (break-even utilization là 55% với mức chiết khấu 45% của hợp đồng Reserved 3 năm). Chuyển sang Reserved giúp cắt giảm gần một nửa chi phí.
  2. **Workload huấn luyện/đánh giá có thể ngắt quãng (`job-train-llm`, `job-train-embed`, `job-finetune`, v.v.):** Chuyển sang Spot Instances kết hợp lưu checkpoint định kỳ. Dù tính thêm chi phí lưu checkpoint (3%) và thời gian chạy lại do gián đoạn (rework), chi phí thực tế vẫn tiết kiệm ròng 39.1% so với on-demand.

### Đòn bẩy phần mềm Inference ($1,212/tháng — Giảm 82.6% đơn giá token):
Áp dụng ngăn xếp chiết khấu nhân tử (multiplicative discount stack):
- **Cascade Routing:** Chuyển 80% truy vấn đơn giản sang model nhỏ ($0.20/1M input vs $3.00/1M của model lớn — rẻ hơn 15 lần).
- **Prompt Caching:** Tận dụng system prompt tĩnh của chat/RAG để hưởng chiết khấu 90% (chỉ trả 10% giá input thông thường).
- **Batch API:** Gom các job đánh giá (eval) không cần thời gian thực để hưởng giảm giá 50%.

### Right-sizing GPU "Util-Lie" ($655/tháng):
Hạ cấp `gpu-h100-4` (98.2% util nhưng chỉ đạt 0.194 MFU) sang GPU A100 có throughput tương đương, giảm chênh lệch giá thuê từ $2.50/giờ xuống $1.79/giờ.

### Tắt GPU nhàn rỗi qua đêm ($600/tháng):
Phát hiện và tự động tắt `gpu-h100-5` bị bỏ không 8 tiếng ban đêm (<10% util), thu hồi $20/ngày lãng phí.

---

## 3. Bản Chất Của "GPU-Util Lie" & Tác Động Tài Chính

- **Bản chất đo lường:** Lệnh `nvidia-smi` báo cáo `gpu_util_pct` dựa trên tỷ lệ thời gian xung nhịp (clock) hoạt động trong cửa sổ lấy mẫu 1 giây. Nó **không đo lường** số lượng phép tính thực sự (FLOPs) mà GPU sinh ra.
- **Các trường hợp vi phạm trong hệ thống:**
  - `gpu-h100-4`: Báo cáo **98.2% GPU-Util** nhưng MFU (Model FLOPs Utilization) chỉ đạt **0.194 (19.4%)**.
  - `gpu-a10g-1`: Báo cáo **96.9% GPU-Util** nhưng MFU chỉ đạt **0.268 (26.8%)**.
- **Nguyên nhân kỹ thuật gốc rễ (Root Cause):**
  - **Nghẽn băng thông bộ nhớ (Memory-bound):** Quá trình decode tự hồi quy (autoregressive decode) của LLM với batch size nhỏ có cường độ tính toán (arithmetic intensity) rất thấp (~1–2 FLOP/byte), nằm dưới điểm uốn (ridge point 295 FLOP/byte) của mô hình Roofline trên H100. Tensor Cores bị nghẽn (stall) đến 80% thời gian để chờ nạp trọng số từ bộ nhớ HBM.
  - **Overhead khởi chạy kernel & chưa fuse kernel:** Nhiều kernel nhỏ chạy tuần tự làm GPU luôn bận rộn trên danh nghĩa xung nhịp nhưng hiệu suất thực rất thấp.
- **Tác động tài chính:** NimbusAI đang trả trọn giá thuê H100 ($2.50/giờ) cho một khối lượng tính toán mà một GPU rẻ hơn nhiều có thể đáp ứng. Right-sizing trường hợp này tiết kiệm ngay lập tức **$655/tháng** mà không làm tăng độ trễ dịch vụ.

---

## 4. Các Phần Mở Rộng Đã Triển Khai (Your Turn Extensions)

### Extension 3: Kinh Tế Học Prompt Caching (`cache_is_worth_it`)
- **Mô hình tính điểm hòa vốn:** Ghi cache phát sinh chi phí ban đầu. Cache chỉ có lợi khi tổng tiền tiết kiệm từ các lần đọc tiếp theo bù đắp được chi phí ghi:
  $$\text{Break-even reads } N = \frac{\text{write\_cost}}{(1 - \text{read\_discount}) \times \text{read\_price}} = \frac{1.0}{0.90} \approx \mathbf{1.11\text{ lần đọc}}$$
- **Đo lường trên dữ liệu:** Với chiết khấu đọc 90% (`read_discount=0.10`), chỉ cần đọc lại từ 2 lần trở lên là có lãi (cho cả model nhỏ $0.20 và model lớn $3.00). Trong file [token_usage.csv](file:///c:/Users/VINH/OneDrive%20-%20VNU-HCMUS/Attachments/Desktop/AI20K/Day25-Track2-GpuFinOps/data/token_usage.csv), các system prompt của đội Chat và RAG được đọc lại trung bình **12.0 lần**.
- **Kết luận:** Hàm `cache_is_worth_it()` trả về `True`, chứng minh việc áp dụng cache mang lại ROI dương rõ rệt.

### Extension 4: Định Lượng Ngân Sách Suy Luận Reasoning
- **Sự phân hóa giữa lượng truy vấn và năng lượng tiêu thụ:**
  - Lượng truy vấn reasoning: Chỉ chiếm **8.4% tổng số truy vấn** (201 / 2,400) và **16.5% tổng số token** (1.24M / 7.53M).
  - Năng lượng tiêu thụ: Chiếm tới **94.0% tổng năng lượng suy luận** (29,787.7 Wh trên tổng 31,675.3 Wh).
  - Năng lượng trung bình: Mỗi truy vấn reasoning ngốn **148.20 Wh** so với chỉ **0.86 Wh** của truy vấn thông thường (**gấp 172.6 lần**, do nhân tử năng lượng 80× kết hợp với độ dài sinh token chuỗi suy nghĩ dài gấp 6×).
- **Quy tắc định tuyến đề xuất:** Thiết lập bộ phân loại độ phức tạp tác vụ; chỉ kích hoạt reasoning khi độ tin cậy < 0.75 hoặc tác vụ giải toán/lập trình chuyên sâu. Giới hạn tỷ lệ reasoning xuống 4% traffic sẽ tiết kiệm **14.8 kWh/ngày (444.6 kWh/tháng)** và giảm đáng kể tải nhiệt GPU.

### Extension 5: Điều Phối Lịch Trình Thân Thiện Với Carbon (Carbon-Aware Scheduling)
- **Quy mô:** Đánh giá 5 job huấn luyện/đánh giá có thể ngắt quãng, tiêu thụ **4,227.0 kWh/tháng**.
- **So sánh 5 vùng đám mây:**
  - `us-east-1` (Mặc định): 380 gCO2/kWh, $0.120/kWh $\rightarrow$ Phát thải **1,606.3 kg CO2e**, tiền điện $507.24/tháng.
  - `europe-north1` (Sạch nhất - Thủy điện Na Uy): 30 gCO2/kWh, $0.090/kWh $\rightarrow$ Phát thải **126.8 kg CO2e**, tiền điện $380.43/tháng.
  - `us-east-wa` (Rẻ nhất - Thủy điện Washington): 90 gCO2/kWh, $0.055/kWh $\rightarrow$ Phát thải **380.4 kg CO2e**, tiền điện $232.49/tháng.
  - `us-west-2` (Cân bằng): 120 gCO2/kWh, $0.070/kWh $\rightarrow$ Phát thải **507.2 kg CO2e**, tiền điện $295.89/tháng.
  - `europe-central2` (Bẩn nhất - Nhiệt điện than Ba Lan): 660 gCO2/kWh, $0.180/kWh $\rightarrow$ Phát thải **2,789.8 kg CO2e**, tiền điện $760.86/tháng.
- **Hiệu quả:** Di chuyển các job ngắt quãng sang `europe-north1` giúp giảm **1,479.5 kg CO2e/tháng (giảm 92.1% lượng phát thải)**, đồng thời tiết kiệm thêm **$126.81/tháng** tiền điện. Vì đây là các job batch bất đồng bộ, việc đổi vùng **không gây bất kỳ ảnh hưởng nào đến độ trễ người dùng**.

---

## 5. Khuyến Nghị Dành Cho NimbusAI (3 Hành Động Đầu Tiên Của FinOps Lead)

1. **Khóa hợp đồng Reserved và kích hoạt Spot ngay trong tuần 1:**  
   Ký cam kết Reserved 3 năm cho 3 job suy luận 24/7 và chuyển toàn bộ job train/dev sang Spot có checkpoint tự động. Thu hồi ngay lập tức **$10,040/tháng** mà không cần thay đổi kiến trúc mã nguồn mô hình.
2. **Cấu hình Gateway định tuyến Cascade & Prompt Caching trong tuần 2:**  
   Triển khai bộ định tuyến tại tầng API gateway (như LiteLLM) để cascade 80% truy vấn đơn giản sang model nhỏ và bật cache tiền tố hệ thống. Động thái này giúp giảm ngay **82.6% đơn giá trên mỗi 1M token**.
3. **Kích hoạt Chargeback và chính sách tự động dọn dẹp trong tuần 3–4:**  
   Hiện tại độ phủ tag đã đạt **92%** (vượt ngưỡng kiểm soát 80%), đủ điều kiện chuyển từ Showback sang Chargeback thực tế cho 4 phòng ban (`assistant`, `search`, `eval`, `rag`). Thiết lập script watchdog tự động kill các GPU có tải <10% kéo dài quá 1 tiếng ngoài giờ làm việc.
