# Đồ án Học sâu — CNN TensorFlow/Keras và ONNX

Mở **[casting-cnn-tensorflow.ipynb](casting-cnn-tensorflow.ipynb)**. Đây là notebook mới để bạn tự thực hiện và giải thích bài, chưa có cell nào được chạy, chưa có trọng số hay kết quả TensorFlow. Không cần backend, thiết kế hệ thống hoặc giao diện web.

Bản PyTorch và kết quả đã chạy trước đây được giữ trong `archive_pytorch/`, không dùng làm kết quả cho bản TensorFlow này. `data/raw/` giữ nguyên dữ liệu local để bạn sử dụng.

## Mô hình dùng trong bài

```text
Ảnh xám 256 × 256, giá trị 0–255
→ Rescaling(1/255)
→ Conv2D + BatchNormalization + ReLU
→ 3 khối SeparableConv2D + BatchNormalization + ReLU + MaxPooling2D
→ GlobalAveragePooling2D
→ Dense(64) + Dropout(0.3)
→ Dense(1, sigmoid)
```

Đây là CNN tự xây gọn, không dùng trọng số pretrained. So với bản Flatten/Dense ban đầu: SeparableConv2D tách tích chập theo kênh và trộn kênh để giảm tính toán; Batch Normalization chuẩn hóa đặc trưng trong quá trình train; Global Average Pooling giảm số tham số của phần phân loại. Những kỹ thuật này đã được dùng lâu, không phải kiến trúc mới năm 2026 và không bảo đảm chính xác hơn nếu chưa thực nghiệm.

Giữ ảnh xám vì dữ liệu đang dùng không có thông tin màu cần khai thác. Chọn 256 thay vì 128 để giữ thêm chi tiết; đây là giả thuyết cần đánh giá, không phải kết luận 256 luôn tốt hơn. Kích thước được khai báo một nơi và dùng đồng bộ cho mô hình/ONNX.

## 1. Bạn tự chuẩn bị môi trường

**Cách đơn giản để dùng GPU: Google Colab.** Tải notebook, `requirements.txt` và dữ liệu lên Colab/Drive; chọn runtime GPU. Trong một ô riêng, tự chạy:

```python
%pip install -r requirements.txt
```

Nếu Colab yêu cầu restart runtime sau khi cài, restart rồi mới import TensorFlow. Cài thư viện không phải huấn luyện. Sửa `DATA_DIR` thành thư mục trực tiếp chứa `def_front/` và `ok_front/`; tải thư mục đầu ra về trước khi phiên Colab kết thúc.

**Trên Windows/VS Code:** dùng Python 3.11 hoặc 3.12, tạo môi trường riêng để không trộn với PyTorch cũ:

```powershell
cd D:\Casting-Defect-Detection-CNN
py -3.12 -m venv .venv-tf
.\.venv-tf\Scripts\python.exe -m pip install --upgrade pip
.\.venv-tf\Scripts\python.exe -m pip install -r requirements.txt
```

Mở notebook, bấm **Select Kernel → Python Environments → .venv-tf**. Nếu không hiện, chọn đường dẫn `.venv-tf\Scripts\python.exe`. Không dùng kernel cũ chỉ vì nó đã cài TensorFlow phiên bản khác.

TensorFlow từ 2.11 trở lên không hỗ trợ CUDA GPU trên Windows native. Với cách cài trên, việc danh sách GPU là `[]` không có nghĩa máy mất RTX 3060: notebook sẽ dùng CPU. Muốn train TensorFlow bằng GPU, dùng Colab hoặc Linux/WSL2 theo [hướng dẫn TensorFlow](https://www.tensorflow.org/install/pip). Chưa có phép đo thời gian chạy cho notebook mới.

Các phiên bản TensorFlow/Keras/tf2onnx được ghi cố định để giảm thay đổi môi trường. Tổ hợp này được lựa chọn từ tài liệu, **chưa được cài hoặc kiểm thử trong phiên làm việc này**. Nếu gặp lỗi, giữ nguyên thông báo lỗi và các phiên bản được in ở mục 1 để cùng xử lý, không đổi thư viện hàng loạt.

## 2. Trình tự bạn tự chạy và kiểm tra

Chưa dùng **Run All** khi chưa hiểu notebook. Chạy từng ô bằng Shift+Enter, đọc kết quả trước khi tiếp tục.

| Mục trong notebook | Bạn làm gì | Cần quan sát |
|---|---|---|
| 1. Môi trường | Chạy ô import | Đúng Python/kernel, TensorFlow/Keras đúng phiên bản, GPU có hoặc không |
| 2. Dữ liệu | Sửa DATA_DIR, đọc ảnh và kiểm tra | Hai lớp, số ảnh, ảnh hỏng/trùng, ảnh minh họa |
| 3. Chia tập | Chạy chia train/validation/test | Không giao nhau; hai lớp có mặt ở cả ba tập |
| 4. Tiền xử lý | Đọc hàm chuẩn bị batch | Kích thước 256 × 256 × 1, đầu vào 0–255, chưa chia 255 ngoài model |
| 5. CNN | Chạy model.summary | Giải thích SeparableConv2D, BatchNorm, GAP, Dropout và sigmoid |
| 6. Kiểm tra nhẹ | Chạy batch qua model ở chế độ suy luận | Shape đầu ra (batch, 1), xác suất hữu hạn trong [0,1]; chưa cập nhật trọng số |
| 7. Không augmentation | Tự chạy cell model.fit đầu tiên | Train/validation loss và accuracy qua từng epoch |
| 8. Có augmentation | Tự chạy cell model.fit thứ hai | Cùng split và cấu hình, chỉ khác augmentation |
| 9. Chọn và đánh giá | Chọn bằng validation rồi mới xem test | Report, ma trận nhầm lẫn, đường cong và ảnh sai |
| 10. Xuất ONNX | Tự chạy export | Có file ONNX và kiểm tra cấu trúc thành công |
| 11. Kiểm chứng ONNX | Tự đối chiếu với Keras | Sai số xác suất trong ngưỡng, nhãn khớp |
| 12. Một ảnh | Sửa IMAGE_PATH và chạy suy luận | Xác suất lớp lỗi/đạt được ánh xạ đúng |

Mục 6 chỉ là kiểm tra dữ liệu và đầu ra, không chứng minh chất lượng. Mục 7 và 8 mới là những ô thực sự train. Mục 11 kiểm tra chuyển đổi mô hình, không thay thế đánh giá trên test.

## 3. Huấn luyện 30 epoch và đối chứng

Mặc định `EPOCHS = 30`, `USE_EARLY_STOPPING = False`: mỗi thí nghiệm chạy đủ 30 epoch nếu không bị gián đoạn. Vẫn lưu checkpoint có validation loss tốt nhất, không mặc định epoch cuối là tốt nhất. ReduceLROnPlateau giảm learning rate khi validation loss chững lại, không dừng huấn luyện.

Nếu muốn dừng khi overfit, tự bật `USE_EARLY_STOPPING = True`; lúc đó có thể kết thúc trước 30 epoch. Chạy lâu hơn không tự bảo đảm chất lượng.

Muốn thử quy trình ngắn trước, bạn có thể tự đặt `RUN_NAME = "thu_quy_trinh"`, `EPOCHS = 1` rồi chạy cả hai thí nghiệm. Kết quả này chỉ để kiểm tra luồng, không dùng làm kết quả chính. Sau đó restart kernel, đổi `RUN_NAME = "lan_01"`, đặt lại 30 epoch và làm thực nghiệm chính. Không cần bước thử một epoch nếu bạn muốn tự chạy chính thức ngay.

Giữ cùng split, seed, batch size, kiến trúc và learning rate giữa hai cấu hình. Augmentation được áp dụng online trên train, không trên validation/test và không nằm trong mô hình export. Không coi ảnh augmentation là sản phẩm mới độc lập. Chỉ chọn mô hình bằng validation loss trước khi xem kết quả test.

## 4. File bạn sẽ tạo ra

Sau khi **bạn chạy**, file nằm trong `outputs_tensorflow/<RUN_NAME>/`:

- `data_check.json`, `split.json`, `config.json`: dữ liệu và cấu hình lần chạy.
- `cnn_no_aug.keras`, `cnn_with_aug.keras`, `history_*.csv`: checkpoint và lịch sử.
- `selection.json`, `comparison.csv`, `classification_report.txt`, `test_predictions.csv`: lựa chọn và đánh giá.
- `learning_curves.png`, `confusion_matrix.png`, `errors.png` nếu có ảnh sai.
- `cnn_casting.onnx`, `metadata.json`: mô hình triển khai và ý nghĩa đầu vào/đầu ra.

Notebook không ghi đè checkpoint đã có. Khi làm lại thí nghiệm, chọn `RUN_NAME` mới và restart kernel trước khi chạy lại từ đầu. Hai thí nghiệm phải thuộc cùng một lần chạy/cấu hình.

## 5. Hiểu đúng sigmoid và ONNX

Nhãn **0 = `def_front` (lỗi)**, **1 = `ok_front` (đạt)**. Sigmoid trả về `p_ok`; do đó `p_defect = 1 - p_ok`. Không ghi nhầm đầu ra sigmoid là xác suất lỗi.

Input ONNX là float32 **NHWC** `[batch, 256, 256, 1]`, giá trị **0–255**, khác input NCHW của bản PyTorch cũ. Lớp Rescaling đã nằm trong model; chia 255 thêm lần nữa bên ngoài sẽ sai. Đầu ra là `[batch, 1]`, đã qua sigmoid, không thêm softmax.

Cell xuất dùng [Keras Model.export(format="onnx")](https://keras.io/api/models/model_saving_apis/export/) với backend TensorFlow và tf2onnx. Cell kiểm chứng kế tiếp dùng cùng batch ảnh validation cho Keras/ONNX Runtime, so sánh xác suất và nhãn. Nếu assert không qua, dừng ở đó và xem lỗi trước khi làm demo. Cell này chưa được chạy.

## 6. Nội dung bạn cần tự viết trong báo cáo

Giải thích nguồn/gán nhãn dữ liệu, tỷ lệ lớp, cách chia tập, ảnh hỏng/trùng, vai trò từng lớp CNN, ảnh hưởng augmentation và dấu hiệu overfitting. Nhận xét recall lớp lỗi, không chỉ accuracy. Đưa ví dụ ảnh sai và giải thích có cơ sở, không bịa nguyên nhân.

Một seed và tập test nhỏ chưa đủ kết luận mô hình tốt trong sản xuất. Hash chỉ phát hiện trùng pixel, chưa xử lý ảnh gần giống hoặc cùng lô. Chưa kiểm chứng camera, ảnh ngoài miền hoặc hiệu chuẩn xác suất. Đây là phân loại cả ảnh, không xác định vị trí vết lỗi.

Nguồn dataset được ghi ở dự án tham khảo: [Casting Product Image Data trên Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product). Cần kiểm tra điều kiện sử dụng và bổ sung URL repo gốc vào báo cáo. Dữ liệu local trước đây có 1.300 ảnh (781 lỗi, 519 đạt); bạn phải đọc lại thống kê khi chạy, không coi đó là kết quả đánh giá mới.

Code TensorFlow mới được soạn với hỗ trợ AI. Tài liệu tham khảo: [SeparableConv2D](https://keras.io/api/layers/convolution_layers/separable_convolution2d/), [GlobalAveragePooling2D](https://keras.io/api/layers/pooling_layers/global_average_pooling2d/), [BatchNormalization](https://keras.io/api/layers/normalization_layers/batch_normalization/). Tuân theo quy định của giảng viên về sử dụng code tham khảo và AI.
