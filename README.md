# Hướng dẫn chạy project

Tài liệu này hướng dẫn cách thiết lập môi trường và chạy project từ đầu.

---

## Yêu cầu hệ thống

- Đã cài đặt **Conda** (Miniconda hoặc Anaconda)
- Python >= 3.9
- Hệ điều hành: Windows / Linux / macOS

---

## Bước 1: Tạo môi trường từ file `environment.yml`

Tại thư mục gốc của project, chạy lệnh:

```bash
conda env create -f environment.yml
```
## Bước 2: Bạn cần thêm các api key cần thiết vào biến môi trường (file .env)

```bash
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.yEUcdzP5fUfezmLK07kMVaDwrS5cBrD_Z5MTg1s7k14
GROQ_API_KEY=...
OPENAI_API_KEY=...
```
API KEY của qdrant chỉ có hiệu lực trong thời gian ngắn, vui lòng liên hệ nếu cần quyền truy cập
Sử dụng OPENAI_API_KEY khi cần chạy test


## Bước 3: Chạy app

```bash
streamlit run app.py
```