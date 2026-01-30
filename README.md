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
QDRANT_API_KEY=...
GROQ_API_KEY=...
OPENAI_API_KEY=...
```
Sử dụng OPENAI_API_KEY khi cần chạy test


## Bước 3: Chạy app

```bash
streamlit run app.py
```