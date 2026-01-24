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

## Bước 2: Chạy app

```bash
streamlit run app.py
```