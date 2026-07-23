# Hướng Dẫn Git Workflow (Pull & Push) Trên Máy Mới

Tài liệu này hướng dẫn chi tiết cách thiết lập, đồng bộ (`git pull`) và cập nhật (`git push`) mã nguồn dự án **Human-Mimicking-UR5** trên bất kỳ máy tính mới nào (Laptop/PC Ubuntu, Windows, macOS).

---

## 📌 Thông Tin Repository
- **URL Repository:** `https://github.com/baotran2312/Human-Mimicking-UR5`
- **Branch chính:** `main`

---

## 🚀 1. Thiết Lập Ban Đầu Trên Máy Mới (Thiết lập 1 lần duy nhất)

### Bước 1.1: Clone dự án về máy
Mở Terminal trên máy mới và chạy lệnh clone:
```bash
git clone https://github.com/baotran2312/Human-Mimicking-UR5.git
cd Human-Mimicking-UR5
```

### Bước 1.2: Cấu hình thông tin tác giả (Git User Identity)
Thiết lập tên và email đại diện của bạn trên Git (chỉ cần chạy 1 lần trên máy):
```bash
git config --global user.name "baotran2312"
git config --global user.email "baotran2312@users.noreply.github.com"
```

### Bước 1.3: Cấu hình tự động lưu Token (Credential Helper)
Đảm bảo bạn không phải gõ lại Token mỗi lần thực hiện `git push`:
```bash
git config --global credential.helper store
```

---

## 🔑 2. Tạo Và Cấu Hình GitHub Personal Access Token (PAT)

Từ năm 2021, GitHub **không cho phép** dùng mật khẩu tài khoản để `git push`. Bạn phải dùng **Personal Access Token (PAT)** làm mật khẩu.

### Bước 2.1: Cách tạo PAT trên GitHub (Tokens Classic - Đơn giản nhất)
1. Đăng nhập vào [GitHub](https://github.com).
2. Nhấp vào ảnh đại diện (góc trên cùng bên phải) $\rightarrow$ **Settings**.
3. Cuộn xuống cuối menu bên trái, chọn **Developer settings**.
4. Chọn **Personal access tokens** $\rightarrow$ **Tokens (classic)**.
5. Nhấn **Generate new token** $\rightarrow$ **Generate new token (classic)**.
6. **Note:** Điền tên gợi nhớ (ví dụ: `Laptop-Lab` hoặc `PC-Home`).
7. **Expiration:** Chọn `No expiration` hoặc `90 days`.
8. **Select scopes:** Tick chọn ô vuông **`repo`** (Full control of private repositories).
9. Nhấn nút **Generate token** ở cuối trang.
10. **LƯU Ý:** Copy chuỗi token bắt đầu bằng `ghp_...` (Chuỗi này chỉ xuất hiện 1 lần duy nhất).

---

## 📥 3. Quy Trình Cập Nhật Code Mới Nhất (`git pull`)

Trước khi bắt đầu code hoặc chạy thử nghiệm, bạn nên luôn cập nhật code mới nhất từ GitHub về máy:

```bash
# 1. Chuyển vào thư mục dự án
cd ~/Baro/Human-Mimicking-UR5

# 2. Kiểm tra trạng thái làm việc (đảm bảo không có thay đổi chưa commit bị dở dang)
git status

# 3. Kéo code mới nhất từ nhánh main trên GitHub về máy
git pull origin main
```

---

## 📤 4. Quy Trình Đẩy Code Mới Lên GitHub (`git push`)

Khi bạn thêm file mới, sửa code hoặc tạo dữ liệu mới (ví dụ file trong thư mục `data/`):

### Bước 4.1: Kiểm tra các file đã thay đổi
```bash
git status
```

### Bước 4.2: Đưa các file muốn đẩy lên bộ đệm (Staging)
```bash
# Đưa toàn bộ thay đổi (code, file mới) vào bộ đệm:
git add .

# Hoặc chỉ đưa từng thư mục/file cụ thể:
git add src/ docs/ data/
```

### Bước 4.3: Tạo Commit kèm mô tả thay đổi
```bash
git commit -m "feat: Cap nhat thuat toan IK va them luu file CSV"
```

### Bước 4.4: Đẩy code lên GitHub (`git push`)

#### Cách A: Push tiêu chuẩn (Nếu đã lưu Credential Store ở Bước 1.3)
```bash
git push origin main
```
*Lần đầu tiên chạy, Git sẽ hỏi `Username` (nhập `baotran2312`) và `Password` (dán mã **Token `ghp_...`** vào). Các lần sau Git sẽ tự động nhớ và không hỏi lại.*

#### Cách B: Push trực tiếp bằng Remote URL gắn sẵn Token (Cách 1 dòng lệnh)
Nếu máy mới chưa lưu Credential, bạn có thể gán trực tiếp Token vào URL để push ngay lập tức:
```bash
git remote set-url origin https://baotran2312:YOUR_PAT_TOKEN@github.com/baotran2312/Human-Mimicking-UR5.git
git push origin main
```
*(Thay `YOUR_PAT_TOKEN` bằng chuỗi token `ghp_...` của bạn)*

---

## 🛠️ 5. Xử Lý Các Lỗi Thường Gặp

### Lỗi 1: `fatal: Could not read from remote repository` hoặc `fatal: 'main' does not appear to be a git repository`
* **Nguyên nhân:** Chưa cài đặt đúng remote URL cho repository.
* **Cách khắc phục:**
  ```bash
  git remote remove origin 2>/dev/null
  git remote add origin https://github.com/baotran2312/Human-Mimicking-UR5.git
  git fetch origin
  git branch -M main
  ```

### Lỗi 2: `error: 403 Permission denied`
* **Nguyên nhân:** Token bị sai, hết hạn, hoặc Token dạng Fine-Grained thiếu quyền **Write**.
* **Cách khắc phục:** 
  1. Tạo lại Token Classic với tick chọn quyền **`repo`** theo mục 2.1.
  2. Cập nhật lại URL với token mới:
     ```bash
     git remote set-url origin https://baotran2312:ghp_TOKEN_MOI@github.com/baotran2312/Human-Mimicking-UR5.git
     git push origin main
     ```

### Lỗi 3: `error: Your local changes to the following files would be overwritten by merge`
* **Nguyên nhân:** Code trên máy cục bộ bị sửa đổi trùng với code trên GitHub khi chạy `git pull`.
* **Cách khắc phục:**
  ```bash
  # Tạm thời cất các thay đổi cục bộ đi:
  git stash
  # Kéo code mới về:
  git pull origin main
  # Lấy lại các thay đổi cục bộ:
  git stash pop
  ```

---

## ⚡ Bảng Lệnh Tóm Tắt (Cheat Sheet)

| Thao tác | Lệnh thực hiện |
| :--- | :--- |
| **Clone máy mới** | `git clone https://github.com/baotran2312/Human-Mimicking-UR5.git` |
| **Kiểm tra trạng thái** | `git status` |
| **Kéo code mới nhất** | `git pull origin main` |
| **Staging tất cả** | `git add .` |
| **Commit thay đổi** | `git commit -m "mo ta thay doi"` |
| **Đẩy code lên GitHub** | `git push origin main` |
| **Xóa lưu pass cũ** | `git credential-approve` / Reset remote URL |

