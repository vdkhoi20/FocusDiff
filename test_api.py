import requests
import os
import io
from PIL import Image
import numpy as np

# Định nghĩa URL của API endpoint
API_URL = "http://127.0.0.1:8001/edit_image"

def test_api():
    """Gửi một yêu cầu POST tới API bằng cách sử dụng file mask.png có sẵn."""
    
    # Tên file mask
    mask_filename = 'panoramic_mask.png'
    
    # Kiểm tra xem file mask.png có tồn tại không
    if not os.path.exists(mask_filename):
        print(f"Lỗi: Không tìm thấy file '{mask_filename}'.")
        print("Vui lòng đảm bảo file mask.png nằm trong cùng thư mục với client.py.")
        return
        
    print(f"Đang đọc file mask: {mask_filename}")
    
    # Mở và đọc file mask
    try:
        with open(mask_filename, 'rb') as f:
            mask_bytes = f.read()
            
    except Exception as e:
        print(f"Lỗi khi đọc file '{mask_filename}': {e}")
        return

    # Chuẩn bị dữ liệu cho yêu cầu POST
    files = {'mask': (mask_filename, mask_bytes, 'image/png')}
    data = {
        'id_image': '1',  # Thay đổi ID hình ảnh này để kiểm tra các hình ảnh khác nhau
        'prompt': 'A teddy bear'
    }
    
    print(f"Đang gửi yêu cầu tới API tại {API_URL}...")
    print(f"Thông tin yêu cầu: id_image='{data['id_image']}', prompt='{data['prompt']}'")
    
    try:
        response = requests.post(API_URL, data=data, files=files)
        response.raise_for_status()  # Ném lỗi nếu mã trạng thái không phải 200 OK
        
        print(f"Yêu cầu thành công! Mã trạng thái: {response.status_code}")
        
        # Lưu hình ảnh đã chỉnh sửa
        output_path = "edited_image.png"
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Hình ảnh đã được chỉnh sửa đã được lưu tại: {output_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"Đã xảy ra lỗi khi kết nối hoặc gửi yêu cầu: {e}")
        print(f"Vui lòng đảm bảo server FastAPI đang chạy tại {API_URL}.")
    except Exception as e:
        print(f"Đã xảy ra lỗi không xác định: {e}")

if __name__ == "__main__":
    test_api()
