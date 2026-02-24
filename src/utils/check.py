import torch
import os

# Đường dẫn file checkpoint bị lỗi cấu hình
checkpoint_path = '/mnt/e/AI/khanh/checkpoints/unit2unit/checkpoint_last.pt'

if os.path.exists(checkpoint_path):
    print(f"Dang doc file: {checkpoint_path}...")
    # Load checkpoint
    state = torch.load(checkpoint_path)
    
    # Kiểm tra xem args có bị dính share_all_embeddings không
    if hasattr(state['args'], 'share_all_embeddings'):
        print(f"Trang thai cu: share_all_embeddings = {state['args'].share_all_embeddings}")
        
        # SỬA LỖI: Ép nó về False
        state['args'].share_all_embeddings = False
        print(f"-> Da sua thanh: share_all_embeddings = False")
        
        # Lưu đè lại file
        torch.save(state, checkpoint_path)
        print("✅ Đã lưu file checkpoint đã sửa lỗi thành công!")
    else:
        print("Checkpoint này không có tham số share_all_embeddings. Có thể lỗi do nguyên nhân khác.")
else:
    print("❌ Không tìm thấy file checkpoint!")