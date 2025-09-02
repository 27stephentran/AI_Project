# 🧑‍🎓 Emotion Recognition with FER-2013

## 1. Mục tiêu dự án
- Xây dựng **một mô hình nhận diện cảm xúc từ khuôn mặt** (không phân biệt cá nhân cụ thể).
- Dữ liệu sử dụng: [FER-2013 trên HuggingFace](https://huggingface.co/datasets/Jeneral/fer-2013).
- 7 loại cảm xúc: `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`.

## 2. Kiến thức cần có
- **Machine Learning cơ bản**: Train/Test/Validation split, Loss function, Optimizer.
- **Deep Learning**:
  - CNN (Convolutional Neural Network) → xử lý ảnh.
  - CrossEntropyLoss → dùng cho phân loại nhiều lớp.
  - Optimizer (Adam).
- **PyTorch**:
  - Dataset & DataLoader.
  - Tensor, GPU training.
- **Computer Vision**:
  - Ảnh grayscale (48x48).
  - Chuẩn hóa ảnh trước khi đưa vào model.

## 3. Cấu trúc dự án
```
Emotion_Project/
│── src/
│   ├── dataset.py        # Load dataset, tách train/val/test
│   ├── model.py          # Định nghĩa CNN
│── result/               # Lưu model sau khi train
│── train.py              # Script huấn luyện chính
│── README.md             # Hướng dẫn dự án
```

## 4. Cách chạy
### Cài đặt môi trường
```bash
pip install torch torchvision datasets pandas pillow
```

### Train model
```bash
python train.py
```

Sau khi chạy xong, model được lưu ở:
```
result/emotion_cnn.pth
```

## 5. Điểm nổi bật
- **Dataset tự động load từ HuggingFace** → không cần tải thủ công.
- **Validation set** được tạo từ 10% train → giúp theo dõi overfitting.
- **Model CNN đơn giản** → dễ hiểu, thích hợp cho Fresher và Sinh viên.

## 6. Hướng phát triển thêm
- Thử các kiến trúc mạnh hơn (ResNet, EfficientNet).
- Áp dụng Data Augmentation để cải thiện accuracy.
- Xây dựng API Flask/FastAPI để demo nhận diện cảm xúc từ webcam.
