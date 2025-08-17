# 🎯 Personal AI Project – Sentiment Analysis  

## 👋 Giới thiệu  
Xin chào! Đây là một trong những dự án **AI cá nhân** mình xây dựng để vừa học, vừa rèn luyện kỹ năng, đồng thời có thể giới thiệu trong **Portfolio / CV**.  

Trong dự án này, mình tập trung vào **Phân tích cảm xúc (Sentiment Analysis)** từ dữ liệu IMDb Reviews – một bài toán kinh điển trong NLP.  

👉 Thay vì sử dụng sẵn các mô hình như BERT, mình muốn **tự xây dựng mô hình deep learning** để:  
- Hiểu rõ cách **embedding, CNN, LSTM** hoạt động trong xử lý ngôn ngữ.  
- Tự **tinh chỉnh siêu tham số** thay vì chỉ fine-tune mô hình có sẵn.  
- Rèn kỹ năng triển khai mô hình học sâu cho bài toán thực tế.  

---

## 🛠️ Công nghệ & Công cụ  
- **Python, PyTorch** – xây dựng và huấn luyện mô hình  
- **CNN + LSTM hybrid** – trích xuất đặc trưng và học ngữ cảnh  
- **Grid Search** – tối ưu hyperparameters (learning rate, dropout, hidden size)  
- **IMDb Dataset** – tập dữ liệu chuẩn cho sentiment analysis  

---

## 📂 Cấu trúc dự án  
Sentiment-Project/
│── data/ # IMDb dataset (đã xử lý train/test)
│── embeddings/ # File embedding imdbEr.txt
│── src/
│ ├── main.py # Script chính để chạy training
│ ├── model.py # Kiến trúc CNN + LSTM
│ ├── train.py # Hàm train + grid search
│ ├── utils.py # Xử lý dữ liệu, embedding
│── README.md # Giới thiệu dự án


---

## 🚀 Cách chạy  
    Cài đặt requirements:  
    bash
    pip install -r requirements.txt
    Chạy huấn luyện
    python src/main.py
    Tuỳ chỉnh grid search trong main.py
    param_grid = {
        "lr": [1e-3, 5e-4],
        "hidden_dim": [128, 256],
        "dropout": [0.3, 0.5]
    }

## 📊 Kết quả sơ bộ

Loss ban đầu ~0.6 → giảm dần sau vài epoch

Accuracy trên test set kỳ vọng đạt 80–85% (sau khi tuning)


## 🌱 Điều mình học được

✔️ Xử lý dữ liệu văn bản và xây dựng embedding matrix
✔️ Hiểu cơ chế CNN trong NLP và cách nó kết hợp với LSTM
✔️ Kỹ năng training + debugging mô hình trong PyTorch
✔️ Ứng dụng Grid Search để tối ưu siêu tham số

## 🔮 Hướng phát triển tiếp

So sánh trực tiếp với baseline (Logistic Regression, simple LSTM)

Thêm visualization (loss/accuracy curves)

Thử nghiệm với embedding pre-trained (GloVe, FastText)

Deploy mô hình thành API nhỏ để demo



##  💡 Why this project matters
- Dự án này không chỉ rèn luyện kiến thức kỹ thuật mà còn:

- Thể hiện khả năng tự nghiên cứu & xây dựng mô hình từ đầu

- Cho thấy mình có thể debug, tối ưu, và triển khai mô hình AI

- Minh chứng cho tư duy giải quyết vấn đề thực tế bằng NLP + Deep Learning