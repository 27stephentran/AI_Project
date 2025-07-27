import torch

# Model hyperparameters
EMBEDDING_DIM = 1  # Vì imdbEr.txt chỉ có 1 giá trị cảm xúc cho mỗi từ
HIDDEN_DIM = 128
DROPOUT = 0.5

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

# Other configurations
MAX_LEN = 300  # Độ dài tối đa của review sau khi padding
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")