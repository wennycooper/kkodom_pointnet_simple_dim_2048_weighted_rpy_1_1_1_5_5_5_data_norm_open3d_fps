import json
import matplotlib.pyplot as plt

# 從 loss.json 檔案中讀取 JSON 訓練紀錄
with open('loss_history.json', 'r') as f:
    data = json.load(f)

# 提取訓練和驗證損失
train_loss = data["train_loss"]
val_loss = data["val_loss"]
epochs = range(1, len(train_loss) + 1)

# 繪製損失曲線
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

