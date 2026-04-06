import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Định nghĩa lại đúng kiến trúc model (phải giống hệt trên Kaggle)
class MLP_Fashion(nn.Module):
    def __init__(self):
        super(MLP_Fashion, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return self.fc3(x)

# 2. Load model đã train
device = torch.device('cpu')  # Dùng CPU trên máy local
model = MLP_Fashion().to(device)
model.load_state_dict(torch.load('fashion_mlp.pth', map_location=device))
model.eval()

# 3. Danh sách nhãn
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 4. Hàm dự đoán
def predict(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_tensor = transforms.ToTensor()(img)
    img_tensor = 1 - img_tensor  # Đảo màu cho giống Fashion-MNIST
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    plt.imshow(img_tensor.squeeze(), cmap='gray')
    plt.title(f'{class_names[pred]} ({conf*100:.1f}%)')
    plt.axis('off')
    plt.show()

# 5. Sử dụng
predict('b8e261a9b9cd3c3961c2ccbc652f57af.jpg')  # Đổi thành đường dẫn ảnh của bạn