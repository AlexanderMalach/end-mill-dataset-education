import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import create_model
from data_preprocessing import ToolDataset
import os
from tqdm import tqdm

# Проверка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры
batch_size = 4
num_epochs = 10
learning_rate = 1e-4
save_model_path = 'model.pth'

# Пути к данным
annotation_file = r'D:/ay/P_project/tool_endmill_dataset_coco/annotations/instances_Test.json'
image_dir = r'D:/ay/P_project/tool_endmill_dataset_coco/images/Test'

assert os.path.exists(annotation_file), f"Файл {annotation_file} не найден!"
assert os.path.exists(image_dir), f"Директория {image_dir} не найдена!"

# Преобразования для изображений
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Преобразования для масок (без нормализации)
mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # Без нормализации
])

# Подготовка данных
train_dataset = ToolDataset(annotation_file, image_dir, image_transform=image_transform, mask_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Инициализация модели
model = create_model(num_classes=4)  # Количество классов в масках
model.to(device)

# Определение оптимизатора и функции потерь
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Цикл обучения
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, masks) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        import torch.nn.functional as F

        # Приводим маску к размеру выхода модели
        masks_resized = F.interpolate(masks, size=outputs.shape[2:], mode="nearest")  # [batch, num_classes, h, w]

        # Берем argmax и переводим в long
        loss = criterion(outputs, masks_resized.argmax(dim=1).long())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}")
            running_loss = 0.0

    torch.save(model.state_dict(), save_model_path)
    print(f"Модель сохранена после эпохи {epoch + 1}")

print("Обучение завершено")
