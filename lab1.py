from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch

# Список персонажей
list_of_heroes = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble',
                  'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler',
                  'comic_book_guy', 'disco_stu', 'edna_krabappel', 'fat_tony', 'gil', 'groundskeeper_willie',
                  'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lionel_hutz', 'lisa_simpson',
                  'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover',
                  'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner',
                  'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel',
                  'snake_jailbird', 'troy_mcclure', 'waylon_smithers']

# Подготовка изображений
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка данных
train_dataset = datasets.ImageFolder(root='simpsons_dataset_v2/train/', transform=preprocess)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 42
epochs = 20
learning_rate = 1e-3

# Загружаем предобученный ResNet-18 и меняем выходной слой
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# Функция потерь, оптимизатор, шедулер
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), weight_decay=1e-5)
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                    steps_per_epoch=len(train_dataloader),
                                    epochs=epochs, pct_start=0.3)

# Функции сохранения и загрузки
def save_model(model, path="resnet18_simpsons.pth"):
    torch.save(model.state_dict(), path)
    print(f"Модель сохранена в {path}")

def load_model(path="resnet18_simpsons.pth"):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    print(f"Модель загружена из {path}")
    return model

# Функция оценки точности
def evaluate_model(model, dataloader):
    model.eval()
    class_correct = {cls: 0 for cls in list_of_heroes}
    class_total = {cls: 0 for cls in list_of_heroes}
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for label, pred in zip(labels, predicted):
                class_total[list_of_heroes[label.item()]] += 1
                if label == pred:
                    class_correct[list_of_heroes[label.item()]] += 1
                    total_correct += 1
                total_samples += 1

    print(f"\nОбщая точность: {100 * total_correct / total_samples:.2f}%")

    for cls in list_of_heroes:
        if class_total[cls] > 0:
            acc = 100 * class_correct[cls] / class_total[cls]
            print(f"{cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

# Цикл обучения
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"Начало эпохи {epoch+1}/{epochs}")

    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Эпоха {epoch+1}, Батч {batch_idx}, Loss: {loss.item():.4f}")

    print(f"Эпоха {epoch+1} завершена. Средний Loss: {running_loss/len(train_dataloader):.4f}")

# Сохранение модели
save_model(model)

# Оценка точности
evaluate_model(model, train_dataloader)
