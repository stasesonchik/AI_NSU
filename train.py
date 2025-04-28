# train.py
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split

# --- Настройки ---

def main():
    list_of_heroes = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble',
                      'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler',
                      'comic_book_guy', 'disco_stu', 'edna_krabappel', 'fat_tony', 'gil', 'groundskeeper_willie',
                      'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lionel_hutz', 'lisa_simpson',
                      'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover',
                      'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner',
                      'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel',
                      'snake_jailbird', 'troy_mcclure', 'waylon_smithers']




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    learning_rate = 1e-3
    batch_size = 64


    from torch.optim import lr_scheduler


    # --- Датасет ---
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])



    dataset = datasets.ImageFolder('simpsons_dataset_v2/train/', transform=preprocess)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)




    # --- Модель ---
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(list_of_heroes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-5)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                        steps_per_epoch=len(train_loader),
                                        epochs=epochs, pct_start=0.3)



    train_losses, val_losses = [], []
    class_accuracy = {cls: [] for cls in list_of_heroes}

    def evaluate_model(model, loader):
        model.eval()
        correct = {cls:0 for cls in list_of_heroes}
        total   = {cls:0 for cls in list_of_heroes}
        tot_corr = 0; tot_samp = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                for lbl, pr in zip(labels, preds):
                    name = list_of_heroes[lbl.item()]
                    total[name] += 1
                    if lbl==pr:
                        correct[name] += 1
                        tot_corr += 1
                    tot_samp += 1

        for cls in list_of_heroes:
            acc = 100*correct[cls]/total[cls] if total[cls]>0 else 0.0
            class_accuracy[cls].append(acc)
        return tot_corr/tot_samp

    # --- Тренировка ---
    for epoch in range(epochs):
        model.train()
        running = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward(); optimizer.step(); scheduler.step()
            running += loss.item()
        train_losses.append(running/len(train_loader))

        # валидация
        model.eval()
        vloss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                vloss += criterion(model(images), labels).item()
        val_losses.append(vloss/len(val_loader))

        val_acc = evaluate_model(model, val_loader)
        print(f"[{epoch+1}/{epochs}] "
              f"TrainLoss={train_losses[-1]:.4f} "
              f"ValLoss={val_losses[-1]:.4f} "
              f"ValAcc={val_acc:.2%}")

    # сохраняем модель и матрицу метрик
    torch.save(model.state_dict(), "batch128/resnet18_simpsons.pth")
    acc_matrix = np.stack([class_accuracy[c] for c in list_of_heroes], axis=0)
    np.save("batch128/acc_matrix.npy", acc_matrix)
    print("Training complete, model and acc_matrix saved.")


if __name__ == "__main__":
    # Для Windows-пакетов (опционально):
    from multiprocessing import freeze_support
    freeze_support()
    main()