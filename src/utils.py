import torch

# Jalankan satu epoch pelatihan
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for img, leaf, disease, _ in loader:
        img = img.to(device)
        target = disease.to(device)  # bisa ganti ke `leaf` tergantung task

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# Evaluasi model (tanpa update bobot)
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    for img, leaf, disease, _ in loader:
        img = img.to(device)
        target = disease.to(device)

        output = model(img)
        loss = criterion(output, target)
        total_loss += loss.item()

    return total_loss / len(loader)