from tqdm import tqdm
import torch


def train(data_loader, model, loss, optimizer, device, scheduler=None):
    model.train()
    final_loss = 0.0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        label = data.pop('label')
        outputs = model(**data)
        ls = loss(outputs, label)

        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        final_loss += ls.item()

    return final_loss / len(data_loader)


def evaluate(data_loader, model, loss, device):
    model.eval()
    final_loss = 0.0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('label')
            outputs = model(**data)
            ls = loss(outputs, label)
            final_loss += ls.item()

    return final_loss / len(data_loader)


def accuracy(data_loader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('label')
            outputs = model(**data)

            predicts = outputs.argmax(dim=1)
            total += predicts.size(0)
            correct += (predicts == label).sum().item()

    return correct / total
