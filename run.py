import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from config import Config
from preprocess import read_sen_pairs
from model import SentenceBERT, BertClassifier
from dataset import SentencePairDataset, SingleBertDataset
from train_eval import train, evaluate, accuracy
import os

config = Config()
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=config.logging_file_name,
    filemode='a+',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

if __name__ == '__main__':
    train_sen_a_list, train_sen_b_list, train_labels = read_sen_pairs(config.train_path)
    test_sen_a_list, test_sen_b_list, test_labels = read_sen_pairs(config.test_path)

    train_set = SingleBertDataset(train_sen_a_list, train_sen_b_list, train_labels, config)
    test_set = SingleBertDataset(test_sen_a_list, test_sen_b_list, test_labels, config)

    # TODO
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertClassifier(config).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    logger.info("START")
    best_model_name = ""
    best_accuracy = 0.0
    for epoch in range(config.num_epochs):
        train_loss = train(train_loader, model, loss, optimizer, device)
        logger.info(f"Epoch: [{epoch + 1} / {config.num_epochs}]  | Train Loss: {train_loss}")
        acc = accuracy(test_loader, model, device)
        logger.info(f"Epoch: [{epoch + 1} / {config.num_epochs}] | Test ACC: {acc}")
        if acc > best_accuracy:
            best_accuracy = acc
            if best_model_name != '':
                os.remove(best_model_name)
            best_model_name = f"{config.save_path.split('.')[0]}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), best_model_name)
            logger.info(f"Model saved in {best_model_name} @Epoch {epoch + 1}")

    model.load_state_dict(torch.load(best_model_name))
    acc = accuracy(test_loader, model, device)
    logger.info(f"Best Model '{best_model_name}' | Test ACC: {acc}")





