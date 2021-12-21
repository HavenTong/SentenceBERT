from transformers import BertTokenizer
import os


class Config:
    def __init__(self, batch_size=16, num_epochs=5, lr=2e-5):
        self.current_path = os.path.dirname(__file__)
        self.train_path = os.path.join(self.current_path, 'data/train_data.tsv')
        self.test_path = os.path.join(self.current_path, 'data/test_data.tsv')
        self.model_name = os.path.join(self.current_path, 'chinese-macbert-base')
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.sen_a_max_len = 100
        self.sen_b_max_len = 136
        self.max_len = 256
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = 2
        self.dropout_rate = 0.3
        self.learning_rate = lr
        self.save_path = f"macbert_{self.batch_size}_{self.num_epochs}.pth"
        self.logging_file_name = f'macbert_logging_{self.batch_size}_{self.num_epochs}.log'


if __name__ == '__main__':
    config = Config()
    print(config.current_path)
