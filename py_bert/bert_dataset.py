from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch

class PYBERTDataset(Dataset):
    def __init__(self, contents, targets, tokenizer, max_len):
        self.contents = contents
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, item):
        content = str(self.contents[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'document_text': content,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }