from torch.utils.data import Dataset
import torch
import json
import os
from pathlib import Path
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence


class CasualLLMDataset(Dataset):
    def __init__(self, file_path: Path, tokenizer: PreTrainedTokenizer, max_length: int = 512, dialogue: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_list = []
        with open(file_path, encoding='utf-8') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokenized = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
                input_ids = tokenized['input_ids'][0]
                attention_mask = tokenized['attention_mask'][0]
                labels = input_ids.clone()
                self.train_list.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, item):
        return self.train_list[item]

    def collate_fn(self, batch):
        input_ids = [sample['input_ids'] for sample in batch]
        attention_masks = [sample['attention_mask'] for sample in batch]
        labels = [sample['labels'] for sample in batch]

        pad_token_id = self.tokenizer.pad_token_id

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }


class SFTAlpacaDataset(Dataset):
    def __init__(self, file_path: Path, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_list = []
        with open(file_path, encoding='utf-8') as f:
            json_data = json.load(f)
            for data in json_data:
                instruction = data['instruction']
                input = data['input']
                response = data['output']
                system_prompt = data['system']


class MSTSDataset(Dataset):
    def __init__(self, file_path: Path, prob_path: Path, tokenizer: PreTrainedTokenizer, max_length: int = 512,
                 dialogue: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_list = []
        with open(file_path, encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                json_data = json.loads(line)
                prefix = json_data['prefix']
                generation = json_data['generation']
                text = prefix + generation
                tokenized = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
                input_ids = tokenized['input_ids'][0]
                attention_mask = tokenized['attention_mask'][0]
                labels = input_ids.clone()
                labels[:len(prefix)] = -100
                self.train_list.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })

        with open(prob_path, 'rb') as f:
            prob_tensor = torch.load(f)
            assert prob_tensor.shape[0] == len(self.train_list)
            for i, prob in enumerate(prob_tensor):
                self.train_list[i]['prob'] = prob

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, item):
        return self.train_list[item]

    def collate_fn(self, batch):
        input_ids = [sample['input_ids'] for sample in batch]
        attention_masks = [sample['attention_mask'] for sample in batch]
        labels = [sample['labels'] for sample in batch]
        probs = [sample['prob'] for sample in batch]
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels,
            'probs': probs
        }
