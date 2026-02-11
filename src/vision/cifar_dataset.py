"""
CIFAR-10 dataset for Llamdex image classification.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import AutoTokenizer
from typing import Optional


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset that returns images and text prompts for LLM classification.
    
    Args:
        root: Root directory for CIFAR-10 data
        train: Whether to use training set (default: True)
        tokenizer: Tokenizer for text prompts
        system_prompt: System prompt (default: fixed prompt for classification)
        transform: Optional image transforms (default: standard CIFAR-10 transforms)
    """
    
    def __init__(
        self,
        root: str = './data/cifar10',
        train: bool = True,
        tokenizer: Optional[AutoTokenizer] = None,
        system_prompt: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.train = train
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = "You are a classifier. Answer with only one token: a digit 0-9."
        self.system_prompt = system_prompt
        
        # Default user prompt
        self.user_prompt = "What is the label?"
        
        # Load CIFAR-10 dataset
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform if transform is not None else self._get_default_transform(),
        )
        
        self.tokenizer = tokenizer
        if tokenizer is not None:
            # Pre-compute prompt tokens
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt}
            ]
            self.prompt_tokens = tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            ).squeeze(0)
            self.attention_mask = (self.prompt_tokens != tokenizer.pad_token_id).long()
            
            # Get token IDs for digits 0-9
            self.digit_token_ids = []
            for digit in range(10):
                digit_str = str(digit)
                token_id = tokenizer.encode(digit_str, add_special_tokens=False)
                if len(token_id) == 1:
                    self.digit_token_ids.append(token_id[0])
                else:
                    # If digit is split into multiple tokens, use the first one
                    # This should be rare for digits
                    self.digit_token_ids.append(token_id[0])
                    print(f"Warning: Digit '{digit_str}' tokenizes to {len(token_id)} tokens, using first token")
        
        # Class names for CIFAR-10
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def _get_default_transform(self):
        """Get default CIFAR-10 transforms."""
        if self.train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Image is already transformed (Tensor of shape (3, 32, 32))
        
        result = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }
        
        if self.tokenizer is not None:
            # Add prompt tokens and target token ID
            result["tokens"] = self.prompt_tokens.clone()
            result["attention_mask"] = self.attention_mask.clone()
            result["target_token_id"] = torch.tensor(self.digit_token_ids[label], dtype=torch.long)
        
        return result


def get_cifar10_collate_fn(pad_token_id: int):
    """
    Collate function for CIFAR-10 dataset.
    
    Args:
        pad_token_id: Padding token ID
        
    Returns:
        Collate function
    """
    def collate_fn(batch):
        # Stack images
        images = torch.stack([item["image"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        
        # Stack tokens and attention masks
        tokens = torch.stack([item["tokens"] for item in batch])
        attention_masks = torch.stack([item["attention_mask"] for item in batch])
        target_token_ids = torch.stack([item["target_token_id"] for item in batch])
        
        return {
            "images": images,
            "labels": labels,
            "tokens": tokens,
            "attention_mask": attention_masks,
            "target_token_ids": target_token_ids,
        }
    
    return collate_fn
