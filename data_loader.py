import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import random
import json


class CaptDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json_path, vocab, vocab_size, max_seq_length, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.files = self.load_files(json_path)
        self.vocab = vocab.word2idx
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.transform = transform

    def load_files(self, json_path):
        files = []
        files_name = {}
        raw_data = json.load(open(json_path, "r"))

        names = raw_data['images']
        anns  = raw_data['annotations'] 

        for name in names: 
            files_name[name['id']] = name['file_name']

        for ann in anns:
            files.append({'path':files_name[ann['image_id']], 'caption':ann['caption']})

        return files

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        file  = self.files[index]
        path, caption = file['path'], file['caption']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        caption = []
        caption.append(self.vocab['<start>'])
        for token in tokens:
            if token in self.vocab:
                caption.append(self.vocab[token]) 
            else: 
                caption.append(self.vocab['<unk>'])        
        caption.append(self.vocab['<end>'])
        while len(caption) < self.max_seq_length:
            caption.append(self.vocab['<end>'])
        if len(caption) > self.max_seq_length:
            caption = caption[:self.max_seq_length]
        caption = torch.Tensor(caption)

        return image, caption

    def __len__(self):
        return len(self.files)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    captions = torch.stack(captions, 0).long()

    return images, captions


def build_datasets(args, vocab):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Build data loader
    train_dataset = CaptDataset(root=args.train_dir.replace('dataset', args.dataset),
                                json_path=args.train_caption_path.replace('dataset', args.dataset),
                                vocab=vocab,
                                vocab_size=len(vocab),
                                max_seq_length=args.max_seq_length,
                                transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=collate_fn)

    return train_loader