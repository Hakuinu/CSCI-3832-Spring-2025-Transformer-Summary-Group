# This was written by me using the previous homeworks with some help from DeepSeek
import os, random, sys, matplotlib.pyplot as plt
import torch, torch.nn as nn, numpy
from tqdm import tqdm  
from nltk.tokenize import word_tokenize

import numpy as np 
import pandas as pd 

from transformers import AutoTokenizer, AutoModel
import torch
# used these tutorials
# https://medium.com/@rakeshrajpurohit/getting-started-with-hugging-face-transformer-models-and-tokenizers-5b46bfc6573
# https://huggingface.co/learn/llm-course/en/chapter2/4
class TransformerTokenizer:
    def __init__(self, model_name='bert-base-uncased', max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

    def tokenize(self, texts):
        """Convert texts to transformer-ready input format"""
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']

class TransformerEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def embed(self, input_ids, attention_mask):
        """Convert tokenized inputs to embeddings"""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device)
            )
        return outputs.last_hidden_state[:, 0, :].cpu()  # [CLS] embeddings

if __name__ == "__main__":
    # Loading the data
    df = pd.read_csv("archive/reviews.csv", encoding='ISO-8859-1')
    texts = df['Text'].tolist()[:1000]
    
    # Initialize the tokenizer + embedder
    tokenizer = TransformerTokenizer(max_length=128)
    embedder = TransformerEmbedder()
    
    # Tokenize it
    input_ids, attention_mask = tokenizer.tokenize(texts)
    
    # Process in batches for efficiency 
    batch_size = 32
    all_embeddings = []
    
    # Create progress bar with total calculated, this part was written with help from DeepSeek
    num_batches = (len(texts) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=num_batches, desc="Generating embeddings")
    
    for i in range(0, len(texts), batch_size):
        batch_ids = input_ids[i:i+batch_size]
        batch_mask = attention_mask[i:i+batch_size]
        
        embeddings = embedder.embed(batch_ids, batch_mask)
        all_embeddings.append(embeddings)
        progress_bar.update(1)  # Manually update progress
    
    progress_bar.close()
    
    # Combine results
    final_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Generated embeddings shape: {final_embeddings.shape}")
    # saving to a file for testing
    torch.save(final_embeddings, 'review_embeddings.pt')