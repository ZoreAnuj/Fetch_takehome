device = 'cpu'
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
class SentenceTransformer(nn.Module):
    def __init__(self, 
                 model_name: 'bert-base-uncased', 
                 pooling_method: 'mean', 
                 output_dim: 768, 
                 add_mlp: False ):
      super().__init__()

      # Load tranformer and tokenizer from HuggigFace
      self.transformer = AutoModel.from_pretrained(model_name)
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.pooling_method = pooling_method

      # Initialize attention layer
      if pooling_method == 'attention': self.attention = nn.Linear(self.transformer.config.hidden_size, 1)

      # Initialize post-processing layers
      if add_mlp:
        self.post_processor = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, output_dim*2),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim),
            nn.LayerNorm(output_dim)
        )

      else:
        self.post_processor = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, output_dim),
            nn.LayerNorm(output_dim)
        )
    # Pooling transformer output
    def pool_output(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
      if self.pooling_method == 'cls': return hidden_states[:, 0]

      # applying attention on the hidden states
      elif self.pooling_method == 'attention':
        attention_weights = self.attention(hidden_states)
        attention_weights = attention_weights.masked_fill(~attention_mask.bool().unsqueeze(-1), float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=1)
        return torch.sum(hidden_states*attention_weights, dim=1)

      # if not just unsqueeze
      else:
        attention_mask = attention_mask.unsqueeze(-1)
        return (hidden_states*attention_mask).sum(1) / attention_mask.sum(1)


    # Function for encoding sentences into batches
    def encode(self, sentences: Union[str, List[str]], batch_size: int =32, **encode_kwargs) -> torch.Tensor:
      if isinstance(sentences, str): sentences = [sentences]
      all_embeddings = []
      for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        embeddings = self.forward(batch, **encode_kwargs)
        all_embeddings.append(embeddings)
      return torch.cat(all_embeddings, dim=0)

    # Basically we are doing: tokenize -> transform -> pool -> post-processing
    def forward(self, sentences: Union[str, List[str]], return_dict = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
      # Tokenize
      encoded = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
      # Get transformer output
      outputs = self.transformer(**encoded)
      hidden_states = outputs.last_hidden_state
      # Pool outputs
      pooled = self.pool_output(hidden_states, encoded['attention_mask'])
      # Post-process
      embeddings = self.post_processor(pooled)

      if return_dict:
        return{'embeddings':embeddings, 'hidden_states': hidden_states, 'pooler_output': pooled}
      else:
        return embeddings
    
    

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 pooling_method: str = 'mean',
                 embedding_dim: int = 768,
                 num_classes: int = 3,
                 num_sentiments: int = 3,
                 add_mlp: bool = False):
        super().__init__()

        # Class labels
        self.class_labels = ['News', 'Technical', 'Casual']
        self.sentiment_labels = ['Negative', 'Neutral', 'Positive']

        # Base transformer and tokenizer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling_method = pooling_method

        if pooling_method == 'attention':
            self.attention = nn.Linear(self.transformer.config.hidden_size, 1)

        if add_mlp:
            self.embedding_layer = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        else:
            self.embedding_layer = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )

        # Task-specific heads
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, num_classes)
        )

        self.sentiment_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, num_sentiments)
        )

    def pool_output(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_method == 'cls':
            return hidden_states[:, 0]
        elif self.pooling_method == 'attention':
            attention_weights = self.attention(hidden_states)
            attention_weights = attention_weights.masked_fill(~attention_mask.bool().unsqueeze(-1), float('-inf'))
            attention_weights = torch.softmax(attention_weights, dim=1)
            return torch.sum(hidden_states * attention_weights, dim=1)
        else:  # mean pooling
            attention_mask = attention_mask.unsqueeze(-1)
            return (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)

    def forward(self, sentences: Union[str, List[str]], task: Optional[str] = None, return_embeddings: bool = False):
        if isinstance(sentences, str):
            sentences = [sentences]

        encoded = self.tokenizer(sentences,
                               padding=True,
                               truncation=True,
                               return_tensors='pt',
                               max_length=512)

        # Move tensors to the same device as model
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}

        outputs = self.transformer(**encoded)
        hidden_states = outputs.last_hidden_state
        pooled = self.pool_output(hidden_states, encoded['attention_mask'])
        embeddings = self.embedding_layer(pooled)

        result = {}
        if task is None or task == 'classification':
            result['classification_logits'] = self.classification_head(embeddings)
        if task is None or task == 'sentiment':
            result['sentiment_logits'] = self.sentiment_head(embeddings)
        if return_embeddings:
            result['embeddings'] = embeddings
        return result

    def get_loss(self, outputs, classification_labels=None, sentiment_labels=None,
                 task_weights={'classification': 1.0, 'sentiment': 1.0}):
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        if classification_labels is not None and 'classification_logits' in outputs:
            classification_loss = criterion(outputs['classification_logits'], classification_labels)
            total_loss += task_weights['classification'] * classification_loss

        if sentiment_labels is not None and 'sentiment_logits' in outputs:
            sentiment_loss = criterion(outputs['sentiment_logits'], sentiment_labels)
            total_loss += task_weights['sentiment'] * sentiment_loss

        return total_loss

    def get_readable_predictions(self, sentences, task, return_probabilities=False):
        if isinstance(sentences, str):
            sentences = [sentences]

        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            outputs = self.forward(sentences, task=task)

            if task == 'classification':
                logits = outputs['classification_logits']
                labels = self.class_labels
                key = 'class'
            elif task == 'sentiment':
                logits = outputs['sentiment_logits']
                labels = self.sentiment_labels
                key = 'sentiment'
            else:
                raise ValueError(f"Unknown task: {task}")

            probabilities = torch.softmax(logits, dim=-1)
            predictions = []

            for i, sentence in enumerate(sentences):
                probs = probabilities[i].cpu().numpy()
                pred_idx = np.argmax(probs)

                result = {
                    'sentence': sentence,
                    f'{key}': labels[pred_idx],
                    'confidence': float(probs[pred_idx])
                }

                if return_probabilities:
                    result['probabilities'] = {
                        label: float(prob)
                        for label, prob in zip(labels, probs)
                    }
                predictions.append(result)

            return predictions

class TextDataset(Dataset):
    def __init__(self, texts, class_labels, sentiment_labels):
        self.texts = texts
        self.class_labels = class_labels
        self.sentiment_labels = sentiment_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'class_label': self.class_labels[idx],
            'sentiment_label': self.sentiment_labels[idx]
        }

def prepare_data():
    print("Loading datasets...")
    news_dataset = load_dataset("ag_news", split="train[:5000]")
    
    # Use multiple sentiment datasets for better coverage
    imdb = load_dataset("imdb", split="train[:2500]")
    sst2 = load_dataset("sst2", split="train[:2500]")  
    
    texts = news_dataset['text']
    
    # Map AG News labels to our 3 classes
    class_mapping = {0: 0, 1: 0, 2: 1, 3: 2}  # World/Sports->News, Business->Technical, Sci/Tech->Casual
    class_labels = [class_mapping[label] for label in news_dataset['label']]
    
    # Create more nuanced 3-class sentiment
    sentiment_labels = []
    positive_words = {'excellent', 'fantastic', 'great', 'amazing', 'wonderful', 'good', 'love', 
                     'happy', 'positive', 'pleasant', 'perfect', 'awesome', 'brilliant'}
    negative_words = {'terrible', 'awful', 'bad', 'poor', 'horrible', 'hate', 'disappointment', 
                     'negative', 'worst', 'disappointed', 'useless', 'waste'}
    
    # Combine IMDB and SST2 reviews for sentiment
    combined_reviews = []
    combined_reviews.extend(imdb['text'])  # IMDB uses 'text'
    combined_reviews.extend(sst2['sentence'])  # SST2 uses 'sentence'
    
    # Map SST2 labels (0: negative, 1: positive) to our format
    sst2_sentiments = []
    for i, label in enumerate(sst2['label']):
        if label == 1:
            sst2_sentiments.append(2)  # Positive
        elif label == 0:
            sst2_sentiments.append(0)  # Negative
            
    # Process reviews for sentiment
    for i, review in enumerate(combined_reviews[:len(texts)]):
        if i < len(imdb['text']):  # IMDB review
            review_lower = review.lower()
            pos_count = sum(word in review_lower for word in positive_words)
            neg_count = sum(word in review_lower for word in negative_words)
            
            if pos_count > neg_count:
                sentiment_labels.append(2)  # Positive
            elif neg_count > pos_count:
                sentiment_labels.append(0)  # Negative
            else:
                sentiment_labels.append(1)  # Neutral
        else:  # SST2 review
            sentiment_labels.append(sst2_sentiments[i - len(imdb['text'])])
    
    # Ensure balanced sentiment classes
    sentiment_counts = [sentiment_labels.count(i) for i in range(3)]
    min_count = min(sentiment_counts)
    
    balanced_indices = []
    class_counts = [0, 0, 0]
    
    for idx, sentiment in enumerate(sentiment_labels):
        if class_counts[sentiment] < min_count:
            balanced_indices.append(idx)
            class_counts[sentiment] += 1
    
    # Create balanced dataset
    texts = [texts[i] for i in balanced_indices]
    class_labels = [class_labels[i] for i in balanced_indices]
    sentiment_labels = [sentiment_labels[i] for i in balanced_indices]
    
    # Train/val split
    train_size = int(0.8 * len(texts))
    
    train_data = TextDataset(
        texts[:train_size],
        class_labels[:train_size],
        sentiment_labels[:train_size]
    )
    
    val_data = TextDataset(
        texts[train_size:],
        class_labels[train_size:],
        sentiment_labels[train_size:]
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Print class distribution
    print("\nSentiment distribution in training data:")
    train_sentiments = train_data.sentiment_labels
    for i, label in enumerate(['Negative', 'Neutral', 'Positive']):
        count = train_sentiments.count(i)
        percentage = count / len(train_sentiments) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")
    
    return train_data, val_data

def train_model(model, train_data, val_data, num_epochs=20, batch_size=128, learning_rate=2e-5):
    # Move model to GPU first
    model = model.to(device)
    print(f"Model moved to: {next(model.parameters()).device}")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.transformer.parameters(), 'lr': learning_rate},
        {'params': model.embedding_layer.parameters(), 'lr': learning_rate * 2},
        {'params': model.classification_head.parameters(), 'lr': learning_rate * 3},
        {'params': model.sentiment_head.parameters(), 'lr': learning_rate * 3}
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)
    
    best_val_loss = float('inf')
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch data to GPU
            batch_texts = batch['text']
            class_labels = batch['class_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_texts)  # Model will handle moving tensors to GPU
            
            loss = model.get_loss(
                outputs, 
                class_labels, 
                sentiment_labels,
                task_weights={'classification': 0.3, 'sentiment': 0.7}
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar with GPU memory info
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'GPU Memory (MB)': f'{gpu_memory:.1f}'
                })
            else:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / train_steps
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_sentiment = 0
        total_sentiment = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch data to GPU
                batch_texts = batch['text']
                class_labels = batch['class_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)
                
                outputs = model(batch_texts)
                
                loss = model.get_loss(outputs, class_labels, sentiment_labels)
                
                # Calculate sentiment accuracy
                sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1)
                correct_sentiment += (sentiment_preds == sentiment_labels).sum().item()
                total_sentiment += sentiment_labels.size(0)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        sentiment_accuracy = correct_sentiment / total_sentiment
        
        print(f"Average validation loss: {avg_val_loss:.4f}")
        print(f"Sentiment accuracy: {sentiment_accuracy:.2%}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Saving best model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pt')
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 3:
                print("Early stopping triggered")
                break
    
    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def test_samples(model):
    test_sentences = [
        "The latest research paper discusses quantum computing advances.",
        "Had an amazing day at the beach today!",
        "The stock market showed mixed results this quarter.",
        "This product is absolutely terrible, would not recommend.",
        "The weather is quite pleasant today."
    ]

    print("\n=== Classification Results ===")
    classification_results = model.get_readable_predictions(
        test_sentences,
        task='classification',
        return_probabilities=True
    )

    for result in classification_results:
        print(f"\nSentence: {result['sentence']}")
        print(f"Predicted Class: {result['class']} (Confidence: {result['confidence']:.2%})")
        print("Class Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  - {label}: {prob:.2%}")

    print("\n=== Sentiment Analysis Results ===")
    sentiment_results = model.get_readable_predictions(
        test_sentences,
        task='sentiment',
        return_probabilities=True
    )

    for result in sentiment_results:
        print(f"\nSentence: {result['sentence']}")
        print(f"Predicted Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2%})")
        print("Sentiment Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  - {label}: {prob:.2%}")

def load_trained_model_and_predict(model_path, sentences):
    # Initialize model (make sure architecture matches training)
    model = MultiTaskSentenceTransformer(add_mlp=True)
    
    # Load trained weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get predictions for both tasks
    print("\n=== Classification Results ===")
    classification_results = model.get_readable_predictions(
        sentences,
        task='classification',
        return_probabilities=True
    )
    
    print("\n=== Sentiment Analysis Results ===")
    sentiment_results = model.get_readable_predictions(
        sentences,
        task='sentiment',
        return_probabilities=True
    )
    
    # Format and return results
    results = []
    for i, sentence in enumerate(sentences):
        result = {
            'sentence': sentence,
            'classification': {
                'predicted_class': classification_results[i]['class'],
                'confidence': classification_results[i]['confidence'],
                'probabilities': classification_results[i]['probabilities']
            },
            'sentiment': {
                'predicted_sentiment': sentiment_results[i]['sentiment'],
                'confidence': sentiment_results[i]['confidence'],
                'probabilities': sentiment_results[i]['probabilities']
            }
        }
        results.append(result)
        
        # Print detailed results
        print(f"\nResults for: {sentence}")
        print(f"Classification: {result['classification']['predicted_class']} "
              f"(Confidence: {result['classification']['confidence']:.2%})")
        print("Class Probabilities:", result['classification']['probabilities'])
        print(f"Sentiment: {result['sentiment']['predicted_sentiment']} "
              f"(Confidence: {result['sentiment']['confidence']:.2%})")
        print("Sentiment Probabilities:", result['sentiment']['probabilities'])
        
    return results