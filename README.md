# ML Apprenticeship Project

## Overview
This project implements a MultiTask Sentence Transformer model for text processing, featuring both single-task and multi-task learning capabilities. The implementation includes sentence embedding generation and multi-task classification with sentiment analysis. All answers and detailed explanations for the assessment questions can be found in the `ML Apprenticeship Take-Home.ipynb` Jupyter notebook.

## Project Structure
```
.
├── ML Apprenticeship Take-Home.ipynb  # Jupyter notebook containing all assessment answers and implementation details
├── best_model.pt                      # Saved model weights
├── task1.py                           # Sentence Transformer implementation
├── task2.py                           # Multi-task model implementation
├── utils.py                           # Utility functions and configurations
└── requirements.txt                   # Project dependencies
```

## Assessment Answers
All detailed answers, explanations, and implementation walkthroughs for the assessment questions are provided in the Jupyter notebook (`ML Apprenticeship Take-Home.ipynb`). The notebook includes:
- Comprehensive explanations of design choices
- Step-by-step implementation details
- Performance analysis and results
- Code examples with annotations
- Architectural decisions and reasoning

## Setup and Installation

0. Please download the weights (https://drive.google.com/file/d/15o3dOq6ON0ojkO-7EQDim6561eoyBCC9/view?usp=sharing) and place it in your current active directory.

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure device settings:
- Open `utils.py` and set your preferred device ('cuda' or 'cpu')
- Default configuration uses CPU if CUDA is not available

3. Ensure you have the pre-trained model file (`best_model.pt`) in your project root directory.

## Running the Tasks

### Task 1: Sentence Transformer
```bash
python task1.py
```
This runs the base Sentence Transformer implementation with:
- BERT base model backbone
- Mean pooling strategy
- Custom embedding dimension
- Optional MLP layer

### Task 2: Multi-Task Learning
```bash
python task2.py
```
This executes the extended multi-task model featuring:
- Classification task
- Sentiment analysis task
- Shared transformer backbone
- Task-specific heads

## Interactive Development
For a detailed walkthrough of the implementation and assessment answers:
```bash
jupyter notebook "ML Apprenticeship Take-Home.ipynb"
```

## Model Architecture

### Base Sentence Transformer (Task 1)
- Backbone: BERT base uncased
- Pooling options: mean, cls, attention
- Optional MLP layer for enhanced representations
- LayerNorm for output stabilization

### Multi-Task Model (Task 2)
- Shared BERT backbone
- Classification head for text categorization 
- Sentiment analysis head
- Task-specific learning rates
- Flexible forward pass for single/multi-task inference

## Usage Examples

### Basic Sentence Transformer
```python
from utils import *

# Initialize base model
model = SentenceTransformer(
    model_name='bert-base-uncased',
    pooling_method='mean',
    output_dim=768,
    add_mlp=True
)

# Generate embeddings
sentences = ["Example text for embedding generation"]
embeddings = model.encode(sentences)
```

### Multi-Task Model
```python
from utils import *

# Initialize multi-task model
model = MultiTaskSentenceTransformer(add_mlp=True)

# Load pre-trained weights
checkpoint = torch.load('best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Run inference
test_samples(model)
```

## Model Features

- Device-agnostic execution
- Flexible pooling strategies
- Task-specific learning rates
- Comprehensive test utilities
- Built-in performance metrics

## Dependencies

- PyTorch
- Transformers
- Datasets
- NumPy
- tqdm
- Additional requirements in `requirements.txt`

## Implementation Notes

- Layer-wise learning rates for optimal training
- Early stopping with validation loss monitoring
- Balanced multi-task loss weighting
- Automatic device selection and memory management
- Comprehensive error handling

## Troubleshooting

Common issues and solutions:
1. CUDA out of memory
   - Reduce batch size
   - Switch to CPU device in utils.py

2. Model loading errors
   - Verify model checkpoint path
   - Check device compatibility
   - Ensure matching architecture configuration

## Future Improvements

- Add support for additional tasks
- Implement cross-validation
- Enhance attention mechanisms
- Add model export capabilities
- Implement distributed training
