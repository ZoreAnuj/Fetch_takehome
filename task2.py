from utils import *

# Example usage

if __name__ == "__main__":
    # Path to saved model
    MODEL_PATH = 'best_model.pt'
    
    model = MultiTaskSentenceTransformer(add_mlp=True)
    
    # Load trained weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_samples(model)