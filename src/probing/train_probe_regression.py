import torch as t
import os
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import json
from transformers import AutoTokenizer
from src.utils import model_paths

from .create_data import SimilarityData

class LinearProbe(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

def load_residuals_and_groundtruth(model_name, format_type, layers, full_dataset_size=10000):
    """Load residuals and corresponding groundtruth values."""
    layer_str = '_'.join(list(map(str, layers.values())))
    residuals = t.load(f'residuals/llama-3.1-8b-instruct_similarity_string_{full_dataset_size}_1000_layer_{layer_str}.pt')
    groundtruth = t.load(f'groundtruths/{model_name}_similarity_string_{full_dataset_size}_1000_{format_type}_groundtruths.pt')
    return residuals, groundtruth

def prepare_data_by_layer(residuals, groundtruth):
    """Prepare data for training, separated by layer."""
    n_layers = residuals[0].shape[0]
    
    # Initialize lists to store data for each layer
    X_by_layer = []
    y = t.tensor(groundtruth)
    
    for layer in range(n_layers):
        # Extract layer-specific residuals
        layer_data = t.stack([res[layer] for res in residuals])
        X_by_layer.append(layer_data)
    
    return X_by_layer, y

def evaluate_probe(model, X, y, device='cuda'):
    """Evaluate regression probe on given data."""
    model.eval()
    X = t.tensor(X, dtype=t.float32).to(device)
    y = t.tensor(y, dtype=t.float32).to(device)
    
    with t.no_grad():
        predictions = model(X).squeeze()
        mse = nn.MSELoss()(predictions, y)
        mae = nn.L1Loss()(predictions, y)
        
        # Calculate correlation coefficient with checks
        predictions_np = predictions.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Check for constant predictions or targets
        if np.std(predictions_np) == 0 or np.std(y_np) == 0:
            correlation = 0.0
            print("Warning: Standard deviation of predictions or targets is 0")
        else:
            correlation = np.corrcoef(predictions_np, y_np)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
                print("Warning: Correlation is NaN")
        
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'correlation': correlation,
        'predictions': predictions.cpu().numpy(),
        'std_pred': float(np.std(predictions_np)),
        'std_target': float(np.std(y_np))
    }

def train_probe(X, y, n_train_points, device='cuda'):
    """Train regression probe and return test results."""
    total_points = X.shape[0]
    
    # Calculate train size as a fraction
    train_size = n_train_points / total_points
    
    # Split data with shuffling - this will give us n_train_points for training
    X_train, X_test, y_train, y_test = train_test_split(
        X.cpu().numpy(), y.cpu().numpy(), 
        train_size=train_size, 
        random_state=42, 
        shuffle=True
    )
    
    print(f"Training on {len(X_train)} points, testing on {len(X_test)} points")
    
    # Convert back to tensors and move to device
    X_train = t.tensor(X_train, dtype=t.float32).to(device)
    X_test = t.tensor(X_test, dtype=t.float32).to(device)
    y_train = t.tensor(y_train, dtype=t.float32).to(device)
    y_test = t.tensor(y_test, dtype=t.float32).to(device)

    # Initialize model
    input_size = X_train.shape[1]
    model = LinearProbe(input_size).to(device)
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters())
    n_epochs = 100
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training loop with early stopping
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate
    eval_results = evaluate_probe(model, X_test, y_test, device)
    print(f'Test MSE: {eval_results["mse"]:.4f}')
    print(f'Test MAE: {eval_results["mae"]:.4f}')
    print(f'Test Correlation: {eval_results["correlation"]:.4f}')
    print(f'Pred StdDev: {eval_results["std_pred"]:.4f}')
    print(f'Target StdDev: {eval_results["std_target"]:.4f}')
    
    return model, eval_results

def save_probe(model, model_name, format_type, layer, n_points):
    """Save trained probe for a specific layer."""
    probe_dir = f'probes/{n_points}/{model_name}_regression_{format_type}'
    os.makedirs(probe_dir, exist_ok=True)
    t.save(model.state_dict(), f'{probe_dir}/layer_{layer}_probe.pt')

def save_results(results, model_name, format_type, n_points):
    """Save regression results for all layers."""
    probe_dir = f'probes/{n_points}/{model_name}_regression_{format_type}'
    os.makedirs(probe_dir, exist_ok=True)
    with open(f'{probe_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

def main(n_points_list=[100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9500], max_nums=1000, seed=42, full_dataset_size=10000):
    model_name = 'llama-3.1-8b-instruct'
    layer_probe_mapping = {i: i for i in list(range(33))}
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    t.manual_seed(seed)
    
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_paths['llama-3.1-8b-instruct'])

    for n_points in n_points_list:
        print(f'\nTraining with {n_points} points')
        for format_type in ['int', 'string']:
            print(f'\nTraining probes for {format_type} similarity')

            # Load full dataset
            residuals, groundtruth = load_residuals_and_groundtruth(
                model_name, 
                format_type, 
                layer_probe_mapping, 
                full_dataset_size
            )
            X_by_layer, y = prepare_data_by_layer(residuals, groundtruth)

            # Train probe for each layer
            training_results = []
            for layer in range(len(X_by_layer)):
                print(f'\nTraining probe for layer {layer}')
                probe, eval_results = train_probe(X_by_layer[layer], y, n_points, device)
                training_results.append({
                    'layer': layer,
                    'mse': eval_results['mse'],
                    'mae': eval_results['mae'],
                    'correlation': eval_results['correlation'],
                    'n_train': n_points,
                    'n_test': len(y) - n_points
                })
                
                # Save probe for this layer
                save_probe(probe, model_name, format_type, layer_probe_mapping[layer], n_points)
            
            # Save training results
            save_results({'training_results': training_results}, model_name, format_type, n_points)

if __name__ == '__main__':
    main()
