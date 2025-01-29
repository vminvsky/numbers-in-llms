# train probe on residuals
# put in the two classes of residuals 
# split them into train and test 
# train probe on train
# report test results
# save probe 

# TODO figure out how to bootstrap results 

import torch as t
import os
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import json

class LinearProbe(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

def load_residuals(model_name):
    """Load string and int residuals."""
    str_residuals = t.load(f'residuals/{model_name}_str_999.pt')
    int_residuals = t.load(f'residuals/{model_name}_int_999.pt')
    return str_residuals, int_residuals

def prepare_data_by_layer(str_residuals, int_residuals, str_labels=None, int_labels=None):
    """Prepare data for training, separated by layer."""
    n_layers = str_residuals[0].shape[0]
    
    # Initialize lists to store data for each layer
    X_by_layer = []
    y_by_layer = []
    
    for layer in range(n_layers):
        # Extract layer-specific residuals
        str_layer = t.stack([res[layer] for res in str_residuals])
        int_layer = t.stack([res[layer] for res in int_residuals])
        
        # default to the 1, 0 case to just predict the class 
        if str_labels is None:
            str_labels = t.ones(len(str_residuals))
        else: 
            str_labels = t.tensor(str_labels)

        if int_labels is None:
            int_labels = t.zeros(len(int_residuals))
        else: 
            int_labels = t.tensor(int_labels)
        
        # Combine data for this layer
        X = t.cat([str_layer, int_layer], dim=0)
        y = t.cat([str_labels, int_labels])
        
        X_by_layer.append(X)
        y_by_layer.append(y)
    
    return X_by_layer, y_by_layer

def evaluate_probe(model, X, y, device='cuda'):
    """Evaluate probe on given data."""
    model.eval()
    X = t.tensor(X, dtype=t.float32).to(device)
    y = t.tensor(y, dtype=t.float32).to(device)
    
    with t.no_grad():
        outputs = model(X).squeeze()
        probs = t.sigmoid(outputs)
        preds = (probs > 0.5).float()
        accuracy = (preds == y).float().mean()
        
        # Calculate per-class accuracy
        str_mask = y == 1
        int_mask = y == 0
        str_acc = (preds[str_mask] == y[str_mask]).float().mean()
        int_acc = (preds[int_mask] == y[int_mask]).float().mean()
        
    return {
        'accuracy': accuracy.item(),
        'str_accuracy': str_acc.item(),
        'int_accuracy': int_acc.item(),
        'probabilities': probs.cpu().numpy()
    }

def train_probe(X, y, device='cuda'):
    """Train probe and return test results."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.cpu().numpy(), y.cpu().numpy(), test_size=0.99, random_state=42
    )
    
    # Convert back to tensors and move to device
    X_train = t.tensor(X_train, dtype=t.float32).to(device)
    X_test = t.tensor(X_test, dtype=t.float32).to(device)
    y_train = t.tensor(y_train, dtype=t.float32).to(device)
    y_test = t.tensor(y_test, dtype=t.float32).to(device)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = LinearProbe(input_size).to(device)
    
    # Training parameters
    criterion = nn.BCEWithLogitsLoss()
    optimizer = t.optim.Adam(model.parameters())
    n_epochs = 100
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate
    eval_results = evaluate_probe(model, X_test, y_test, device)
    print(f'Test Accuracy: {eval_results["accuracy"]:.4f}')
    print(f'String Accuracy: {eval_results["str_accuracy"]:.4f}')
    print(f'Integer Accuracy: {eval_results["int_accuracy"]:.4f}')
    
    return model, eval_results

def save_probe(model, model_name, layer):
    """Save trained probe for a specific layer."""
    probe_dir = f'probes/{model_name}'
    os.makedirs(probe_dir, exist_ok=True)
    t.save(model.state_dict(), f'{probe_dir}/layer_{layer}_probe.pt')

def load_probe(model_name, layer, input_size, device='cuda'):
    """Load a trained probe."""
    model = LinearProbe(input_size).to(device)
    model.load_state_dict(t.load(f'probes/{model_name}/layer_{layer}_probe.pt'))
    return model

def save_results(results, model_name):
    """Save accuracies and evaluation results for all layers."""
    with open(f'probes/{model_name}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

def evaluate_residual_file(model_name, residual_file, device='cuda'):
    """Evaluate all probes on a given residual file."""
    # Load residuals
    residuals = t.load(residual_file)
    n_layers = residuals[0].shape[0]
    
    # Prepare data by layer
    X_by_layer = []
    for layer in range(n_layers):
        layer_data = t.stack([res[layer] for res in residuals])
        X_by_layer.append(layer_data)
    
    # Evaluate each layer's probe
    results = []
    for layer in range(n_layers):
        print(f'\nEvaluating layer {layer}')
        input_size = X_by_layer[layer].shape[1]
        probe = load_probe(model_name, layer, input_size, device)
        
        # Get predictions and probabilities
        layer_results = evaluate_probe(probe, X_by_layer[layer], t.zeros(len(residuals)), device)
        results.append({
            'layer': layer,
            'mean_probability': float(np.mean(layer_results['probabilities'])),
            'std_probability': float(np.std(layer_results['probabilities'])),
            'predictions': [bool(p > 0.5) for p in layer_results['probabilities']],
            'probabilities': [float(p) for p in layer_results['probabilities']]
        })
    
    return results

def main():
    model_name = 'llama-3.1-8b-instruct'
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    
    # Load data
    str_residuals, int_residuals = load_residuals(model_name)
    X_by_layer, y_by_layer = prepare_data_by_layer(str_residuals, int_residuals)
    
    # Train probe for each layer
    training_results = []
    for layer in range(len(X_by_layer)):
        print(f'\nTraining probe for layer {layer}')
        probe, eval_results = train_probe(X_by_layer[layer], y_by_layer[layer], device)
        training_results.append({
            'layer': layer,
            'accuracy': eval_results['accuracy'],
            'str_accuracy': eval_results['str_accuracy'],
            'int_accuracy': eval_results['int_accuracy']
        })
        
        # Save probe for this layer
        save_probe(probe, model_name, layer)
    
    # Save training results
    save_results({'training_results': training_results}, model_name)
    
    # Evaluate on test file
    test_file = f'residuals/{model_name}_999.pt'
    if os.path.exists(test_file):
        print(f'\nEvaluating on {test_file}')
        eval_results = evaluate_residual_file(model_name, test_file, device)
        
        # Save evaluation results
        with open(f'probes/{model_name}/eval_results_999.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Print summary
        print('\nEvaluation Results Summary:')
        for res in eval_results:
            print(f"Layer {res['layer']}: Mean prob = {res['mean_probability']:.4f} (±{res['std_probability']:.4f})")

if __name__ == '__main__':
    main()

