import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import Dict, List, Tuple
import logging
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

class QuantumLayer(nn.Module):
    """Quantum layer implementing a variational quantum circuit."""
    
    def __init__(self, n_qubits, n_layers=1):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize weights
        self.weights = nn.Parameter(torch.randn(3 * n_layers * n_qubits, dtype=torch.float32))
        
        # Move weights to device
        self.to(self.device)
        
    def circuit(self, inputs):
        """Implement the quantum circuit."""
        # Encode input data
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Apply variational layers
        for layer in range(self.n_layers):
            # Rotations
            for i in range(self.n_qubits):
                qml.RX(self.weights[3 * (layer * self.n_qubits + i)], wires=i)
                qml.RY(self.weights[3 * (layer * self.n_qubits + i) + 1], wires=i)
                qml.RZ(self.weights[3 * (layer * self.n_qubits + i) + 2], wires=i)
            
            # Entangling gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """Forward pass through the quantum layer."""
        # Ensure input is on correct device and type
        x = x.to(device=self.device, dtype=torch.float32)
        
        # Process each sample in the batch
        results = []
        dev = qml.device("default.qubit", wires=self.n_qubits)
        qnode = qml.QNode(self.circuit, dev, interface="torch")
            
        for sample in x:
            # Get quantum circuit output - ensure it's a tensor
            output = qnode(sample)
            # Check if output is a list and stack it if necessary
            if isinstance(output, list):
                # Assuming the list contains tensors or numbers convertible to tensor
                output = torch.stack([torch.as_tensor(o, dtype=torch.float32, device=self.device) for o in output])
            elif not isinstance(output, torch.Tensor):
                # Handle cases where output might be a single number
                output = torch.as_tensor(output, dtype=torch.float32, device=self.device)
                
            results.append(output)
        
        # Stack results from all samples in the batch
        output = torch.stack(results)
        # Ensure the final output has the correct batch dimension
        if output.dim() == 1 and x.shape[0] > 1: # If batch size > 1 and output is 1D
             output = output.unsqueeze(0) # Add batch dimension back (this case might indicate issues)
        elif output.dim() > 2: # If output has more than 2 dims (e.g., from stacking per-qubit results)
             output = output.view(x.shape[0], -1) # Flatten features per sample

        return output
    
    def get_circuit_params(self):
        """Get the current quantum circuit parameters."""
        return self.weights

class QuantumNeuralNetwork(nn.Module):
    """Quantum Neural Network combining classical and quantum layers."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, n_qubits=4, dropout_rate=0.1):
        super(QuantumNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.dropout_rate = dropout_rate
        
        # Force CPU device for quantum operations to avoid CUDA kernel image errors
        self.device = torch.device('cpu')
        
        # Input layer to quantum layer
        self.input_layer = nn.Linear(input_dim, n_qubits)
        
        # Quantum layer
        self.quantum_layer = QuantumLayer(n_qubits)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        prev_dim = n_qubits
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Ensure model is on CPU
        self.to(self.device)
        
    def forward(self, x):
        # Ensure input is on CPU for quantum operations
        x = x.to(device=self.device, dtype=torch.float32)
        
        # Input layer
        x = self.input_layer(x.to(self.device))
        
        # Quantum layer
        x = self.quantum_layer(x.to(self.device))
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x.to(self.device))
            
        # Output layer
        x = self.output_layer(x.to(self.device))
        
        return x
    
    def predict(self, x):
        """Make predictions with the model."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.forward(x).cpu().numpy()
    
    def get_config(self):
        """Get model configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'n_qubits': self.n_qubits,
            'dropout_rate': self.dropout_rate
        }

    async def fit(self, train_loader, val_loader=None, epochs=10):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(dtype=torch.float32, device=self.device)
                batch_y = batch_y.to(dtype=torch.float32, device=self.device)
                
                loss = self.train_step(batch_x, batch_y)
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = (self(batch_x) > 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            
            metrics['train_loss'].append(avg_train_loss)
            metrics['train_accuracy'].append(train_accuracy)
            
            # Validation step
            if val_loader is not None:
                val_loss, val_accuracy = self.evaluate(val_loader)
                metrics['val_loss'].append(val_loss)
                metrics['val_accuracy'].append(val_accuracy)
                
                logging.info(f'Epoch {epoch+1}/{epochs} - '
                           f'Train Loss: {avg_train_loss:.4f} - '
                           f'Train Acc: {train_accuracy:.4f} - '
                           f'Val Loss: {val_loss:.4f} - '
                           f'Val Acc: {val_accuracy:.4f}')
            else:
                logging.info(f'Epoch {epoch+1}/{epochs} - '
                           f'Train Loss: {avg_train_loss:.4f} - '
                           f'Train Acc: {train_accuracy:.4f}')
        
        return metrics

    def train_step(self, batch_x, batch_y):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        optimizer.zero_grad()
        outputs = self(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        return loss

    def evaluate(self, val_loader):
        self.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(dtype=torch.float32, device=self.device)
                batch_y = batch_y.to(dtype=torch.float32, device=self.device)
                
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        self.train()
        return avg_val_loss, val_accuracy

    def save(self, path: str):
        """Save the model"""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        """Load the model"""
        self.load_state_dict(torch.load(path)) 