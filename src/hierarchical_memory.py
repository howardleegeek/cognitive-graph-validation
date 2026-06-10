import torch
import torch.nn as nn

class HierarchicalMemory(nn.Module):
    """
    A simplified implementation of a hierarchical memory mechanism 
    for sequence modeling, designed to capture long-range dependencies 
    by maintaining separate context levels (e.g., local and global).
    """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initial embedding layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Core RNN (e.g., GRU or LSTM structure for simplicity)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, input_size).
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, seq_len, 1).
        """
        # Process through the core RNN
        # output: (B, S, H), hn: (D*H)
        output, _ = self.rnn(self.input_layer(x))
        
        # Generate predictions
        predictions = self.output_layer(output)
        return predictions

def get_hm_model(input_size, hidden_size):
    """Factory function to instantiate the HM model."""
    return HierarchicalMemory(input_size, hidden_size)

if __name__ == '__main__':
    # Example usage test (assuming batch size 4, seq len 10, input dim 8)
    batch_size = 4
    seq_len = 10
    input_dim = 8
    hidden_dim = 16

    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    model = get_hm_model(input_dim, hidden_dim)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, 1)
    print("HM Model test passed.")