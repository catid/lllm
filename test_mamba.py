import unittest
import torch
from torch import nn, optim
import time

from mamba_ssm import Mamba
from model.lru import LRULayer

class TestMamba(unittest.TestCase):
    def setUp(self):
        # Set the device for the test
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize parameters
        self.batch, self.length, self.dim = 2, 64, 16
        self.d_state = 16
        self.d_conv = 4
        self.expand = 2

    def test_mamba_output_shape(self):
        # Create a random input tensor
        x = torch.randn(self.batch, self.length, self.dim).to(self.device)
        # Instantiate the Mamba model with specified parameters
        model = Mamba(
            d_model=self.dim,  # Model dimension
            d_state=self.d_state,  # SSM state expansion factor
            d_conv=self.d_conv,  # Local convolution width
            expand=self.expand,  # Block expansion factor
        ).to(self.device)
        # Pass the input tensor through the model
        y = model(x)
        # Check if the output shape matches the input shape
        self.assertEqual(y.shape, x.shape, "Output shape does not match input shape.")

class TestMambaLearning(unittest.TestCase):
    def setUp(self):
        # Set device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Mamba model parameters
        self.dim = 16
        self.model = Mamba(
            d_model=self.dim,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(self.device)

        #self.model = torch.compile(self.model)

        # Set hyperparameters
        self.learning_rate = 0.01
        self.batch_size = 64
        self.seq_len = 64
        self.num_epochs = 100
        
        # Create a simple optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()

    def generate_smooth_data(self, batch_size, seq_len, dim):
        """
        Generates smooth sequence data where the target is a smoothed version of the input.
        """
        x = torch.randn(batch_size, seq_len, dim)
        y = torch.roll(x, shifts=1, dims=1)  # Simple target: shift input by one step
        return x, y

    def test_training_loss_reduction(self):
        initial_loss = None
        final_loss = None

        # Generate synthetic smooth data
        x, y = self.generate_smooth_data(self.batch_size, self.seq_len, self.dim)
        x, y = x.to(self.device), y.to(self.device)

        for epoch in range(self.num_epochs):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(x)

            # Compute loss
            loss = self.loss_fn(outputs, y)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            if epoch == self.num_epochs // 2:
                initial_loss = loss.item()
                print(f"Epoch {epoch} loss: {loss.item()}")
            elif epoch == self.num_epochs - 1:
                final_loss = loss.item()
                print(f"Epoch {epoch} loss: {loss.item()}")

        # Verify that the loss has decreased
        self.assertIsNotNone(initial_loss, "Initial loss is not computed.")
        self.assertIsNotNone(final_loss, "Final loss is not computed.")
        self.assertLess(final_loss, initial_loss, "Loss did not decrease after training.")

class TestModelComparison(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mamba setup
        self.dim = 16
        self.mamba_model = Mamba(
            d_model=self.dim,
            d_state=8,
            d_conv=4,
            expand=2,
        ).to(self.device)
        
        # RNN setup
        self.rnn_model = nn.RNN(
            input_size=self.dim,
            hidden_size=self.dim,  # Keeping the same dimension for a fair comparison
            num_layers=1,         # Comparable to a single Mamba block
            nonlinearity='tanh',
            batch_first=True,
        ).to(self.device)
        self.rnn_model = torch.compile(self.rnn_model)

        self.lru_model = LRULayer(self.dim)
        self.lru_model.to(self.device)
        self.lru_model = torch.compile(self.lru_model)

        # Shared training setup
        self.learning_rate = 0.01
        self.batch_size = 64
        self.seq_len = 65536//4
        self.num_epochs = 100

        self.loss_fn = nn.MSELoss()

    def generate_smooth_data(self, batch_size, seq_len, dim):
        x = torch.randn(batch_size, seq_len, dim)
        y = torch.roll(x, shifts=1, dims=1)
        return x, y

    def train_model(self, model, x, y):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):

            if epoch >= 2:
                start_time = time.time()

            optimizer.zero_grad()
            outputs = model(x)

            # Check if the model's output is a tuple and select the first element if so
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Assuming the first element is the desired output for the loss calculation

            loss = self.loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
        end_time = time.time()

        torch.cuda.empty_cache()

        return end_time - start_time, loss.item()

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def test_model_comparison(self):
        x, y = self.generate_smooth_data(self.batch_size, self.seq_len, self.dim)
        x, y = x.to(self.device), y.to(self.device)

        # Print number of parameters
        mamba_params = self.count_parameters(self.mamba_model)
        rnn_params = self.count_parameters(self.rnn_model)
        lru_params = self.count_parameters(self.lru_model)
        print(f"Mamba - Number of trainable parameters: {mamba_params}")
        print(f"RNN - Number of trainable parameters: {rnn_params}")
        print(f"LRU - Number of trainable parameters: {lru_params}")

        # Train Mamba and measure performance
        mamba_time, mamba_loss = self.train_model(self.mamba_model, x, y)
        print(f"Mamba - Training time: {mamba_time:.3f} seconds, Final loss: {mamba_loss:.4f}")

        # Train RNN and measure performance
        rnn_time, rnn_loss = self.train_model(self.rnn_model, x, y)
        print(f"RNN - Training time: {rnn_time:.3f} seconds, Final loss: {rnn_loss:.4f}")

        # Train LRU and measure performance
        lru_time, lru_loss = self.train_model(self.lru_model, x, y)
        print(f"LRU - Training time: {lru_time:.3f} seconds, Final loss: {lru_loss:.4f}")

if __name__ == '__main__':
    unittest.main()
