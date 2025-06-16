"""
Neural network architecture for the Diffusion-PINN solver.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

class PINN(nn.Module):
    """Physics-Informed Neural Network for solving the diffusion equation."""
    
    def __init__(self, hidden_layers: List[int], activation: str = "tanh"):
        """
        Initialize the PINN.
        
        Args:
            hidden_layers: List of integers specifying the number of neurons in each hidden layer
            activation: Activation function to use ('tanh', 'relu', or 'sigmoid')
        """
        super().__init__()
        
        # Input layer (2 inputs: x and t)
        layers = [nn.Linear(2, hidden_layers[0])]
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(self._get_activation(activation))
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        
        # Output layer (1 output: u(x,t))
        layers.append(self._get_activation(activation))
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get the activation function module."""
        if activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            Predicted solution u(x,t)
        """
        # Combine inputs
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)
    
    def compute_derivatives(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the required derivatives for the diffusion equation.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            Tuple of (u_t, u_x, u_xx) where:
            - u_t is the first derivative with respect to time
            - u_x is the first derivative with respect to space
            - u_xx is the second derivative with respect to space
        """
        # Enable gradient computation
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Forward pass
        u = self.forward(x, t)
        
        # Compute derivatives
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0]
        
        return u_t, u_x, u_xx 