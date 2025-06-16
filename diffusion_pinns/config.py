"""
Configuration management for the Diffusion-PINN solver.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

@dataclass
class Config:
    """Configuration class for the Diffusion-PINN solver."""
    
    # Domain parameters
    x_domain: Tuple[float, float] = (0.0, 1.0)  # Spatial domain
    t_domain: Tuple[float, float] = (0.0, 1.0)  # Temporal domain
    diffusion_coefficient: float = 1.0  # Diffusion coefficient
    
    # Neural network parameters
    hidden_layers: List[int] = (20, 20, 20)  # Number of neurons in each hidden layer
    activation: str = "tanh"  # Activation function
    
    # Training parameters
    n_train: int = 10000  # Number of training points
    n_test: int = 1000    # Number of test points
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 1000
    
    # PINN specific parameters
    physics_weight: float = 1.0  # Weight for physics loss
    ic_weight: float = 1.0       # Weight for initial condition loss
    bc_weight: float = 1.0       # Weight for boundary condition loss
    
    # Device configuration
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.hidden_layers, (list, tuple)):
            self.hidden_layers = [self.hidden_layers]
        
        if self.x_domain[0] >= self.x_domain[1]:
            raise ValueError("Invalid spatial domain: x_domain[0] must be less than x_domain[1]")
        
        if self.t_domain[0] >= self.t_domain[1]:
            raise ValueError("Invalid temporal domain: t_domain[0] must be less than t_domain[1]")
        
        if self.diffusion_coefficient <= 0:
            raise ValueError("Diffusion coefficient must be positive") 