"""
Main solver class for the Diffusion-PINN implementation.
"""

import torch
import numpy as np
from typing import Tuple, Callable, Dict, Optional
from tqdm import tqdm

from .network import PINN
from .loss import DiffusionLoss
from .config import Config

class DiffusionPINN:
    """Solver for the diffusion equation using Physics-Informed Neural Networks."""
    
    def __init__(self, config: Config):
        """
        Initialize the solver.
        
        Args:
            config: Configuration object containing solver parameters
        """
        self.config = config
        self.device = config.device
        
        # Initialize network and loss
        self.network = PINN(
            hidden_layers=config.hidden_layers,
            activation=config.activation
        ).to(self.device)
        
        self.loss_fn = DiffusionLoss(config)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )
        
        # Training history
        self.history: Dict[str, list] = {
            'physics_loss': [],
            'initial_condition_loss': [],
            'boundary_condition_loss': [],
            'total_loss': []
        }
    
    def _generate_training_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training points in the domain.
        
        Returns:
            Tuple of (x, t) tensors containing the training points
        """
        # Generate random points in the domain
        x = torch.rand(self.config.n_train, 1, device=self.device) * \
            (self.config.x_domain[1] - self.config.x_domain[0]) + \
            self.config.x_domain[0]
        
        t = torch.rand(self.config.n_train, 1, device=self.device) * \
            (self.config.t_domain[1] - self.config.t_domain[0]) + \
            self.config.t_domain[0]
        
        return x, t
    
    def train(self,
             initial_condition: Callable,
             boundary_condition: Callable,
             epochs: Optional[int] = None) -> Dict[str, list]:
        """
        Train the PINN.
        
        Args:
            initial_condition: Function that computes the initial condition u(x,0)
            boundary_condition: Function that computes the boundary condition
            epochs: Number of training epochs (overrides config if provided)
            
        Returns:
            Dictionary containing training history
        """
        epochs = epochs or self.config.epochs
        self.network.train()
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Generate training points
            x, t = self._generate_training_points()
            
            # Forward pass
            u = self.network(x, t)
            u_t, u_x, u_xx = self.network.compute_derivatives(x, t)
            
            # Compute loss
            loss, losses = self.loss_fn.total_loss(
                x, t, u, u_t, u_xx,
                initial_condition,
                boundary_condition
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record history
            for key, value in losses.items():
                self.history[key].append(value)
        
        return self.history
    
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the trained network.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            
        Returns:
            Predicted solution u(x,t)
        """
        self.network.eval()
        with torch.no_grad():
            return self.network(x, t)
    
    def save(self, path: str):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'DiffusionPINN':
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded DiffusionPINN instance
        """
        checkpoint = torch.load(path)
        config = checkpoint['config']
        
        solver = cls(config)
        solver.network.load_state_dict(checkpoint['network_state_dict'])
        solver.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        solver.history = checkpoint['history']
        
        return solver 