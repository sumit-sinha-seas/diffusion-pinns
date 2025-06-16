"""
Loss functions for the Diffusion-PINN solver.
"""

import torch
import torch.nn as nn
from typing import Tuple, Callable

class DiffusionLoss(nn.Module):
    """Loss functions for the diffusion equation PINN."""
    
    def __init__(self, config):
        """
        Initialize the loss functions.
        
        Args:
            config: Configuration object containing loss weights and parameters
        """
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
    
    def physics_loss(self, u_t: torch.Tensor, u_xx: torch.Tensor) -> torch.Tensor:
        """
        Compute the physics loss (PDE residual).
        
        Args:
            u_t: Time derivative of u
            u_xx: Second spatial derivative of u
            
        Returns:
            Physics loss
        """
        # Diffusion equation: u_t = D * u_xx
        residual = u_t - self.config.diffusion_coefficient * u_xx
        return self.mse(residual, torch.zeros_like(residual))
    
    def initial_condition_loss(self, 
                             x: torch.Tensor, 
                             t: torch.Tensor, 
                             u: torch.Tensor,
                             initial_condition: Callable) -> torch.Tensor:
        """
        Compute the initial condition loss.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            u: Predicted solution
            initial_condition: Function that computes the initial condition u(x,0)
            
        Returns:
            Initial condition loss
        """
        # Get points at t=0
        mask = (t == self.config.t_domain[0])
        if not torch.any(mask):
            return torch.tensor(0.0, device=self.config.device)
        
        x_ic = x[mask]
        u_ic = u[mask]
        u_ic_true = initial_condition(x_ic)
        
        return self.mse(u_ic, u_ic_true)
    
    def boundary_condition_loss(self,
                              x: torch.Tensor,
                              t: torch.Tensor,
                              u: torch.Tensor,
                              boundary_condition: Callable) -> torch.Tensor:
        """
        Compute the boundary condition loss.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            u: Predicted solution
            boundary_condition: Function that computes the boundary condition
            
        Returns:
            Boundary condition loss
        """
        # Get points at boundaries
        mask = (x == self.config.x_domain[0]) | (x == self.config.x_domain[1])
        if not torch.any(mask):
            return torch.tensor(0.0, device=self.config.device)
        
        x_bc = x[mask]
        t_bc = t[mask]
        u_bc = u[mask]
        u_bc_true = boundary_condition(x_bc, t_bc)
        
        return self.mse(u_bc, u_bc_true)
    
    def total_loss(self,
                  x: torch.Tensor,
                  t: torch.Tensor,
                  u: torch.Tensor,
                  u_t: torch.Tensor,
                  u_xx: torch.Tensor,
                  initial_condition: Callable,
                  boundary_condition: Callable) -> Tuple[torch.Tensor, dict]:
        """
        Compute the total loss.
        
        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            u: Predicted solution
            u_t: Time derivative of u
            u_xx: Second spatial derivative of u
            initial_condition: Function that computes the initial condition
            boundary_condition: Function that computes the boundary condition
            
        Returns:
            Tuple of (total loss, dictionary of individual losses)
        """
        # Compute individual losses
        physics = self.physics_loss(u_t, u_xx)
        ic = self.initial_condition_loss(x, t, u, initial_condition)
        bc = self.boundary_condition_loss(x, t, u, boundary_condition)
        
        # Weighted sum
        total = (self.config.physics_weight * physics +
                self.config.ic_weight * ic +
                self.config.bc_weight * bc)
        
        # Return total loss and individual losses
        losses = {
            'physics': physics.item(),
            'initial_condition': ic.item(),
            'boundary_condition': bc.item(),
            'total': total.item()
        }
        
        return total, losses 