"""
Example script demonstrating the usage of the Diffusion-PINN solver.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusion_pinns.solver import DiffusionPINN
from diffusion_pinns.config import Config
from diffusion_pinns.utils import plot_solution, plot_training_history, set_seed

def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Initial condition: u(x,0) = sin(πx)"""
    return torch.sin(np.pi * x)

def boundary_condition(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Boundary conditions: u(0,t) = u(1,t) = 0"""
    return torch.zeros_like(x)

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create configuration
    config = Config(
        x_domain=(0.0, 1.0),
        t_domain=(0.0, 1.0),
        diffusion_coefficient=0.1,
        hidden_layers=[20, 20, 20],
        n_train=10000,
        n_test=1000,
        epochs=1000,
        learning_rate=1e-3
    )
    
    # Initialize solver
    solver = DiffusionPINN(config)
    
    # Train the model
    history = solver.train(
        initial_condition=initial_condition,
        boundary_condition=boundary_condition
    )
    
    # Plot training history
    plot_training_history(history, save_path="training_history.png")
    
    # Generate test points
    x_test = torch.linspace(config.x_domain[0], config.x_domain[1], 100, device=config.device).reshape(-1, 1)
    t_test = torch.linspace(config.t_domain[0], config.t_domain[1], 100, device=config.device).reshape(-1, 1)
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze())
    x_test = X.reshape(-1, 1)
    t_test = T.reshape(-1, 1)
    
    # Make predictions
    u_pred = solver.predict(x_test, t_test)
    
    # Plot solution
    plot_solution(x_test, t_test, u_pred, title="PINN Solution", save_path="solution.png")
    
    # Save the model
    solver.save("diffusion_pinn_model.pt")
    
    print("Training completed and results saved!")

if __name__ == "__main__":
    main() 