"""
Utility functions for the Diffusion-PINN solver.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import seaborn as sns

def plot_solution(x: torch.Tensor,
                 t: torch.Tensor,
                 u: torch.Tensor,
                 title: str = "Solution",
                 save_path: Optional[str] = None):
    """
    Plot the solution u(x,t).
    
    Args:
        x: Spatial coordinates
        t: Temporal coordinates
        u: Solution values
        title: Plot title
        save_path: Path to save the plot (if None, plot is displayed)
    """
    # Convert to numpy arrays
    x_np = x.cpu().numpy()
    t_np = t.cpu().numpy()
    u_np = u.cpu().numpy()
    
    # Create meshgrid
    X, T = np.meshgrid(np.unique(x_np), np.unique(t_np))
    U = u_np.reshape(X.shape)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.contourf(X, T, U, levels=50, cmap='viridis')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(history: dict,
                         save_path: Optional[str] = None):
    """
    Plot the training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot each loss component
    for key, values in history.items():
        if key != 'total_loss':
            plt.semilogy(values, label=key)
    
    # Plot total loss
    plt.semilogy(history['total_loss'], label='total_loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compute_error(u_pred: torch.Tensor,
                 u_true: torch.Tensor) -> Tuple[float, float]:
    """
    Compute the relative L2 error and maximum error.
    
    Args:
        u_pred: Predicted solution
        u_true: True solution
        
    Returns:
        Tuple of (relative L2 error, maximum error)
    """
    # Convert to numpy
    u_pred_np = u_pred.cpu().numpy()
    u_true_np = u_true.cpu().numpy()
    
    # Compute errors
    l2_error = np.sqrt(np.mean((u_pred_np - u_true_np)**2)) / \
               np.sqrt(np.mean(u_true_np**2))
    max_error = np.max(np.abs(u_pred_np - u_true_np))
    
    return l2_error, max_error

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 