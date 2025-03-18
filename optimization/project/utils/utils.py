import matplotlib.pyplot as plt
import torch
torch

def frobenius_norm(U, V, X, lambda_reg= 0):
    UV = U @ V.T
    n = UV.shape[0] * UV.shape[1]
    loss = (1 / (2 * n)) * (torch.norm(UV- X, p="fro") ** 2)
    reg = (lambda_reg / 2) * ((torch.norm(U, p="fro") ** 2 + torch.norm(V, p="fro") ** 2))
    return loss + reg

def visualize(data: dict, title: str):
    """
    Args:
        - data: dictionnary with plot labels as keys and list as values
    """
    plt.figure(figsize=(7, 5))
    for label, values in data.items():
        plt.semilogy(values, label=label, lw=2)
    plt.title(f"Convergence in terms of {title}", fontsize=16)
    plt.xlabel("#Iterations", fontsize=14)
    plt.ylabel("Fct values", fontsize=14)
    plt.legend()