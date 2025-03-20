import numpy as np
from scipy.linalg import norm  # Norm euclidienne (correspondant au produit scalaire)
from scipy.linalg import svdvals  # Décomposition en valeurs singulières
import torch
from torch import nn

from problems.problem import Problem


class MatFactPb(Problem):
    """
    Matrix factorization problem class.
    For the sake of simplicity, we use pytorch to differentiate the frobenius norm.

    Attributs:
        u: vercteur de données (attributs)
        v: Vecteur de données (labels)
        X
        n,d: Dimensions de X
        loss: Fonction de perte choisie
            'l2': Perte aux moindres carrés
            'logit': Perte logistique
        lambda_reg: Paramètre de régularisation
    """

    # Initialisation
    def __init__(self, U, V, X, lambda_reg=0):
        self.U = torch.tensor(U)
        self.U.requires_grad = True
        self.U_grad_norm_list = []

        self.V = torch.tensor(V)
        self.V.requires_grad = True
        self.V_grad_norm_list = []

        self.avg_grad_norm_list = []

        self.X = torch.tensor(X)
        self.n, self.d = self.X.shape[0], self.X[0]
        self.lambda_reg = lambda_reg
        self.optimizer = torch.optim.SGD(params=[self.U, self.V], lr=1)

    # Loss using Frobenius norm
    def loss(self):
        UV = self.U @ self.V.T
        n = UV.shape[0] * UV.shape[1]
        loss = (1 / (2 * n)) * (torch.norm(UV - self.X, p="fro") ** 2)
        reg = (self.lambda_reg / 2) * (
            (torch.norm(self.U, p="fro") ** 2 + torch.norm(self.V, p="fro") ** 2)
        )
        return loss + reg

    def loss_i(self, indices):
        # Draw batch for each component
        U_batch = self.U[indices, :]
        V_batch = self.V[indices, :]
        UV_batch = U_batch @ V_batch.T
        X_batch = self.X[indices, :]
        # UV = self.U @ self.V.T
        # UV_batch = UV[indices, :]
        # n = UV_batch.shape[0] * UV_batch.shape[1]

        # Compute objective function
        loss = (1 / (2 * self.n)) * (torch.norm(UV_batch - X_batch, p="fro") ** 2)
        reg = (self.lambda_reg / 2) * (
            (torch.norm(self.U, p="fro") ** 2 + torch.norm(self.V, p="fro") ** 2)
        )
        return (loss + reg) / len(indices)

    # Calcul de gradient
    def step(self, s, indices):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = s

        # Compute one gradient algorithm step
        loss = self.loss_i(indices)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store gradient history
        U_norm_grad = torch.norm(self.U.grad, p="fro").item()
        V_norm_grad = torch.norm(self.V.grad, p="fro").item()
        self.U_grad_norm_list.append(U_norm_grad)
        self.V_grad_norm_list.append(V_norm_grad)
        self.avg_grad_norm_list.append(np.mean([U_norm_grad, V_norm_grad]))
