import numpy as np
from scipy.linalg import norm  # Norm euclidienne (correspondant au produit scalaire)
from scipy.linalg import svdvals  # Décomposition en valeurs singulières
import torch
from torch import nn

from problems.problem import Problem


class OverParamRegPb(Problem):
    """
    Over parameterized regression problem class.
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
    def __init__(self, X, y, W, lambda_reg=0):
        self.W = torch.tensor(W, requires_grad=True)
        self.X = torch.tensor(X, requires_grad=False)
        self.y = torch.tensor(y, requires_grad=False)
        self.W_grad_norm_list = []
        self.loss_vals = []
        self.n, self.d = self.X.shape[0], self.X[0]
        self.lambda_reg = lambda_reg

    def loss(self):
        return (
            torch.norm(self.W @ self.X - self.y) ** 2 / (2.0 * self.n)
            + self.lambda_reg * torch.norm(self.W) ** 2 / 2.0
        )

    def loss_i(self, indices):
        # Compute batch
        X_batch = self.X[indices, :]
        W_batch = self.W[:, indices]
        WX_batch = (W_batch @ X_batch)[indices]
        y_batch = self.y[indices, :]

        loss = (1 / (2 * self.n)) * torch.norm(WX_batch - y_batch) ** 2
        reg = (self.lambda_reg / 2.0) * torch.norm(self.W) ** 2
        return loss + reg

    # Compute one gradient algorithm step
    def step(self, lr, indices):
        # Compute batch loss
        loss = self.loss_i(indices)
        self.loss_vals.append(loss.item())

        # Compute gradient
        loss.backward()

        # Update parameter
        with torch.no_grad():
            Wgrad = self.W.grad
            self.W_grad_norm_list.append(torch.norm(Wgrad).item())
            self.W -= lr * Wgrad
        self.W.grad.zero_()  # Reset gradient for next iter
