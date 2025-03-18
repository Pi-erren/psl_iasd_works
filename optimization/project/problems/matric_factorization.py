import numpy as np
from scipy.linalg import norm  # Norm euclidienne (correspondant au produit scalaire)
from scipy.linalg import svdvals  # Décomposition en valeurs singulières
import torch
from torch import nn

from problem import Problem
from utils.utils import frobenius_norm


class MatFactPb(Problem):
    """
    Matrix factorization problem class

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
        self.V = torch.tensor(U)
        self.V.requires_grad = True
        self.X = torch.tensor(X)
        self.n1, self.n2 = self.U.shape[0], self.V.shape[0]
        self.n, self.d = self.X.shape
        self.lambda_reg = lambda_reg
        self.optimizer = torch.optim.SGD

    def fun(self, U, V, X, lambda_reg):
        return frobenius_norm(U, V, X, lambda_reg)

    # Calcul de gradient
    def step(self, *args):
        
        return self.U.grad, self.V.grad

 