import numpy as np
from scipy.linalg import norm  # Norm euclidienne (correspondant au produit scalaire)
from scipy.linalg import svdvals  # Décomposition en valeurs singulières
from problems.problem import Problem


class RegPb(Problem):
    """
    Most of the code borrowed from Clément Royer: https://www.lamsade.dauphine.fr/%7Ecroyer/ensdocs/OAA/LabOAA-DescenteGradient-solutions.zip

    Classe de problèmes de régression avec modèles linéaires

    Attributs:
        X: Matrice de données (attributs)
        y: Vecteur de données (labels)
        n,d: Dimensions de X
        loss: Fonction de perte choisie
            'l2': Perte aux moindres carrés
            'logit': Perte logistique
        lbda: Paramètre de régularisation
    """

    # Initialisation
    def __init__(self, X, y, w, lbda=0, loss="l2"):
        self.X = X
        self.y = y
        self.w = w
        self.w_list = [self.w]
        self.g_list = list()
        self.n, self.d = X.shape
        self.loss = loss
        self.lbda = lbda

    def step(self, s):
        g = self.grad(self.w)
        self.w = self.w - s * g

        self.g_list.append(g)
        self.w_list.append(self.w)

    # Fonction objectif
    def fun(self, w):
        if self.loss == "l2":
            return (
                norm(self.X.dot(w) - self.y) ** 2 / (2.0 * self.n)
                + self.lbda * norm(w) ** 2 / 2.0
            )
        elif self.loss == "logit":
            yXw = self.y * self.X.dot(w)
            return np.mean(np.log(1.0 + np.exp(-yXw))) + self.lbda * norm(w) ** 2 / 2.0

    # Calcul de gradient
    def grad(self, w):
        if self.loss == "l2":
            return self.X.T.dot(self.X.dot(w) - self.y) / self.n + self.lbda * w
        elif self.loss == "logit":
            yXw = self.y * self.X.dot(w)
            aux = 1.0 / (1.0 + np.exp(yXw))
            return -(self.X.T).dot(self.y * aux) / self.n + self.lbda * w

    # Constante de Lipschitz pour le gradient
    def lipgrad(self):
        if self.loss == "l2":
            L = norm(self.X, ord=2) ** 2 / self.n + self.lbda
        elif self.loss == "logit":
            L = norm(self.X, ord=2) ** 2 / (4.0 * self.n) + self.lbda
        return L

    # Constante de convexité ''forte'' (potentiellement 0 si self.lbda=0)
    def cvxval(self):
        if self.loss == "l2":
            s = svdvals(self.X)
            mu = min(s) ** 2 / self.n
            return mu + self.lbda
        elif self.loss == "logit":
            return self.lbda
