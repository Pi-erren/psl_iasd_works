import numpy as np
from scipy.linalg import norm  # Norm euclidienne (correspondant au produit scalaire)
from scipy.linalg import svdvals  # Décomposition en valeurs singulières
from problems.regression import RegPb


def gd(
    problem,
    stepchoice=0,
    init_step=1,
    n_iter=1000,
    grad_dist_criteria=None,
    verbose=False,
):
    """
    Most of this function was borrowed from Clément Royer: https://www.lamsade.dauphine.fr/%7Ecroyer/ensdocs/OAA/LabOAA-DescenteGradient-solutions.zip
    Implémentation de descente de gradient avec différentes tailles de pas.

    Entrées:
        w0: Point initial
        problem: Instance à minimiser
            problem.fun(x) Fonction objectif
            problem.grad(x) Gradient
            problem.lipgrad() Constante de Lipschitz pour le gradient
        grad_dist_criteria (float): stop algorithm when dist(w_{i+1} - w_{i}) <  grad_dist_criteria
        stepchoice: Choix de taille de pas
            0: Constante proportionnelle à 1/L (L constante de Lipschitz pour le gradient)
            a>0: Décroissante, set to 1/((k+1)**a)
        init_step: Taille de pas initiale
        n_iter: Maximum d'itérations
        verbose: Affichage d'informations à chaque itération

    Sorties:
        w_output: Dernier itéré de la méthode
        loss_val: Historique de valeurs de fonctions (tableau Numpy de taille n_iter+1)
        distits: Historique de distances à l'optimum cible (tableau Numpy de taille n_iter+1)
    """
    if stepchoice is None:
        return ValueError("Please select a stepchoice")

    # Utils variables
    loss_val = []
    k = 0

    # Algorithm loop
    while k < n_iter:
        s = choose_step_size(stepchoice, init_step, k)
        problem.step(s)
        curr_loss = problem.loss()

        if verbose:
            print(
                " | ".join(
                    [
                        ("%d" % k).rjust(8),
                        ("%.2e" % curr_loss).rjust(8),
                        ("%.2e" % s).rjust(8),
                    ]
                )
            )

        # Update algo values
        # if
        loss_val.append(curr_loss.detach().numpy())
        k += 1
    # if is
    return np.array(loss_val)


def choose_step_size(mode, init_step, k):
    # 1 - Définir la taille de pas s en fonction de k, L, step0 et g
    if mode == 0:
        # Taille de pas constante
        return init_step
    elif mode > 0:
        # Taille de pas décroissante
        return init_step / (k + 1) ** mode
