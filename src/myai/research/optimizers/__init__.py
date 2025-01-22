"""deepseek coded optimizers, i changed some of them

IMPORTANT

for many of them closure should look like this

def closure(backward=True):
    loss = ...
    if backward:
        opt.zero_grad()
        loss.backward()
    return loss

IMPORTANT

many of them SUCK

so far there are NO good ones for mini-batch training (maybe some for specific tasks but i havent had any luck)

but a few are good for minimization (especially GradCobyla its insane)
"""
from .bifurcation import BifurcationOptimizer
# seems worse than SGD

from .bouncyball import BouncyBall
# incorrect and horrible (I am manually making correct one)

from .cellular import CellularAutomaton
# normalizes gradients for each weight by neigbours USING A FOR LOOP enjoy

from .conv_corr import GradConvCorrOptimizer
# extremely slow and horrible convergence

from .equilibrium import EquilibriumOptimizer
# second order and evaluates hvps to solve HΔθ = -g with CG, so on test functions seems good

from .fft_sgd import FFTSGD, FFTMomentum, FrequencyOptimizer
# FFTSGD - very slow and horrible convergence
# FrequencyOptimizer - very slow, convergence not better than SGD
# the momentum one is bad too

from .fractal import FractalOptimizer
# SGD but seems to waste 2 additional evaluates per step which doesn't help at all
# tho code for generating those fractal petrubations is interesting

from .gaussian_smoothing import GaussianHomotopy
# it was supposed to be potential function optimzier but it just made gaussian smoothing i added sigma decay

from .graph_traversal import GraphTraversalOptimizer
# a lot of crazy code for what i dont think is any better than stochastic hill climbing although more testing needed

from .hill_climbing import ExhaustiveHillClimbing, HillClimbing
# brute force hill climbing

from .hopfield import HopfieldOptimizer
# potentially interesting, performance and LR similar to SGD.

from .knot import KnotOptimizer, KnotRegularization
# this one is INSANE, bro coded a textbook, its horribly slow and i will not even attempt to understand what it does

from .lstsq_gradient_interpolation import GradientExtrapolationSGD, GradientLossExtrapolationSGD
# extrapolates gradients with least squares. which is just momentum.
# But the other one also uses loss and seems kinda interesting but unstable
# also won't work with more than 5000 parameters due to memory

from .morse import MorseFlow
# seems just worse in every way compared to SGD or Adam

from .poincare import PoincareDualStep, PoincareDualThreshold, PoincareThreshold
# all of them seem bad

from .coord_search import CoordSearch
# coord search

from .ray_subspace import RaySubspace
# i changed this a lot and added a giant line search with accelerated random search in the end if it completely fails
# and i added more directions to the subspace
# so it wont be unstable due to line search but something is wrong i think because somethimes loss goes up which shouldn't happen
# with line search
# this is hoped to approximate newtons method well with finite differences
# not for minibtach!

from .reliable_gradients import ReliableGradient
# Dumbest Optimizer ever

from .rl import RLOptimizer
# uses a reinforcement learning agent to minimize a function, not very useful but interesting and actually works

from .runge_kutta import RungeKutta
# uses this ODE integration method but I don't think that does anything tbh

from .wavelet import WaveletOptimizer
# applies wavelet transform to flattened gradients so i dont think this can be any good

from .zo_basis_subspace import ZOBasisSubspace
# zeroth order optimizer that is supposed to refine a good subspace but it seems slow

from .fft_subspace_crn import FFTSubspaceSCRN
# cubic regularized newton in a subspace extracted from gradient vector as top fft frequencies
# actually works quite well but subspace ndim should be quite high like 20
# and eventually it stales

from .stp import ImprovedSTP
# better STP with few of my improvements its pretty good with large history size but uses a lot of memory

from .third_order import CubicPrecond
# yes this is claimed to be third order preconditioning with just the gradients
# but convergence on neural net is bad

from .insane import INSANE
# probably the worst optimizer ever made

from .evolutionary import EvolutionaryGradientOptimizer
# just a pattern moving in gradient direction, seems somewhat okay?

from .stationary import StationaryOptimizer
# seems to be similar to LBFGS, very cheap computationally

from .cash_karp import CashKarp
# idk for now

from .cobyla_grad import CobylaGrad
# this is actually a really good one (not for minibatch). probably the best one it made

from .powell_grad import PowellGrad
# good too

from .fuzzy import FuzzyLogicOptimizer
# something fuzzy logic related but seems similar to SGD

from .directional_exp_model import DirectionalExp
# directional step minimizing an exponential model

from .tame import TAME
# better than adam?

from .fft_precond import FourierPreconditioner
# baababababbab