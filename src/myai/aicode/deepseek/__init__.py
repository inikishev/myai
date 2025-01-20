"""deepseek coded optimizers, i changed some of them
"""
from .bifurcation import BifurcationOptimizer  # almost SGD
from .bouncyball import BouncyBall
from .cellular import CellularAutomaton
from .conv_corr import GradConvCorrOptimizer
from .equilibrium import EquilibriumOptimizer # second order seems quite good (lacks line search but just chain with it)
from .fft_sgd import FFTSGD, FFTMomentum, FrequencyOptimizer
from .fractal import FractalOptimizer
from .gaussian_smoothing import (
    GaussianHomotopy,  # it was supposed to be potential function optimzier but it just made gs
)
from .hill_climbing import ExhaustiveHillClimbing, HillClimbing
from .hopfield import HopfieldOptimizer  # somewhat interesting
from .knot import KnotOptimizer, KnotRegularization  # horribly slow
from .lstsq_gradient_interpolation import (
    GradientExtrapolationSGD,
    GradientLossExtrapolationSGD,
)
from .morse import MorseFlow
from .poincare import (
    PoincareDualStep,  # interesting
    PoincareDualThreshol,  # interesting
    PoincareThreshold,  # meh
)
from .powell import Powell
from .ray_subspace import RaySubspace  # interesting i changed this a lot
from .reliable_gradients import ReliableGradient  # bad
from .rl import RLOptimizer
from .runge_kutta import RungeKutta
from .wavelet import WaveletOptimizer
from .zo_basis_subspace import ZOBasisSubspace
