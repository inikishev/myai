"""deepseek coded optimizers, some of them i changed a lot tho

good ones

- PoincareDualThreshold seems insane

- FFTSGD/FrequencyOptimizer maybe interesting
"""
from .bouncyball import BouncyBall
from .fft_sgd import FFTSGD, FrequencyOptimizer, FFTMomentum
from .hill_climbing import HillClimbing, ExhaustiveHillClimbing
from .lstsq_gradient_interpolation import GradientExtrapolationSGD, GradientLossExtrapolationSGD
from .powell import Powell
from .ray_subspace import RaySubspace
from .reliable_gradients import ReliableGradient
from .rl import RLOptimizer
from .zo_basis_subspace import ZOBasisSubspace
from .patch_whitening import CellularAutomaton
from .poincare import PoincareThreshold, PoincareDualStep, PoincareDualThreshold