"""nflstate: dynamic Bayesian team ratings for NFL game forecasting.

Modules
-------
data      load and clean the nflverse games file, build pre-game features
kalman    linear-Gaussian state-space model: Kalman filter + RTS smoother
model     hyperparameter estimation by one-step-ahead predictive likelihood
elo       538-style Elo baseline
market    betting-market probabilities (vig removal, spread -> probability)
evaluate  proper scoring rules, calibration, bootstrap and ATS summaries
"""

__version__ = "0.1.0"
