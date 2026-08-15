# Lightweight PDF utilities for the MSP kernels (P0 kernel geometry fix).

"""Unnormalized MSP kernel log density: semantic + metric + directional.

Coordinate and angle conventions (Habitat world frame, y is up)
---------------------------------------------------------------
All points and vectors live in the Habitat world frame, where +y is up.
The direction parameters (theta0, phi0) in ``params`` follow the same
convention as the planners and the evaluator (src/evals/angular.py,
yaw_pitch_from_vector; src/planners/vlm_planner_msp.py,
_unit_dir_from_theta_phi):

- theta0 (yaw): angle in the horizontal x-z plane, measured from +x
  toward +z, in radians. yaw(v) = atan2(v_z, v_x).
- phi0 (polar angle): measured from the +y (up) axis, in radians.
  phi0 = 0 is straight up (zenith), phi0 = pi/2 is level (horizontal),
  phi0 = pi is straight down. The evaluator's elevation "pitch" relates
  by pitch = pi/2 - phi0.

The unit mean direction implied by (theta0, phi0) is therefore

    m = ( cos(theta0) * sin(phi0),    # x
          cos(phi0),                  # y (up)
          sin(theta0) * sin(phi0) )   # z

so that yaw_pitch_from_vector(m) in src/evals/angular.py recovers
(degrees(theta0), 90 - degrees(phi0)) exactly. The directional (von
Mises-Fisher) exponent is the frame-free dot product

    exponent_spatial = kappa * dot(m, (p - anchor) / max(||p - anchor||, eps))

with no spherical-coordinate math anywhere in the scoring path.

What was wrong before (the z-polar vs y-up clash)
-------------------------------------------------
The previous implementation expanded the vMF exponent in spherical
coordinates with **z as the polar axis** (theta = atan2(dy, dx),
phi = arccos(dz / r)), while every producer of (theta0, phi0), and the
proposal sampler that generates candidate points, uses the y-up
convention above. The y and z components of the mean direction were
effectively swapped. Consequences: a systematic yaw bias toward the
world x axis, the directional weight vanishing entirely for goals along
world +/-z (cos(theta0) near 0), and "above"/"below" queries being
scored as a preference along horizontal world z instead of up/down.
This module now builds the mean direction with the same y-up formula as
the samplers and the evaluator, so the two agree by construction.

Normalization
-------------
These densities are deliberately unnormalized (no vMF constant, no
radial truncation constant, no Jacobian, no free-space mask). Proper
normalization over the navigable free space Omega_free is deferred to
the navmesh-masking work item (P1), which restricts the density to
navigable cells and computes Z by logsumexp over the grid. Until then,
only argmax comparisons under a single fixed parameter set are
meaningful; integrated quantities are not.

The radial (metric) Gaussian term and the semantic Gaussian term are
unchanged from the previous implementation.
"""

import math

import numpy as np


def unit_direction_from_angles(theta0, phi0):
    """Unit mean direction for yaw theta0 and polar angle phi0 (y-up).

    theta0: yaw in radians, x-z plane, from +x toward +z.
    phi0: polar angle in radians from the +y (up) axis; pi/2 is level.

    Returns (mx, my, mz), a unit vector in the Habitat world frame,
    consistent with src/evals/angular.py yaw_pitch_from_vector and with
    _unit_dir_from_theta_phi in the planners.
    """
    sin_phi = math.sin(phi0)
    return (
        math.cos(theta0) * sin_phi,
        math.cos(phi0),
        math.sin(theta0) * sin_phi,
    )


def combined_logpdf(x, y, z, params, verbose: bool = False):
    """Unnormalized combined log density at (x, y, z).

    params must contain:
      mu_x, mu_y, mu_z, sigma_s,
      x0, y0, z0, d0, sigma_m,
      theta0, phi0, kappa

    (theta0, phi0) follow the y-up convention in the module docstring.
    x, y, z may be scalars or numpy arrays (broadcast together).
    """
    mu_x = params['mu_x']; mu_y = params['mu_y']; mu_z = params['mu_z']; sigma_s = params['sigma_s']
    x0 = params['x0']; y0 = params['y0']; z0 = params['z0']
    d0 = params['d0']; sigma_m = params['sigma_m']
    theta0 = params['theta0']; phi0 = params['phi0']; kappa = params['kappa']

    # Semantic Gaussian (unchanged)
    exponent_semantic = -((x - mu_x)**2 + (y - mu_y)**2 + (z - mu_z)**2) / (2 * sigma_s**2 + 1e-12)

    # Metric Gaussian over radial distance from the anchor (unchanged)
    dx = x - x0; dy = y - y0; dz = z - z0
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    exponent_metric = -((r - d0)**2) / (2 * sigma_m**2 + 1e-12)

    # Directional (vMF) exponent: kappa * dot(m, unit(p - anchor)),
    # with m built under the same y-up convention as the planners and
    # the evaluator. No spherical-coordinate branch math.
    mx, my, mz = unit_direction_from_angles(theta0, phi0)
    r_nonzero = np.maximum(r, 1e-6)
    exponent_spatial = kappa * (mx * dx + my * dy + mz * dz) / r_nonzero

    total_exponent = exponent_semantic + exponent_metric + exponent_spatial
    if verbose:
        print("logpdf params:", {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in params.items()})
    return total_exponent


def combined_pdf(x, y, z, params, verbose: bool = False):
    """Unnormalized combined PDF at (x, y, z): exp of combined_logpdf.

    Same params contract and conventions as combined_logpdf.
    """
    if verbose:
        print("params:", {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in params.items()})
    return np.exp(combined_logpdf(x, y, z, params, verbose=False))
