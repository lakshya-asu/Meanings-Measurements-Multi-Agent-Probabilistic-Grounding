# Lightweight, dependency-free PDF utilities vendored from your MSP repo.

import numpy as np

def combined_pdf(x, y, z, params, verbose: bool = False):
    """
    Unnormalized combined PDF at (x,y,z) from semantic, metric, and predicate parts.
    params must contain:
      mu_x, mu_y, mu_z, sigma_s,
      x0, y0, z0, d0, sigma_m,
      theta0, phi0, kappa
    """
    mu_x = params['mu_x']; mu_y = params['mu_y']; mu_z = params['mu_z']; sigma_s = params['sigma_s']
    x0 = params['x0']; y0 = params['y0']; z0 = params['z0']
    d0 = params['d0']; sigma_m = params['sigma_m']
    theta0 = params['theta0']; phi0 = params['phi0']; kappa = params['kappa']

    # Semantic (3D Gaussian)
    exponent_semantic = -((x - mu_x)**2 + (y - mu_y)**2 + (z - mu_z)**2) / (2 * sigma_s**2)

    # Metric (Gaussian over radial distance from anchor to target distance d0)
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    exponent_metric = -((r - d0)**2) / (2 * sigma_m**2)

    # Predicate (approx. von Mises–Fisher using spherical angles)
    dx = x - x0; dy = y - y0; dz = z - z0
    r_nonzero = np.maximum(r, 1e-6)
    theta = np.arctan2(dy, dx)           # [-pi, pi]
    phi   = np.arccos(dz / r_nonzero)    # [0, pi]
    exponent_spatial = kappa * (
        np.sin(phi0)*np.sin(phi)*np.cos(theta - theta0) + np.cos(phi0)*np.cos(phi)
    )

    total_exponent = exponent_semantic + exponent_metric + exponent_spatial
    if verbose:
        print("params:", {k: (round(v,4) if isinstance(v,(int,float)) else v) for k,v in params.items()})
    return np.exp(total_exponent)

def combined_logpdf(x, y, z, params, verbose: bool = False):
    """
    Stable log-version of combined_pdf (avoids overflow in exp).
    """
    mu_x = params['mu_x']; mu_y = params['mu_y']; mu_z = params['mu_z']; sigma_s = params['sigma_s']
    x0 = params['x0']; y0 = params['y0']; z0 = params['z0']
    d0 = params['d0']; sigma_m = params['sigma_m']
    theta0 = params['theta0']; phi0 = params['phi0']; kappa = params['kappa']

    # Semantic Gaussian
    exponent_semantic = -((x - mu_x)**2 + (y - mu_y)**2 + (z - mu_z)**2) / (2 * sigma_s**2 + 1e-12)

    # Metric Gaussian over radial distance
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    exponent_metric = -((r - d0)**2) / (2 * sigma_m**2 + 1e-12)

    # Directional kernel (approx. vMF in spherical coords)
    dx = x - x0; dy = y - y0; dz = z - z0
    r_nonzero = np.maximum(r, 1e-6)
    theta = np.arctan2(dy, dx)
    phi   = np.arccos(dz / r_nonzero)
    exponent_spatial = kappa * (
        np.sin(phi0)*np.sin(phi)*np.cos(theta - theta0) + np.cos(phi0)*np.cos(phi)
    )

    total_exponent = exponent_semantic + exponent_metric + exponent_spatial
    if verbose:
        print("logpdf params:", {k: (round(v,4) if isinstance(v,(int,float)) else v) for k,v in params.items()})
    return total_exponent
