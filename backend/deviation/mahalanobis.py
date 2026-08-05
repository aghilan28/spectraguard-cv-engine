import numpy as np
from typing import Optional
from backend.config.logging import logger

def calculate_mahalanobis_distance(
    live_vector: np.ndarray,
    means: np.ndarray,
    covariance_matrix: Optional[np.ndarray] = None
) -> float:
    """
    Computes multivariate distance footprint accounting for covariance properties.
    Performs ridge regularization automatically if covariance evaluates to singular.
    """
    delta = live_vector - means
    dim = len(live_vector)
    
    if covariance_matrix is None or covariance_matrix.shape != (dim, dim):
        covariance_matrix = np.eye(dim, dtype=np.float64)

    try:
        det = np.linalg.det(covariance_matrix)
        if np.abs(det) < 1e-6:
            covariance_matrix = covariance_matrix + np.eye(dim) * 1e-6
        
        inv_cov = np.linalg.inv(covariance_matrix)
        distance = np.sqrt(np.dot(np.dot(delta, inv_cov), delta))
        return float(distance)
    except np.linalg.LinAlgError:
        try:
            reg_cov = covariance_matrix + np.eye(dim) * 1e-4
            inv_cov = np.linalg.inv(reg_cov)
            return float(np.sqrt(np.dot(np.dot(delta, inv_cov), delta)))
        except Exception as err:
            logger.error(f"Covariance matrix inversion failure dropped to baseline: {err}")
            return float(np.sqrt(np.dot(delta, delta)))
