import os
import pandas as pd
import numpy as np

def generate_dataset():
    features = [
        "mean_magnitude",
        "max_magnitude",
        "spectral_flatness",
        "mean_intensity",
        "skewness",
        "mean_motion",
        "laplacian_variance",
        "edge_density",
        "kurtosis",
        "temporal_instability",
        "log_spectral_energy"
    ]
    
    np.random.seed(42)
    n_samples = 1500
    half_samples = n_samples // 2
    
    data = []
    # Class 0: Nominal CCTV (standard ranges based on clear video profiles)
    for i in range(half_samples):
        row = {
            "mean_magnitude": float(np.random.normal(loc=48.0, scale=3.0)),
            "max_magnitude": float(np.random.normal(loc=630.0, scale=30.0)),
            "spectral_flatness": float(np.random.normal(loc=0.98, scale=0.005)),
            "mean_intensity": float(np.random.normal(loc=128.0, scale=10.0)),
            "skewness": float(np.random.normal(loc=0.0, scale=0.05)),
            "mean_motion": float(np.random.normal(loc=15.0, scale=2.0)),
            "laplacian_variance": float(np.random.normal(loc=130.0, scale=15.0)),
            "edge_density": float(np.random.normal(loc=0.35, scale=0.05)),
            "kurtosis": float(np.random.normal(loc=-1.2, scale=0.05)),
            "temporal_instability": float(np.random.normal(loc=5.0, scale=1.0)),
            "log_spectral_energy": float(np.random.normal(loc=22.5, scale=0.3)),
            "label": 0,
            "video_id": f"vid_0_{i // 10}",
            "is_synthetic": 1
        }
        data.append(row)
        
    # Class 1: Tampered/Anomaly (covering blur, black/white, noisy, and lens covers)
    for i in range(half_samples):
        rand_val = np.random.rand()
        
        # Initialize default values
        mean_mag = float(np.random.normal(loc=45.0, scale=5.0))
        max_mag = float(np.random.normal(loc=480.0, scale=50.0))
        spec_flat = float(np.random.normal(loc=0.96, scale=0.01))
        mean_int = float(np.random.normal(loc=128.0, scale=15.0))
        skew = float(np.random.normal(loc=0.0, scale=0.1))
        motion = float(np.random.normal(loc=18.0, scale=3.0))
        lap_var = float(np.random.normal(loc=45.0, scale=15.0))
        edge = float(np.random.normal(loc=0.15, scale=0.05))
        kurt = float(np.random.normal(loc=-1.2, scale=0.1))
        temp_inst = float(np.random.normal(loc=10.0, scale=2.0))
        log_spec_energy = float(np.random.normal(loc=21.0, scale=1.0))
        
        if rand_val < 0.4:
            # 1. Blur anomaly (low laplacian variance, but slightly wider to cover 70)
            lap_var = float(np.random.normal(loc=50.0, scale=12.0))
            edge = float(np.random.normal(loc=0.25, scale=0.05))
        elif rand_val < 0.6:
            # 2. Black video (extremely low intensity and laplacian variance)
            lap_var = float(np.random.normal(loc=1.0, scale=0.5))
            mean_int = float(np.random.normal(loc=2.0, scale=1.0))
            edge = float(np.random.normal(loc=0.01, scale=0.005))
            mean_mag = float(np.random.normal(loc=1.0, scale=0.5))
            max_mag = float(np.random.normal(loc=10.0, scale=2.0))
        elif rand_val < 0.8:
            # 3. White video (extremely high intensity, low laplacian variance)
            lap_var = float(np.random.normal(loc=1.0, scale=0.5))
            mean_int = float(np.random.normal(loc=253.0, scale=1.0))
            edge = float(np.random.normal(loc=0.01, scale=0.005))
            mean_mag = float(np.random.normal(loc=1.0, scale=0.5))
            max_mag = float(np.random.normal(loc=10.0, scale=2.0))
        else:
            # 4. Noisy/Random video (extremely high laplacian variance and edge density)
            lap_var = float(np.random.normal(loc=300.0, scale=50.0))
            edge = float(np.random.normal(loc=0.8, scale=0.1))
            mean_int = float(np.random.normal(loc=128.0, scale=5.0))
            temp_inst = float(np.random.normal(loc=25.0, scale=5.0))
            
        row = {
            "mean_magnitude": mean_mag,
            "max_magnitude": max_mag,
            "spectral_flatness": spec_flat,
            "mean_intensity": mean_int,
            "skewness": skew,
            "mean_motion": motion,
            "laplacian_variance": max(0.1, lap_var),
            "edge_density": max(0.001, edge),
            "kurtosis": kurt,
            "temporal_instability": temp_inst,
            "log_spectral_energy": log_spec_energy,
            "label": 1,
            "video_id": f"vid_1_{i // 10}",
            "is_synthetic": 1
        }
        data.append(row)
            
    df = pd.DataFrame(data)
    out_dir = os.path.normpath("datasets/core/uhctd/raw")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "uhctd_features.csv"), index=False)
    print(f"[DATASET] Generated synthetic uhctd features dataset at datasets/core/uhctd/raw/uhctd_features.csv with {n_samples} samples.")

if __name__ == "__main__":
    generate_dataset()
