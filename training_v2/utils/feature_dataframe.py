import pandas as pd
import numpy as np

def build_feature_dataframe(features, feature_names):
    """
    Constructs a pandas DataFrame from raw features and asserts that columns
    match the expected feature ordering exactly.
    """
    if isinstance(features, (np.ndarray, list)):
        features_list = list(features)
    else:
        features_list = features

    # Determine if it's 1D or 2D
    if len(features_list) > 0 and isinstance(features_list[0], (list, np.ndarray)):
        data = features_list
    else:
        data = [features_list]

    df = pd.DataFrame(data, columns=feature_names)
    if list(df.columns) != list(feature_names):
        raise RuntimeError(
            f"Feature ordering mismatch between training and inference.\n"
            f"Expected: {feature_names}\n"
            f"Received: {list(df.columns)}"
        )
    return df
