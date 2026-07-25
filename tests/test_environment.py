import cv2
import xgboost as xgb
import scipy
import sklearn
import joblib


def test_environment_dependencies():
    # Verify OpenCV is 4.x
    assert cv2.__version__.startswith(
        "4."
    ), f"Expected OpenCV 4.x, got {cv2.__version__}"

    # Verify XGBoost is < 3.0
    xgb_major_version = int(xgb.__version__.split(".")[0])
    assert xgb_major_version < 3, f"Expected XGBoost < 3.0, got {xgb.__version__}"


def test_framework_imports():
    assert scipy is not None
    assert sklearn is not None
    assert joblib is not None
