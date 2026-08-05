class CameraError(Exception):
    """Base exception for all CCTV application errors."""
    pass

class CameraConnectionError(CameraError):
    """Raised when a connection to the camera cannot be established."""
    pass

class CameraReadError(CameraError):
    """Raised when frame retrieval fails or drops consistently."""
    pass

class CameraAuthenticationError(CameraError):
    """Raised when authentication credentials fail."""
    pass

class InvalidRTSPError(CameraError):
    """Raised when the generated RTSP string fails syntactical or structural rules."""
    pass
