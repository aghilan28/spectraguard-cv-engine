import urllib.parse
from camera.camera_config import CameraConfig
from enum import Enum

class CameraBrand(Enum):
    GENERIC = "generic"
    HIKVISION = "hikvision"
    DAHUA = "dahua"
    CP_PLUS = "cp_plus"
    AXIS = "axis"

class RTSPBuilder:
    """
    Constructs valid RTSP URLs based on CameraConfig and hardware brand.
    Safely percent-encodes credentials containing special characters.
    """
    
    @staticmethod
    def _get_credentials(config: CameraConfig) -> str:
        if config.username or config.password:
            # URL-encode credentials to safely handle characters like '@' or ':'
            safe_user = urllib.parse.quote(config.username)
            safe_pass = urllib.parse.quote(config.password)
            return f"{safe_user}:{safe_pass}@"
        return ""

    @classmethod
    def build_url(cls, config: CameraConfig, brand: CameraBrand = CameraBrand.GENERIC) -> str:
        credentials = cls._get_credentials(config)
        base_url = f"{config.protocol}://{credentials}{config.ip_address}:{config.port}"

        if brand == CameraBrand.HIKVISION:
            return f"{base_url}/Streaming/Channels/101"
            
        elif brand in (CameraBrand.DAHUA, CameraBrand.CP_PLUS):
            return f"{base_url}/cam/realmonitor?channel=1&subtype=0"
            
        elif brand == CameraBrand.AXIS:
            return f"{base_url}/axis-media/media.amp"
            
        else:
            return f"{base_url}{config.stream_path}"
