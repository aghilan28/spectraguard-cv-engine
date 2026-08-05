from pydantic import BaseModel, Field, field_validator

class CameraConfig(BaseModel):
    """
    Configuration model for a CCTV camera.
    Handles data validation and defaults for RTSP connections.
    """
    name: str = Field(default="Camera 1", description="Friendly name for the camera")
    ip_address: str = Field(..., description="IP address or hostname of the camera")
    port: int = Field(default=554, ge=1, le=65535, description="Connection port")
    username: str = Field(default="admin", description="Authentication username")
    password: str = Field(default="", description="Authentication password")
    protocol: str = Field(default="rtsp", description="Streaming protocol")
    stream_path: str = Field(default="/", description="Specific stream path/channel")

    @field_validator('ip_address')
    @classmethod
    def validate_ip(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("IP address/hostname cannot be empty")
        return v.strip()

    @field_validator('protocol')
    @classmethod
    def validate_protocol(cls, v: str) -> str:
        v = v.lower()
        allowed = ['rtsp', 'http', 'https', 'rtmp']
        if v not in allowed:
            raise ValueError(f"Protocol must be one of {allowed}")
        return v

    @field_validator('stream_path')
    @classmethod
    def format_stream_path(cls, v: str) -> str:
        # Ensure the stream path starts with a slash
        if v and not v.startswith('/'):
            return f"/{v}"
        if not v:
            return "/"
        return v
