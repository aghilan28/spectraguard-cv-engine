import time
import threading

class RateLimiter:
    def __init__(self, cooldown_seconds: int = 30):
        self.cooldown = cooldown_seconds
        self.history = {}
        self.lock = threading.Lock()

    def allow_notification(self, camera_name: str, tamper_type: str) -> bool:
        current_time = time.time()
        key = (camera_name, tamper_type)
        
        with self.lock:
            last_sent = self.history.get(key, 0)
            if current_time - last_sent >= self.cooldown:
                self.history[key] = current_time
                return True
            return False
