import time
import logging
import requests
from backend.notifications.telegram_settings import TelegramSettings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TelegramService:
    def __init__(self):
        self.bot_token = None
        self.chat_id = None
        self.initialize()

    def initialize(self):
        """Loads Bot Token and Chat ID from settings."""
        self.bot_token = TelegramSettings.get_bot_token()
        self.chat_id = TelegramSettings.get_chat_id()

    def validate_configuration(self) -> bool:
        """Validates that Bot Token and Chat ID are present."""
        self.initialize()
        return bool(self.bot_token and self.chat_id)

    def format_message(self, event: dict) -> str:
        """Formats the caption message template exactly as requested."""
        camera_name = event.get("camera_name", "Gate-1")
        tamper_type = event.get("tamper_type", "UNKNOWN")
        severity = event.get("severity", "HIGH").upper()
        
        conf = event.get("confidence", 0.0)
        if isinstance(conf, (int, float)):
            if conf <= 1.0:
                conf = conf * 100.0
            conf_str = f"{conf:.2f}%"
        else:
            conf_str = f"{conf}"
            if not conf_str.endswith("%"):
                conf_str += "%"

        ts = event.get("timestamp", "Unknown")
        # Normalize timestamp format: 2026-08-06_11-45-08 -> 2026-08-06 11:45:08
        if "_" in ts:
            try:
                date_part, time_part = ts.split("_")
                time_part = time_part.replace("-", ":")
                ts = f"{date_part} {time_part}"
            except Exception:
                pass
        elif "T" in ts:
            try:
                ts = ts.replace("T", " ").split(".")[0]
            except Exception:
                pass

        msg = (
            f"🚨 SpectraGuard Alert\n\n"
            f"Camera:\n{camera_name}\n\n"
            f"Tampering:\n{tamper_type}\n\n"
            f"Severity:\n{severity}\n\n"
            f"Confidence:\n{conf_str}\n\n"
            f"Time:\n{ts}"
        )
        return msg

    def send_message(self, chat_id=None, text=None) -> dict:
        """Sends a text message using POST /sendMessage."""
        if not self.validate_configuration():
            raise ValueError("Telegram Bot Token or Chat ID is not configured.")

        target_chat_id = chat_id or self.chat_id
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": target_chat_id,
            "text": text
        }

        logging.info(f"Sending Telegram message to {target_chat_id}")
        response = requests.post(url, data=data, timeout=15)
        
        logging.info(f"Telegram HTTP Status: {response.status_code}")
        try:
            res_json = response.json()
            logging.info(f"Telegram JSON Response: {res_json}")
        except Exception as e:
            res_json = {"ok": False, "error": f"Failed to parse JSON response: {e}"}

        if response.status_code != 200 or not res_json.get("ok"):
            raise RuntimeError(f"Telegram API /sendMessage failed: {response.status_code} - {res_json}")

        return res_json

    def send_photo(self, chat_id=None, photo_path=None, caption=None) -> dict:
        """Sends a photo using POST /sendPhoto with multipart/form-data."""
        if not self.validate_configuration():
            raise ValueError("Telegram Bot Token or Chat ID is not configured.")

        target_chat_id = chat_id or self.chat_id
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        
        data = {
            "chat_id": target_chat_id,
            "caption": caption
        }

        logging.info(f"Sending Telegram photo to {target_chat_id} from {photo_path}")
        with open(photo_path, "rb") as photo_file:
            files = {
                "photo": photo_file
            }
            response = requests.post(url, data=data, files=files, timeout=30)

        logging.info(f"Telegram HTTP Status: {response.status_code}")
        try:
            res_json = response.json()
            logging.info(f"Telegram JSON Response: {res_json}")
        except Exception as e:
            res_json = {"ok": False, "error": f"Failed to parse JSON response: {e}"}

        if response.status_code != 200 or not res_json.get("ok"):
            raise RuntimeError(f"Telegram API /sendPhoto failed: {response.status_code} - {res_json}")

        return res_json

    def test_connection(self, chat_id=None) -> dict:
        """Sends the configured test connection message."""
        return self.send_message(chat_id=chat_id, text="✅ SpectraGuard Telegram successfully configured.")
