import os
import json
from dotenv import load_dotenv

# Load .env file using python-dotenv
load_dotenv()

class TelegramSettings:
    SETTINGS_FILE = "config/user_settings.json"

    @staticmethod
    def get_bot_token() -> str:
        # Check environment first
        token = os.getenv("BOT_TOKEN", "").strip()
        if token:
            return token
        # Fallback to user settings json
        if os.path.exists(TelegramSettings.SETTINGS_FILE):
            try:
                with open(TelegramSettings.SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("telegram", {}).get("bot_token", "").strip()
            except Exception:
                pass
        return ""

    @staticmethod
    def get_chat_id() -> str:
        # Check environment first
        chat_id = os.getenv("CHAT_ID", "").strip()
        if chat_id:
            return chat_id
        # Fallback to user settings json
        if os.path.exists(TelegramSettings.SETTINGS_FILE):
            try:
                with open(TelegramSettings.SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return str(data.get("telegram", {}).get("chat_id", "")).strip()
            except Exception:
                pass
        return ""

    @staticmethod
    def is_telegram_enabled() -> bool:
        # Check environment first
        enabled_env = os.getenv("TELEGRAM_ENABLED")
        if enabled_env is not None:
            return enabled_env.lower() == "true"
        # Fallback to user settings json
        if os.path.exists(TelegramSettings.SETTINGS_FILE):
            try:
                with open(TelegramSettings.SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return bool(data.get("telegram", {}).get("enabled", True))
            except Exception:
                pass
        return True

    @staticmethod
    def validate_configuration() -> bool:
        """Verifies that all required Telegram configurations are populated."""
        token = TelegramSettings.get_bot_token()
        chat_id = TelegramSettings.get_chat_id()
        return bool(token and chat_id)
