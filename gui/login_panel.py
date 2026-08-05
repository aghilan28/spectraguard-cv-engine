import json
import os
from PyQt6.QtWidgets import (QWidget, QFormLayout, QLineEdit, QComboBox, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QDialog, QDialogButtonBox)
from PyQt6.QtCore import pyqtSignal

SETTINGS_FILE = "config/user_settings.json"

import json
import os
import threading
from PyQt6.QtWidgets import (QWidget, QFormLayout, QLineEdit, QComboBox, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QDialog, QDialogButtonBox,
                             QTabWidget, QCheckBox, QLabel, QGroupBox)
from PyQt6.QtCore import pyqtSignal

SETTINGS_FILE = "config/user_settings.json"

class SettingsDialog(QDialog):
    """Dialog popup for configuring notification contacts and Telegram settings."""
    def __init__(self, current_numbers: list, telegram_enabled: bool, telegram_token: str, telegram_chat_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Notification Settings")
        self.resize(450, 320)
        
        layout = QVBoxLayout(self)
        
        # SMS Group
        sms_group = QGroupBox("SMS Alert Settings")
        sms_layout = QFormLayout(sms_group)
        current_text = ", ".join(current_numbers)
        self.phone_input = QLineEdit(current_text)
        self.phone_input.setPlaceholderText("+919876543210, +918072264018")
        sms_layout.addRow("Emergency Numbers:", self.phone_input)
        layout.addWidget(sms_group)
        
        # Telegram Group
        telegram_group = QGroupBox("Telegram Notification Settings")
        telegram_layout = QFormLayout(telegram_group)
        
        self.tg_enabled_cb = QCheckBox("Enable Telegram Notifications")
        self.tg_enabled_cb.setChecked(telegram_enabled)
        
        self.tg_token_input = QLineEdit(telegram_token)
        self.tg_token_input.setPlaceholderText("Enter Telegram Bot Token")
        
        self.tg_chat_id_input = QLineEdit(telegram_chat_id)
        self.tg_chat_id_input.setPlaceholderText("Enter Telegram Chat ID")
        
        self.btn_test_conn = QPushButton("Test Connection")
        self.btn_test_conn.clicked.connect(self.test_telegram_connection)
        
        telegram_layout.addRow(self.tg_enabled_cb)
        telegram_layout.addRow("Telegram Bot Token:", self.tg_token_input)
        telegram_layout.addRow("Telegram Chat ID:", self.tg_chat_id_input)
        telegram_layout.addRow("", self.btn_test_conn)
        
        layout.addWidget(telegram_group)
        
        # Standard buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_numbers(self) -> list:
        text = self.phone_input.text().strip()
        if not text:
            return []
        return [num.strip() for num in text.split(",") if num.strip()]

    def get_telegram_settings(self) -> dict:
        return {
            "enabled": self.tg_enabled_cb.isChecked(),
            "bot_token": self.tg_token_input.text().strip(),
            "chat_id": self.tg_chat_id_input.text().strip()
        }

    def test_telegram_connection(self):
        token = self.tg_token_input.text().strip()
        chat_id = self.tg_chat_id_input.text().strip()
        
        if not token or not chat_id:
            QMessageBox.warning(self, "Configuration Error", "Both Telegram Bot Token and Chat ID are required for test.")
            return

        self.btn_test_conn.setEnabled(False)
        self.btn_test_conn.setText("Testing...")

        # Run connection test in a separate thread to prevent GUI lockup
        def run_test():
            try:
                # Dynamically set environments or pass to service
                from backend.notifications.telegram_service import TelegramService
                svc = TelegramService()
                svc.bot_token = token
                svc.chat_id = chat_id
                
                # Perform the real API call
                res = svc.test_connection(chat_id=chat_id)
                
                # Success callback
                def on_success():
                    self.btn_test_conn.setEnabled(True)
                    self.btn_test_conn.setText("Test Connection")
                    QMessageBox.information(self, "Success", "✅ SpectraGuard Telegram successfully configured.\n\nMessage sent successfully.")
                
                from PyQt6.QtCore import QMetaObject, Q_ARG
                QMetaObject.invokeMethod(self, "on_test_success", Qt.ConnectionType.QueuedConnection)
            except Exception as e:
                # Failure callback
                def on_failure():
                    self.btn_test_conn.setEnabled(True)
                    self.btn_test_conn.setText("Test Connection")
                    QMessageBox.critical(self, "Failure", f"Connection test failed:\n{e}")
                
                from PyQt6.QtCore import QMetaObject, Q_ARG
                QMetaObject.invokeMethod(self, "on_test_failure", Qt.ConnectionType.QueuedConnection, Q_ARG(str, str(e)))

        threading.Thread(target=run_test, daemon=True).start()

    from PyQt6.QtCore import pyqtSlot
    @pyqtSlot()
    def on_test_success(self):
        self.btn_test_conn.setEnabled(True)
        self.btn_test_conn.setText("Test Connection")
        QMessageBox.information(self, "Success", "✅ SpectraGuard Telegram successfully configured.\n\nMessage sent successfully.")

    @pyqtSlot(str)
    def on_test_failure(self, err_msg):
        self.btn_test_conn.setEnabled(True)
        self.btn_test_conn.setText("Test Connection")
        QMessageBox.critical(self, "Failure", f"Connection test failed:\n{err_msg}")


class LoginPanel(QWidget):
    """
    Handles CCTV connection parameter inputs, device profile mapping,
    and field validation constraints prior to camera engagement.
    Includes persistent state saving via local JSON.
    """
    connect_requested = pyqtSignal(dict)
    disconnect_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phone_numbers = []
        self.telegram_enabled = True
        self.telegram_token = ""
        self.telegram_chat_id = ""
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Camera 1")
        
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("192.168.1.100")
        
        self.port_input = QLineEdit("554")
        
        self.user_input = QLineEdit("admin")
        
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.vendor_select = QComboBox()
        self.vendor_select.addItems(["Generic", "Hikvision", "Dahua", "CP Plus", "Axis"])

        form_layout.addRow("Camera Name:", self.name_input)
        form_layout.addRow("IP Address:", self.ip_input)
        form_layout.addRow("Port:", self.port_input)
        form_layout.addRow("Username:", self.user_input)
        form_layout.addRow("Password:", self.pass_input)
        form_layout.addRow("Vendor Config:", self.vendor_select)
        main_layout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        self.btn_connect = QPushButton("CONNECT")
        self.btn_disconnect = QPushButton("DISCONNECT")
        self.btn_disconnect.setEnabled(False) 
        self.btn_settings = QPushButton("SETTINGS")

        btn_layout.addWidget(self.btn_connect)
        btn_layout.addWidget(self.btn_disconnect)
        btn_layout.addWidget(self.btn_settings)
        main_layout.addLayout(btn_layout)

        self.btn_connect.clicked.connect(self._validate_and_submit)
        self.btn_disconnect.clicked.connect(self.disconnect_requested.emit)
        self.btn_settings.clicked.connect(self.open_settings)

    def open_settings(self):
        dialog = SettingsDialog(self.phone_numbers, self.telegram_enabled, self.telegram_token, self.telegram_chat_id, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.phone_numbers = dialog.get_numbers()
            tg_settings = dialog.get_telegram_settings()
            self.telegram_enabled = tg_settings["enabled"]
            self.telegram_token = tg_settings["bot_token"]
            self.telegram_chat_id = tg_settings["chat_id"]
            
            # Save connection and phone details to file immediately
            payload = self.build_payload()
            self.save_settings(payload)
            QMessageBox.information(self, "Success", "Notification settings saved successfully.")

    def build_payload(self) -> dict:
        ip = self.ip_input.text().strip()
        user = self.user_input.text().strip()
        pwd = self.pass_input.text().strip()
        name = self.name_input.text().strip() or "Camera 1"
        
        return {
            "name": name,
            "ip_address": ip,
            "port": int(self.port_input.text().strip() or "554"),
            "username": user,
            "password": pwd,
            "vendor": self.vendor_select.currentText().lower(),
            
            "camera_name": name,
            "camera_ip": ip,
            "emergency_phone": self.phone_numbers[0] if self.phone_numbers else "",
            "phone_numbers": self.phone_numbers,
            "emergency_contacts": self.phone_numbers,
            
            "telegram": {
                "enabled": self.telegram_enabled,
                "bot_token": self.telegram_token,
                "chat_id": self.telegram_chat_id
            }
        }

    def load_settings(self):
        """Loads saved connection parameters from local JSON if available."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    name = data.get("camera_name") or data.get("name", "")
                    ip = data.get("camera_ip") or data.get("ip_address", "")
                    
                    self.name_input.setText(name)
                    self.ip_input.setText(ip)
                    self.port_input.setText(str(data.get("port", "554")))
                    self.user_input.setText(data.get("username", ""))
                    self.pass_input.setText(data.get("password", ""))
                    
                    vendor = data.get("vendor", "generic").title()
                    index = self.vendor_select.findText(vendor)
                    if index >= 0:
                        self.vendor_select.setCurrentIndex(index)
                        
                    self.phone_numbers = data.get("emergency_contacts", []) or data.get("phone_numbers", [])
                    if not self.phone_numbers and data.get("emergency_phone"):
                        self.phone_numbers = [data.get("emergency_phone")]
                        
                    # Load Telegram settings
                    tg_data = data.get("telegram", {})
                    self.telegram_enabled = tg_data.get("enabled", True)
                    self.telegram_token = tg_data.get("bot_token", "")
                    self.telegram_chat_id = tg_data.get("chat_id", "")
            except Exception as e:
                print(f"[GUI] Failed to load settings: {e}")

    def save_settings(self, payload):
        """Saves successful connection parameters to local JSON."""
        try:
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
        except Exception as e:
            print(f"[GUI] Failed to save settings: {e}")

    def _validate_and_submit(self):
        ip = self.ip_input.text().strip()
        user = self.user_input.text().strip()
        pwd = self.pass_input.text().strip()

        if not ip:
            QMessageBox.warning(self, "Validation Error", "IP Address / Host Target cannot be blank.")
            return
        if not user:
            QMessageBox.warning(self, "Validation Error", "Username field cannot be blank.")
            return
        if not pwd:
            QMessageBox.warning(self, "Validation Error", "Password field cannot be blank.")
            return

        config_payload = self.build_payload()
        self.save_settings(config_payload)
        self.connect_requested.emit(config_payload)

    def set_connected_state(self, connected: bool):
        self.btn_connect.setEnabled(not connected)
        self.btn_disconnect.setEnabled(connected)
        self.btn_settings.setEnabled(not connected)

