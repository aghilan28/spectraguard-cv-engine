import json
import os
from PyQt6.QtWidgets import (QWidget, QFormLayout, QLineEdit, QComboBox, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox)
from PyQt6.QtCore import pyqtSignal

SETTINGS_FILE = "settings.json"

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

        btn_layout.addWidget(self.btn_connect)
        btn_layout.addWidget(self.btn_disconnect)
        main_layout.addLayout(btn_layout)

        self.btn_connect.clicked.connect(self._validate_and_submit)
        self.btn_disconnect.clicked.connect(self.disconnect_requested.emit)

    def load_settings(self):
        """Loads saved connection parameters from local JSON if available."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                    self.name_input.setText(data.get("name", ""))
                    self.ip_input.setText(data.get("ip_address", ""))
                    self.port_input.setText(str(data.get("port", "554")))
                    self.user_input.setText(data.get("username", ""))
                    self.pass_input.setText(data.get("password", ""))
                    
                    # Safely map vendor selection index
                    vendor = data.get("vendor", "generic").title()
                    index = self.vendor_select.findText(vendor)
                    if index >= 0:
                        self.vendor_select.setCurrentIndex(index)
            except Exception as e:
                print(f"[GUI] Failed to load settings: {e}")

    def save_settings(self, payload):
        """Saves successful connection parameters to local JSON."""
        try:
            with open(SETTINGS_FILE, "w") as f:
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

        config_payload = {
            "name": self.name_input.text().strip() or "Camera 1",
            "ip_address": ip,
            "port": int(self.port_input.text().strip() or "554"),
            "username": user,
            "password": pwd,
            "vendor": self.vendor_select.currentText().lower()
        }
        
        # Persist data to disk before firing the connection signal
        self.save_settings(config_payload)
        self.connect_requested.emit(config_payload)

    def set_connected_state(self, connected: bool):
        self.btn_connect.setEnabled(not connected)
        self.btn_disconnect.setEnabled(connected)
