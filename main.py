import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    """
    Initializes the Qt application context and launches the SpectraGuard GUI.
    """
    # Create the core application event loop
    app = QApplication(sys.argv)
    
    # Apply a dark theme base style to the application globally
    app.setStyle("Fusion")
    
    # Initialize and display the main interface window
    window = MainWindow()
    window.show()
    
    # Execute the application loop and ensure clean exit codes
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
