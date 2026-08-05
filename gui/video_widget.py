import cv2
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QSize

class VideoWidget(QLabel):
    """
    Custom QLabel tailored to receive, transform, and draw OpenCV image metrics 
    while preserving aspect ratio constraints fluidly during scale updates.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Set a baseline size policy to allow expansion and shrinking
        self.setMinimumSize(QSize(320, 240))
        self._current_frame = None
        
        # Style hint to cleanly show bounds before connection
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")

    def update_frame(self, opencv_frame):
        """Processes and maps a raw OpenCV frame matrix onto the UI window surface."""
        if opencv_frame is None:
            return

        self._current_frame = opencv_frame.copy()
        self._render_current()

    def _render_current(self):
        """Converts the raw BGR matrix array to an updated QPixmap scene component."""
        if self._current_frame is None:
            return

        # Fetch shape metadata dimensions
        h, w, ch = self._current_frame.shape
        bytes_per_line = ch * w

        # Convert OpenCV BGR array pixel alignment format to standard RGB color matching
        rgb_image = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2RGB)

        # Map memory references directly to matching Qt container frameworks
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Generate target viewport allocation size parameters based on label scaling boundaries
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale the pixmap down to fit current boundaries while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Ensures visible frame content rescales gracefully on UI resize operations."""
        super().resizeEvent(event)
        if self._current_frame is not None:
            self._render_current()

    def clear_frame(self):
        """Flushes layout state hooks cleanly back down to empty display basics."""
        self._current_frame = None
        self.clear()
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
