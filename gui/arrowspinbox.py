from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSpinBox
from PyQt6.QtCore import Qt

class ArrowSpinBox(QSpinBox):
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right:
            # Increment the value when the right arrow key is pressed
            self.stepUp()
        elif event.key() == Qt.Key.Key_Left:
            # Decrement the value when the left arrow key is pressed
            self.stepDown()
        else:
            # Handle all other key events as usual
            super().keyPressEvent(event)