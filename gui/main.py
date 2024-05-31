from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QLineEdit, QLabel
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import Qt
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Process Frequency Sequence")
        self.setGeometry(100, 100, 1400, 600)

        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Control panel layout
        control_layout = QVBoxLayout()

        # Load file button and display
        self.load_button = QPushButton("Load Excel")
        self.load_button.clicked.connect(self.load_excel)
        self.file_path_display = QLineEdit()
        self.file_path_display.setReadOnly(True)

        # Start line selector
        self.start_line_selector = QLineEdit()  # Use a QSpinBox for numerical input if needed
        self.start_line_selector.setPlaceholderText("Enter start line")

        # Run button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_processing)

        # Adding widgets to control layout
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.file_path_display)
        control_layout.addWidget(self.start_line_selector)
        control_layout.addWidget(self.run_button)

        # Result displays
        self.trial_index_display = QLineEdit("Trial Index")
        self.start_frame_display = QLineEdit("Start Frame")
        self.end_frame_display = QLineEdit("End Frame")
        control_layout.addWidget(self.trial_index_display)
        control_layout.addWidget(self.start_frame_display)
        control_layout.addWidget(self.end_frame_display)

        # Next button
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.confirm_next)

        control_layout.addWidget(self.next_button)

        # Rate map area (Placeholder for your rate map display widget)
        self.rate_map_display = QLabel("Rate Map Here")
        self.rate_map_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rate_map_display.setStyleSheet("background-color: lightgray")

        # Main layout configuration
        self.main_layout.addLayout(control_layout, 1)
        self.main_layout.addWidget(self.rate_map_display, 2)

    def load_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Excel File", "", "Excel Files (*.xlsx)")
        if file_path:
            self.file_path_display.setText(file_path)
            # Load your Excel file processing logic here

    def run_processing(self):
        # Add your processing code here
        print("Processing started...")

    def confirm_next(self):
        reply = QMessageBox.question(self, 'Confirm Next', 'Finish processing this line?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            print("Processing next line...")

app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec())
