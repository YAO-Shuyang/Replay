import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox
from PyQt6.QtWidgets import QPushButton, QMessageBox
from PyQt6.QtCore import pyqtSignal

class MagnitudesCorrector(QWidget):
    rateMapUpdated = pyqtSignal(np.ndarray)
    
    def __init__(
        self, 
        magnitudes: np.ndarray,
        left: int = 0,
        right: int | None = None,
        bottom: int = 20,
        top: int = 180
    ):
        if right is None:
            right = magnitudes.shape[0] - 1
            
        super().__init__()
         
        self.magnitudes = magnitudes

        self.layouts = QVBoxLayout()
        self.spinBoxLayout = QHBoxLayout()

        # Labels and SpinBoxes for boundaries
        self.labels = ['Left:', 'Right:', 'Bottom:', 'Top:']
        defaults = [left, right, bottom, top]
        print(magnitudes.shape)
        ranges = [
            magnitudes.shape[1]-1, 
            magnitudes.shape[1]-1, 
            magnitudes.shape[0]-1, 
            magnitudes.shape[0]-1
        ]
        self.spinBoxes = []
        for i, label in enumerate(self.labels):
            lbl = QLabel(label, self)
            spb = QSpinBox(self)
            spb.setRange(0, ranges[i])  # Example range, adjust as needed
            spb.setValue(defaults[i])
            spb.valueChanged.connect(self.check_values)
            spb.valueChanged.connect(self.update_plot)
            self.spinBoxLayout.addWidget(lbl)
            self.spinBoxLayout.addWidget(spb)
            self.spinBoxes.append(spb)
            
        self.setter_max = QPushButton("Set Max")
        self.setter_max.clicked.connect(self.set_maximum)
        self.setter_min = QPushButton("Set Min")
        self.setter_min.clicked.connect(self.set_minimum)
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.setter_max)
        self.buttons_layout.addWidget(self.setter_min)

        self.layouts.addLayout(self.spinBoxLayout)
        self.layouts.addLayout(self.buttons_layout)

        # Matplotlib Canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layouts.addWidget(self.canvas)
        
        self.temp_objects = []

        self.update_magnitudes()
        self.setLayout(self.layouts)

    def check_values(self):
        if self.spinBoxes[0].value() >= self.spinBoxes[1].value():
            self.spinBoxes[0].setValue(self.spinBoxes[1].value() - 1)
        if self.spinBoxes[2].value() >= self.spinBoxes[3].value():
            self.spinBoxes[2].setValue(self.spinBoxes[3].value() - 1)

    def set_maximum(self):
        self.magnitudes[
            self.spinBoxes[2].value():self.spinBoxes[3].value(),
            self.spinBoxes[0].value():self.spinBoxes[1].value()
        ] = 1.01
        self.update_magnitudes()
        self.rateMapUpdated.emit(self.magnitudes)
        
    def set_minimum(self):
        self.magnitudes[
            self.spinBoxes[2].value():self.spinBoxes[3].value(),
            self.spinBoxes[0].value():self.spinBoxes[1].value()
        ] = 0
        self.update_magnitudes()
        self.rateMapUpdated.emit(self.magnitudes)

    def update_magnitudes(self):
        self.ax.clear()
        self.ax.imshow(
            self.magnitudes, 
            aspect="auto", 
            cmap = "hot", 
            interpolation='nearest'
        )
        self.update_plot()

    def update_plot(self):
        if (self.spinBoxes[0].value() < self.spinBoxes[1].value() and
            self.spinBoxes[2].value() < self.spinBoxes[3].value()):
            
            left = self.spinBoxes[0].value() - 5
            right = self.spinBoxes[1].value() + 5
            bottom = self.spinBoxes[2].value() - 5
            top = self.spinBoxes[3].value() + 5
            
            for a in self.temp_objects:
                a.remove()
                
            a = self.ax.axvline(left + 5, color = 'gray', linewidth = 0.3)
            b = self.ax.axvline(right - 5, color = 'gray', linewidth = 0.3)
            c = self.ax.axhline(top - 5, color = 'gray', linewidth = 0.3)
            d = self.ax.axhline(bottom + 5, color = 'gray', linewidth = 0.3)
            self.temp_objects = [a, b, c, d]
            
            self.ax.axis([left, right, bottom, top])
            self.ax.figure.canvas.draw()

        else:
            QMessageBox.warning(
                self, "Warning", "Please select valid boundaries."
            )