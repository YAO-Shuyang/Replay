from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QLineEdit, QLabel
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QSpinBox, QCheckBox
from PyQt6.QtCore import Qt
import sys

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy as cp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from replay.gui.behav_func import run_section_one, run_section_two, save_trace
from replay.preprocess.frequency import correct_freq, display_spectrum, reset_freq
from replay.preprocess.frequency import get_background_noise
from replay.gui.arrowspinbox import ArrowSpinBox

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.f_data = None
        self.curr_trace = None
        self.curr_trial = None
        self._frames = None
        self._endfreq = None
        self._is_iter = True
        self._process_line = 0
        self._process_trial = 0
        self._init_line = 0

        self._modify_frame_gate = False

        self.setWindowTitle("Process Frequency Sequence")
        self.setGeometry(100, 100, 1400, 600)

        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()

        # Load file button and display
        loadLayout = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_excel)
        self.file_path_display = QLineEdit()
        self.file_path_display.setReadOnly(True)

        loadLayout.addWidget(self.load_button, 1)
        loadLayout.addWidget(self.file_path_display, 4)
        self.leftLayout.addLayout(loadLayout)

        # Start line selector
        line_notice = QLabel("Start Line:")
        self.start_line_selector = QSpinBox()
        self.start_line_selector.setValue(0)
        self.start_line_selector.setEnabled(False)
        self.start_line_selector.valueChanged.connect(self.set_process_line)
        # Run button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.init_processing)
        self.run_button.clicked.connect(self.run_processing)
        self.run_button.setEnabled(False)
        # Check if iterating all lines of the excel
        self.checkBox = QCheckBox("Iterate All Lines")
        self.checkBox.setEnabled(False)
        self.checkBox.stateChanged.connect(self.set_iter)
        self.checkBox.setCheckState(Qt.CheckState.Checked)

        runLayout = QHBoxLayout()
        runLayout.addWidget(line_notice, 1)
        runLayout.addWidget(self.start_line_selector, 2)
        runLayout.addWidget(self.run_button, 1)
        runLayout.addWidget(self.checkBox, 1)
        self.leftLayout.addLayout(runLayout)
        
        # Result displays
        self.trial_id_text = QLabel("Trial ID")
        self.onset_text = QLabel("Onset Frame")
        self.end_text = QLabel("End Frame")


        # Trial info display
        self.trial_id = ArrowSpinBox()
        self.trial_id.setValue(0)
        self.trial_id.setEnabled(False)
        self.trial_id.valueChanged.connect(self.switch_trial)
        self.onset_frame = QSpinBox()
        self.onset_frame.setValue(0)
        self.onset_frame.setEnabled(False)
        self.onset_frame.valueChanged.connect(self.update_trial_info)
        self.end_frame = QSpinBox()
        self.end_frame.setValue(0)
        self.end_frame.setEnabled(False)
        self.end_frame.valueChanged.connect(self.update_trial_info)
        trialLayout = QHBoxLayout()
        trialLayout.addWidget(self.trial_id_text)
        trialLayout.addWidget(self.trial_id)
        trialLayout.addWidget(self.onset_text)
        trialLayout.addWidget(self.onset_frame)
        trialLayout.addWidget(self.end_text)
        trialLayout.addWidget(self.end_frame)
        self.leftLayout.addLayout(trialLayout)

        # Next button
        self.insert_button = QPushButton("Insert")
        self.insert_button.clicked.connect(self.insert_trial)
        self.insert_button.setEnabled(False)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_trial)
        self.delete_button.setEnabled(False)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.confirm_next)
        self.next_button.setEnabled(False)
        modifyLayout = QHBoxLayout()
        modifyLayout.addWidget(self.insert_button, 1)
        modifyLayout.addWidget(self.delete_button, 1)
        modifyLayout.addWidget(self.next_button, 1)
        self.leftLayout.addLayout(modifyLayout)

        # Insert Line
        insert_start_label = QLabel("Insert Onset Frame")
        self.insert_start_selector = QSpinBox()
        self.insert_start_selector.setValue(0)
        self.insert_start_selector.setEnabled(False)
        self.insert_start_selector.valueChanged.connect(self.update_insert_spectrum)
        insert_end_label = QLabel("Insert End Frame")
        self.insert_end_selector = QSpinBox()
        self.insert_end_selector.setValue(0)
        self.insert_end_selector.setEnabled(False)
        self.insert_end_selector.valueChanged.connect(self.update_insert_spectrum) 
        insertLayout = QHBoxLayout()
        insertLayout.addWidget(insert_start_label, 2)
        insertLayout.addWidget(self.insert_start_selector, 1)
        insertLayout.addWidget(insert_end_label, 2)
        insertLayout.addWidget(self.insert_end_selector, 1)
        self.leftLayout.addLayout(insertLayout)

        # Separate Line
        Line = QLabel(
            "-----------------------------------------------------------------"
            "----------------------------------"
        )
        self.leftLayout.addWidget(Line)

        noise_start_label = QLabel("Noise Start")
        self.noise_start_selector = QSpinBox()
        self.noise_start_selector.setValue(0)
        self.noise_start_selector.setEnabled(False)
        self.noise_start_selector.valueChanged.connect(self.update_noise_spectrum)
        noise_end_label = QLabel("Noise End")
        self.noise_end_selector = QSpinBox()
        self.noise_end_selector.setValue(0)
        self.noise_end_selector.setEnabled(False)
        self.noise_end_selector.valueChanged.connect(self.update_noise_spectrum)
        self.set_noise_range = QPushButton("Set")
        self.set_noise_range.clicked.connect(self.process_noise)
        self.set_noise_range.setEnabled(False)
        noiseLayout = QHBoxLayout()
        noiseLayout.addWidget(noise_start_label)
        noiseLayout.addWidget(self.noise_start_selector)
        noiseLayout.addWidget(noise_end_label)
        noiseLayout.addWidget(self.noise_end_selector)
        noiseLayout.addWidget(self.set_noise_range)


        # Spectrum area (Placeholder for your spectrum display widget)
        self.right_fig = plt.figure(figsize=(5, 3))
        self.right_ax = plt.axes()
        self.trial_spectrum = FigureCanvas(figure=self.right_fig)
        self.rightLayout.addWidget(self.trial_spectrum)

        self.left_fig = plt.figure(figsize=(5, 3))
        self.left_ax = plt.axes()
        self.browse_spectrum = FigureCanvas(figure=self.left_fig)
        
        leftbottomLayout = QVBoxLayout()
        leftbottomLayout.addLayout(noiseLayout, 1)
        leftbottomLayout.addWidget(self.browse_spectrum, 5)
        
        # Main widget and layout
        leftColumn = QVBoxLayout()
        leftColumn.addLayout(self.leftLayout, 1)
        leftColumn.addLayout(leftbottomLayout, 3)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        self.main_layout.addLayout(leftColumn, 1)
        self.main_layout.addLayout(self.rightLayout, 2)

    def load_excel(self):
        self.reset_varaibles()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Excel File", 
            "", 
            "Excel Files (*.xlsx)"
        )
        if file_path:
            self.file_path_display.setText(file_path)
            self.f_data = pd.read_excel(file_path)
            self.start_line_selector.setRange(0, len(self.f_data)-1)
            self.start_line_selector.setEnabled(True)
            self.checkBox.setEnabled(True)
            self.run_button.setEnabled(True)

            self.f_data_savedir = os.path.join(
                os.path.dirname(file_path),
                "output.xlsx"
            )

    def set_process_line(self):
        """Iterate over all lines of the excel, selecting the start line"""
        self._init_line = self.start_line_selector.value()
        self._process_line = self._init_line

    def set_iter(self):
        """set whether to iterate over all lines of the excel"""
        self._is_iter = self.checkBox.isChecked()

    def init_processing(self):
        if self.f_data is None:
            return
        
        self.start_line_selector.setEnabled(False)
        self.run_button.setEnabled(False)
        self.checkBox.setEnabled(False)

        if self._is_iter:
            self._line_range = (self._init_line, len(self.f_data))
        else:
            self._line_range = (self._init_line, self._init_line+1)

    def reset_varaibles(self):
        self.curr_trace = None
        self.curr_trial = None
        self._process_trial = 0
        self._enter_noise_process = False
        self._frames = None
        self._endfreq = None
        self._modify_frame_gate = False

        self.trial_id.setEnabled(False)
        self.onset_frame.setEnabled(False)
        self.end_frame.setEnabled(False)

        self.noise_start_selector.setEnabled(False)
        self.noise_end_selector.setEnabled(False)
        self.set_noise_range.setEnabled(False)

        self.insert_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.next_button.setEnabled(False)

        self.left_ax.clear()
        self.right_ax.clear()

        self.insert_start_selector.setEnabled(False)
        self.insert_end_selector.setEnabled(False)

    def run_processing(self):
        if self.f_data is None:
            return
        
        if self._process_line <= self._line_range[1]:
            self.curr_trace = run_section_one(self.f_data, self._process_line)

            # noise process
            self._enter_noise_process = True
            self.noise_start_selector.setEnabled(True)
            self.noise_start_selector.setRange(
                0, self.curr_trace['magnitudes'].shape[1]-100
            )
            self.noise_end_selector.setEnabled(True)
            self.noise_end_selector.setRange(
                100, self.curr_trace['magnitudes'].shape[1]-1
            )
            self.noise_end_selector.setValue(100)
            self.set_noise_range.setEnabled(True)

            self.left_ax.clear()
            self.left_ax = display_spectrum(
                self.left_ax,
                magnitudes=self.curr_trace['magnitudes'],
                freq_range=(0, 257),
                frame_range=(0, self.curr_trace['magnitudes'].shape[1]-1)
            )
            self.browse_spectrum.draw()
        else:
            QMessageBox.information(self, "Info", "All sessions have been processed.")

    def update_noise_spectrum(self):
        
        if self.noise_start_selector.value() + 100 > self.noise_end_selector.value():
            return

        lef, rig = self.noise_start_selector.value(), self.noise_end_selector.value()
        self.left_ax.clear()
        self.left_ax = display_spectrum(
            self.left_ax,
            magnitudes=self.curr_trace['magnitudes'],
            freq_range=(0, 257),
            frame_range=(lef - 100, rig + 100)
        )
        self.left_ax.axvline(lef, color = 'blue')
        self.left_ax.axvline(rig, color = 'blue')
        self.browse_spectrum.draw()

        self.f_data.loc[self._process_line, 'noise start'] = lef
        self.f_data.loc[self._process_line, 'noise end'] = rig

        self.f_data.to_excel(self.f_data_savedir, index=False)

    def update_insert_spectrum(self):
        if self.insert_start_selector.value() >= self.insert_end_selector.value():
            return

        lef, rig = self.insert_start_selector.value(), self.insert_end_selector.value()
        
        self.left_ax.clear()
        self.left_ax = display_spectrum(
            self.left_ax,
            magnitudes=self.curr_trace['magnitudes'],
            freq_range=(20, 180),
            frame_range=(lef-10, rig+10),
            dominant_freq=self.curr_trace['dominant_freq_filtered']
        )
        self.left_ax.axvline(lef, color = 'blue')
        self.left_ax.axvline(rig, color = 'blue')
        self.left_ax.axhline(23, color = 'white', linewidth = 0.3)
        self.left_ax.axhline(174, color = 'white', linewidth = 0.3)
        self.browse_spectrum.draw()


    def update_spectrum(self):
        if self.end_frame.value() < self.onset_frame.value():
            return

        self.right_ax.clear()
        onset, end = self.onset_frame.value(), self.end_frame.value()
        self.right_ax = display_spectrum(
            self.right_ax,
            magnitudes=self.curr_trace['magnitudes'],
            freq_range=(20, 180),
            frame_range=(onset - 10, end + 10),
            dominant_freq=self._dominant_freq
        )
        self.right_ax.axhline(23, color = 'white', linewidth = 0.3)
        self.right_ax.axhline(174, color = 'white', linewidth = 0.3)
        self.right_ax.axvline(onset, color = 'blue')
        self.right_ax.axvline(end, color = 'blue')
        self.trial_spectrum.draw()

    def process_noise(self):
        if self.noise_end_selector.value() < self.noise_start_selector.value():
            QMessageBox.warning(
                self, 
                "Warning", 
                "End frame is smaller than start frame! Select again."
            )
        elif self.noise_end_selector.value() < self.noise_start_selector.value() + 100:
            QMessageBox.warning(
            self, 
                "Warning", 
                "Noise range is too small! 100 frames is the minimum. Select again."
            )
        else:
            lef, rig = self.noise_start_selector.value(), self.noise_end_selector.value()
            self.noise_start_selector.setEnabled(False)
            self.noise_end_selector.setEnabled(False)
            self.set_noise_range.setEnabled(False)

            self.f_data.loc[self._process_line, 'noise start'] = lef
            self.f_data.loc[self._process_line, 'noise end'] = rig

            self.f_data.to_excel(self.f_data_savedir, index=False)

            self.curr_trace['background_noise'] = get_background_noise(
                self.curr_trace['magnitudes'],
                (lef, rig)
            )

            self.curr_trace = run_section_two(self.curr_trace)
            _limits = len(self.curr_trace['dominant_freq_filtered'])-1
            self.trial_id.setEnabled(True)
            self.onset_frame.setEnabled(True)
            self.onset_frame.setRange(0, _limits)
            self.end_frame.setEnabled(True)
            self.end_frame.setRange(0, _limits)

            self.insert_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            self.next_button.setEnabled(True)
            self.insert_start_selector.setEnabled(True)
            self.insert_end_selector.setEnabled(True)

            self._frames = np.vstack(
                [self.curr_trace['onset_frames'], self.curr_trace['end_frames']]
            ).astype(np.int64).T

            self._endfreq = cp.deepcopy(self.curr_trace['end_freq'])

            self.trial_id.setValue(0)
            self.trial_id.setRange(0, self._frames.shape[0]-1)
            self.onset_frame.setValue(self._frames[0, 0])
            self.end_frame.setValue(self._frames[0, 1])

            self.insert_start_selector.setRange(0, _limits)
            self.insert_start_selector.setValue(0)
            self.insert_end_selector.setRange(0, _limits)
            self.insert_end_selector.setValue(self._frames[0, 0])

            self._dominant_freq = cp.deepcopy(self.curr_trace['dominant_freq_filtered'])

            self.update_insert_spectrum()
            self.update_spectrum()
            self._modify_frame_gate = True


    def switch_trial(self):
        if self._modify_frame_gate == False:
            return
        _id = self.trial_id.value()
        self._modify_frame_gate = False

        self.onset_frame.setValue(self._frames[_id, 0])
        self.end_frame.setValue(self._frames[_id, 1])

        if _id == 0:
            self.insert_start_selector.setValue(0)
            self.insert_end_selector.setValue(self._frames[0, 0])
        else:
            self.insert_start_selector.setValue(self._frames[_id-1, 1])
            self.insert_end_selector.setValue(self._frames[_id, 0])

        self._modify_frame_gate = True

        self.update_insert_spectrum()
        self.update_spectrum()

    def update_trial_info(self):
        if self._modify_frame_gate == False:
            return
        _id = self.trial_id.value()
        self._frames[_id, 0] = self.onset_frame.value()
        self._frames[_id, 1] = self.end_frame.value()
        self._dominant_freq = correct_freq(
            cp.deepcopy(self.curr_trace['dominant_freq_filtered']),
            self.curr_trace['magnitudes'],
            self._frames[_id, 0],
            self._frames[_id, 1]
        )
        self.update_spectrum()

    def insert_trial(self):
        lef, rig = self.insert_start_selector.value(), self.insert_end_selector.value()
        if lef >= rig:
            QMessageBox.warning(
                self, 
                "Warning", 
                "End frame is smaller than start frame! Select again."
            )
            return
        
        idx1 = np.where(self._frames[:, 1] <= lef)[0]
        idx2 = np.where(self._frames[:, 0] <= rig)[0]

        if len(idx1) != len(idx2):
            QMessageBox.warning(
                self, 
                "Warning", 
                "Boundry of the new trial contains onset or end frame of pre-existing trials! Select again."
            )
            return
        
        idx = len(idx1)
        self.curr_trace['dominant_freq_filtered'] = correct_freq(
            self.curr_trace['dominant_freq_filtered'],
            self.curr_trace['magnitudes'],
            lef,
            rig
        )
        self._frames = np.insert(self._frames, idx, (lef, rig), axis = 0)
        self._endfreq = np.insert(
            self._endfreq, 
            idx, 
            self.curr_trace['dominant_freq_filtered'][rig]
        )
        self.trial_id.setValue(idx)
        self.switch_trial()

    def delete_trial(self):
        _id = self.trial_id.value()
        self.curr_trace['dominant_freq_filtered'] = reset_freq(
            self.curr_trace['dominant_freq_filtered'],
            self._frames[_id, 0],
            self._frames[_id, 1]
        )
        self._frames = np.delete(self._frames, _id, axis = 0)
        self._endfreq = np.delete(self._endfreq, _id)
        self.switch_trial()

    def confirm_next(self):
        reply = QMessageBox.question(self, 'Confirm Next', 'Finish processing this line?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            print("Processing next line...")
            self._process_line += 1

            self.save()
            self.reset_varaibles()
            self.run_processing()

    def save(self):
        for i in range(len(self._frames)):
            self.curr_trace['dominant_freq_filtered'] = correct_freq(
                self.curr_trace['dominant_freq_filtered'],
                self.curr_trace['magnitudes'],
                self._frames[i, 0],
                self._frames[i, 1]
            )

        self.curr_trace['onset_frames'] = self._frames[:, 0]
        self.curr_trace['end_frames'] = self._frames[:, 1]
        self.curr_trace['end_freq'] = self._endfreq
        self.f_data.to_excel(self.f_data_savedir, index=False)
        save_trace(self.curr_trace)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
