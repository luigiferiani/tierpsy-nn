import sys
import os
import tables
import pandas as pd
import numpy as np
from functools import partial

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt

from image_labeler_ui import Ui_image_labeler

from tierpsy.gui.TrackerViewerAux import TrackerViewerAuxGUI
from tierpsy.gui.HDF5VideoPlayer import LineEditDragDrop, setChildrenFocusPolicy


#FILE_NAME = '/Users/ajaver/OneDrive - Imperial College London/recognize_worms/worm_ROI_samplesI.hdf5'
# FILE_NAME = '/Volumes/rescomp1/data/WormData/experiments/training_rois.hdf5'
# FILE_NAME = '/Users/lferiani/work_repos/tierpsy-nn/data/worm_ROI_samples_20191205.hdf5'
FILE_NAME = ''

class d_Ui_image_labeler(Ui_image_labeler):
    def __init__(self):
        super().__init__()

    def setupUi(self, image_labeler):
        #widgets that are not present in Ui_image_labeler, but are required by TrackerViewerAuxGUI.
        #we just create dummy objects here
        super().setupUi(image_labeler)
        self.playButton = QtWidgets.QPushButton()
        self.pushButton_skel = QtWidgets.QPushButton()
        self.lineEdit_skel = QtWidgets.QLineEdit()
        self.imageSlider = QtWidgets.QSlider()
        self.doubleSpinBox_fps = QtWidgets.QDoubleSpinBox()
        self.spinBox_step = QtWidgets.QDoubleSpinBox()
        self.spinBox_frame = QtWidgets.QDoubleSpinBox()

class image_labeler_GUI(TrackerViewerAuxGUI):

    def __init__(self, mask_file=FILE_NAME):
        super().__init__(d_Ui_image_labeler())

        self.btn_colors = {1:'darkRed', 2:'green', 3:'yellow', 4:'blue', 5:'magenta', 6:'darkCyan'}
        self.buttons = {1:self.ui.pushButton_bad,
                        2:self.ui.pushButton_worm,
                        3:self.ui.pushButton_worm_hard,
                        4:self.ui.pushButton_aggregate,
                        5:self.ui.pushButton_eggs,
                        6:self.ui.pushButton_larvae}

        self._setup_buttons()
        self.ui.spinBox_samp_ord.valueChanged.connect(self.updateSampleNumber)
        self.ui.pushButton_save.clicked.connect(self.saveData)

        self.sample_number = 0
        self.ui.spinBox_samp_ord.setValue(0)
        print(self.ui.comboBox_h5path.itemText(0))


        if os.path.exists(mask_file):
            self.updateVideoFile(mask_file)
            self.mainImage.zoomFitInView()

    def updateVideoFile(self, vfilename):
        try:
            self.fid.close()
        except:
            pass

        with pd.HDFStore(vfilename) as fid:
            self.sample_data = fid['/sample_data']

        if not 'resampled_index' in self.sample_data:
            self.sample_data['resampled_index'] = np.random.permutation(len(self.sample_data))
            # self.sample_data['resampled_index'] = np.arange(len(self.sample_data))
        self.sample_data['img_row_id'] = self.sample_data.index
        self.sample_data.index = self.sample_data['resampled_index'].values

        if not 'label_id' in self.sample_data:
            self.sample_data['label_id'] = 0

        #i want to use the father of TrackerViewerAux_GUI (i do not need the skeleton file updates)
        super().updateVideoFile(vfilename)
        self.ui.spinBox_samp_ord.setMaximum(self.tot_frames - 1)

        #move the the next unanalyzed frame
        self.moveNextNotAnalyzed()

    def moveNextNotAnalyzed(self):
        next_f = self.sample_data.loc[self.sample_data['label_id']==0, 'resampled_index'].min()
        if next_f != next_f:
            next_f = self.sample_data.shape[0]

        # print("setting sample number to {}".format(next_f))
        self.ui.spinBox_samp_ord.setValue(next_f)
        # print(self.ui.spinBox_samp_ord.value())
        self.updateSampleNumber()
    #def updateImGroup(self):
    #    super().updateImGroup()
    #    self.updateSampleNumber()



    def keyPressEvent(self, event):
        if self.fid == -1:
            # break no file open, nothing to do here
            return

        key = event.key()

        for btn_id, btn in self.buttons.items():
            if key == (Qt.Key_0+btn_id):
                btn.toggle()
                return
        # Move backwards when  are pressed
        if key in [Qt.Key_Left, Qt.Key_Less, Qt.Key_Comma]:
            self.ui.spinBox_samp_ord.setValue(self.sample_number - 1)
        # Move forward when  are pressed
        elif key in [Qt.Key_Right, Qt.Key_Greater, Qt.Key_Period]:
            self.ui.spinBox_samp_ord.setValue(self.sample_number + 1)

        elif key == Qt.Key_Minus:
            self.mainImage.zoom(-1)
        elif key == Qt.Key_Plus:
            self.mainImage.zoom(1)
        else:
            QtWidgets.QMainWindow.keyPressEvent(self, event)


    def updateSampleNumber(self):
        # print('updating sample number')
        self.sample_number = self.ui.spinBox_samp_ord.value()
        self.frame_number = int(self.sample_data.loc[self.sample_number, 'img_row_id'])

        self.updateImage()
        label_id = int(self.sample_data.loc[self.sample_number, 'label_id'])
        # print('label_id:{}'.format(label_id))
        if label_id >0:
            self.buttons[label_id].setChecked(True)
        else:
            for btn in self.buttons.values():
                # print('resetting {} which was {}'.format(btn, btn.isChecked()))
                btn.setChecked(False)
                # print("now it's {}".format(btn.isChecked()))
                btn.repaint()


    def updateImage(self):
        self.readCurrentFrame()
        # self.drawSkelSingleWorm()
        self.mainImage.setPixmap(self.frame_qimg)

    def saveData(self, is_gui_to_close=False):
        '''save data from manual labelling. pytables saving format is more convenient than pandas'''

        if os.name == 'nt':
            # I Windows the paths return by QFileDialog use / as the file
            # separation character. We need to correct it.
            self.vfilename = self.vfilename.replace('/', os.sep)


        #close data before reopen in a write mode
        self.fid.close()
        with tables.File(self.vfilename, "r+") as fid:
            # pytables filters.
            table_filters = tables.Filters(
                complevel=5, complib='zlib', shuffle=True, fletcher32=True)

            newT = fid.create_table(
                '/',
                'sample_data_d',
                obj=self.sample_data.to_records(index=False),
                filters=table_filters)

            fid.remove_node('/', 'sample_data')
            newT.rename('sample_data')

        #re-reopen the video file
        if is_gui_to_close:
            return
        else:
            self.updateVideoFile(self.vfilename)



    def _setup_buttons(self):
        stylesheet_str = "QPushButton:checked {border: 2px solid; border-radius: 6px; background-color: %s }"
        def _make_label(label_id, checked):
            if checked:
                for btn_id, btn in self.buttons.items():
                    if btn_id != label_id:
                        btn.setChecked(False)

            if self.fid != -1:
                if checked:
                    #add label
                    self.sample_data.loc[self.sample_number, 'label_id'] = label_id
                else:
                    old_lab = self.sample_data.loc[self.sample_number, 'label_id']
                    if old_lab == label_id:
                        #if the labeld was unchecked remove the label
                        self.sample_data.loc[self.sample_number, 'label_id'] = 0


        for btn_id, btn in self.buttons.items():
            btn.setCheckable(True)
            btn.setStyleSheet(stylesheet_str % self.btn_colors[btn_id])
            btn.toggled.connect(partial(_make_label, btn_id))

    def closeEvent(self, event):
        quit_msg = "Do you want to save the current progress before exiting?"
        reply = QtWidgets.QMessageBox.question(self,
                                               'Message',
                                               quit_msg,
                                               QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.Yes)


        if reply == QtWidgets.QMessageBox.Yes:
            self.saveData(is_gui_to_close=True)

        super().closeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = image_labeler_GUI()
    ui.show()
    sys.exit(app.exec_())