# -*- coding: big5 -*-
"""
Vertical Datum Transformation - PyQt5 version (accelerated)
Version : 3.0.0
Requirements:
  - Python 3.9+
  - numpy, scipy, pillow, PyQt5
Folder layout:
  ./vertical_datum_qt.py
  ./file/
      MSS.xyz, HAT.xyz, MHW.xyz, MLW.xyz, LAT.xyz, ISLW.xyz, geoid.xyz
      fig1.png
"""

import os
import sys
import io
import numpy as np
from functools import lru_cache

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QMessageBox
)

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


# -----------------------------
# Config & constants
# -----------------------------
GUI_TITLE = "Vertical Datum Transformation (PyQt5)"
GUI_SIZE = (840, 760)

Surface = [
    "Mean Sea Surface (MSS)",
    "Highest Astronomical Tide (HAT)",
    "Mean High Water (MHW)",
    "Mean Low Water (MLW)",
    "Lowest Astronomical Tide (LAT)",
    "Indian Spring Low Water (ISLW)",
    "Geoid",
    "Ellipsoid"
]
Surface_nickname = ["MSS", "HAT", "MHW", "MLW", "LAT", "ISLW", "Geoid", "EL"]

# 外部檔案路徑（請放在專案根目錄的 file/ 資料夾）
Surface_file = [
    "file/MSS.xyz",
    "file/HAT.xyz",
    "file/MHW.xyz",
    "file/MLW.xyz",
    "file/LAT.xyz",
    "file/ISLW.xyz",
    "file/geoid.xyz"
]
FIG_PATH = "file/fig1.png"

# 有效範圍（台灣近海）：118–125E, 21–27N
LON_MIN, LON_MAX = 118.0, 125.0
LAT_MIN, LAT_MAX = 21.0, 27.0


# -----------------------------
# IO helpers
# -----------------------------
def read_llv_from_file(filename: str):
    lon, lat, val = [], [], []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().split()
            if len(s) < 3:
                continue
            try:
                lo = float(s[0]); la = float(s[1]); v = float(s[2])
                lon.append(lo); lat.append(la); val.append(v)
            except ValueError:
                continue
    return np.asarray(lon), np.asarray(lat), np.asarray(val)

def write_llvn(filename: str, lon, lat, value1, value2):
    with open(filename, "w", encoding="utf-8") as g:
        for i in range(len(lon)):
            lo = float(lon[i]); la = float(lat[i])
            v1 = value1[i]; v2 = value2[i]
            s_v1 = f"{v1:8.3f}" if np.isfinite(v1) else "      NaN"
            s_v2 = f"{v2:8.3f}" if np.isfinite(v2) else "      NaN"
            g.write(f"{lo:11.7f} {la:10.7f} {s_v1} {s_v2}\n")

def check_range(lon, lat, value):
    lon = np.asarray(lon); lat = np.asarray(lat); value = np.asarray(value)
    mask = (lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX)
    idx_in = np.where(mask)[0]
    idx_out = np.where(~mask)[0]
    return lon[idx_in], lat[idx_in], value[idx_in], idx_out


# -----------------------------
# Accelerated interpolators with cache
# -----------------------------
@lru_cache(maxsize=None)
def load_surface_points(surface_idx: int):
    """讀取點雲並快取。"""
    if surface_idx == 7:
        return np.empty((0, 2)), np.empty((0,))
    path = Surface_file[surface_idx]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing surface file: {path}")
    lon, lat, val = read_llv_from_file(path)
    P = np.column_stack([lon, lat]).astype(np.float64)
    V = val.astype(np.float64)
    return P, V

_interpolators_linear = {}
_interpolators_nearest = {}

def get_linear_interp(surface_idx: int):
    if surface_idx == 7:
        return None
    if surface_idx not in _interpolators_linear:
        P, V = load_surface_points(surface_idx)
        _interpolators_linear[surface_idx] = LinearNDInterpolator(P, V)
    return _interpolators_linear[surface_idx]

def get_nearest_interp(surface_idx: int):
    if surface_idx == 7:
        return None
    if surface_idx not in _interpolators_nearest:
        P, V = load_surface_points(surface_idx)
        _interpolators_nearest[surface_idx] = NearestNDInterpolator(P, V)
    return _interpolators_nearest[surface_idx]

def interp_surface_with_fallback(surface_idx: int, XY: np.ndarray) -> np.ndarray:
    """線性插值，NaN 以最近鄰回補。Ellipsoid 回 0。"""
    if surface_idx == 7:
        return np.zeros(XY.shape[0], dtype=np.float64)
    lin = get_linear_interp(surface_idx)
    z = lin(XY[:, 0], XY[:, 1])
    if np.isscalar(z):
        z = np.asarray([z], dtype=np.float64)
    if np.isnan(z).any():
        nn = get_nearest_interp(surface_idx)
        z_nn = nn(XY[:, 0], XY[:, 1])
        z = np.where(np.isnan(z), z_nn, z)
    return z.astype(np.float64)


# -----------------------------
# Core transform
# -----------------------------
def transform_values(input_surface_idx: int,
                     output_surface_idx: int,
                     lon: np.ndarray,
                     lat: np.ndarray,
                     values: np.ndarray,
                     input_value_type: str):
    """
    input_value_type:
        'DEPTH'    深度（向下為正）
        'ELLI_BED' 海床橢球高（向上為正）
    假設：所有 surface 的值皆為對橢球高程 H_surface(λ,φ)（向上為正）
    """
    XY = np.column_stack([lon.astype(np.float64), lat.astype(np.float64)])
    H_in  = interp_surface_with_fallback(input_surface_idx,  XY)
    H_out = interp_surface_with_fallback(output_surface_idx, XY)

    if input_value_type == "DEPTH":
        # 純位移：d_out = d_in + (H_out - H_in)
        new_vals = values + (H_out - H_in)
    elif input_value_type == "ELLI_BED":
        # d_out = H_out - h_bed
        new_vals = H_out - values
    else:
        raise ValueError("Unknown input_value_type")
    return new_vals, H_in, H_out


# -----------------------------
# Worker for file transform
# -----------------------------
class FileTransformWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, input_surface_idx, output_surface_idx,
                 input_path, output_dir, output_name, input_value_type):
        super().__init__()
        self.input_surface_idx = input_surface_idx
        self.output_surface_idx = output_surface_idx
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_name = output_name
        self.input_value_type = input_value_type

    def run(self):
        try:
            lon0, lat0, val0 = [], [], []
            with open(self.input_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip().split()
                    if len(s) < 3: continue
                    try:
                        lo = float(s[0]); la = float(s[1]); v = float(s[2])
                        lon0.append(lo); lat0.append(la); val0.append(v)
                    except ValueError:
                        continue
            lon0 = np.asarray(lon0); lat0 = np.asarray(lat0); val0 = np.asarray(val0)
            if lon0.size == 0:
                self.finished.emit(False, "Input file is empty or invalid.")
                return

            lon_in, lat_in, val_in, idx_out = check_range(lon0, lat0, val0)
            if lon_in.size > 0:
                new_in, _, _ = transform_values(
                    self.input_surface_idx, self.output_surface_idx,
                    lon_in, lat_in, val_in, self.input_value_type
                )
                # 遮罩回填到原長度
                new_full = np.full(lon0.shape[0], np.nan, dtype=np.float64)
                mask = np.ones(lon0.shape[0], dtype=bool)
                if idx_out.size > 0:
                    mask[idx_out] = False
                new_full[mask] = new_in
            else:
                new_full = np.full(lon0.shape[0], np.nan, dtype=np.float64)

            os.makedirs(self.output_dir, exist_ok=True)
            out_path = os.path.join(self.output_dir, self.output_name)
            write_llvn(out_path, lon0, lat0, val0, new_full)

            msg = "OK!"
            if idx_out.size == 1:
                msg += "  (There is 1 point outside the range)"
            elif idx_out.size > 1:
                msg += f"  (There are {idx_out.size} points outside the range)"
            self.finished.emit(True, msg)
        except Exception as e:
            self.finished.emit(False, f"Error: {e}")


# -----------------------------
# Main window (Single Page)
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(GUI_TITLE)
        self.resize(*GUI_SIZE)

        # Default selection
        self.input_surface_idx = 0
        self.output_surface_idx = 1

        # Central widget
        root = QWidget()
        self.setCentralWidget(root)
        layout = QGridLayout(root)

        # A) 下拉選單（在 Single point 之上）
        lbl_sel_title = QLabel("Surfaces")
        lbl_sel_title.setStyleSheet("font-weight:600;")
        layout.addWidget(lbl_sel_title, 0, 0, Qt.AlignLeft)

        layout.addWidget(QLabel("Input Surface"), 1, 0, Qt.AlignLeft)
        self.cmb_in_surface = QComboBox()
        self.cmb_in_surface.addItems(Surface)
        self.cmb_in_surface.setCurrentIndex(self.input_surface_idx)
        self.cmb_in_surface.currentIndexChanged.connect(self.on_in_surface_changed)
        layout.addWidget(self.cmb_in_surface, 1, 1, 1, 2)

        layout.addWidget(QLabel("Output Surface"), 1, 3, Qt.AlignLeft)
        self.cmb_out_surface = QComboBox()
        self.cmb_out_surface.addItems(Surface)
        self.cmb_out_surface.setCurrentIndex(self.output_surface_idx)
        self.cmb_out_surface.currentIndexChanged.connect(self.on_out_surface_changed)
        layout.addWidget(self.cmb_out_surface, 1, 4, 1, 2)

        # B) Range 區塊（小圖示意）
        lbl_range_title = QLabel("Range")
        lbl_range_title.setStyleSheet("font-weight:600;")
        layout.addWidget(lbl_range_title, 3, 0, Qt.AlignLeft)

        lbl_range = QLabel(
            "     ---------  27 N ---------\n"
            "    |                                 |\n"
            "    118 E                           125 E\n"
            "    |                                 |\n"
            "     ---------  21 N ---------\n"
        )
        layout.addWidget(lbl_range, 4, 0, 1, 3, Qt.AlignLeft)

        # C) Single point
        lbl_single = QLabel("Single point")
        lbl_single.setStyleSheet("font-weight:600;")
        layout.addWidget(lbl_single, 6, 0, Qt.AlignLeft)

        layout.addWidget(QLabel("Longitude"), 7, 0)
        layout.addWidget(QLabel("Latitude"), 7, 1)
        layout.addWidget(QLabel("Input value"), 7, 2)

        self.ed_lon = QLineEdit(); self.ed_lon.setPlaceholderText("e.g. 121.5")
        self.ed_lat = QLineEdit(); self.ed_lat.setPlaceholderText("e.g. 24.0")
        self.ed_val = QLineEdit(); self.ed_val.setPlaceholderText("float")

        layout.addWidget(self.ed_lon, 8, 0)
        layout.addWidget(self.ed_lat, 8, 1)
        layout.addWidget(self.ed_val, 8, 2)

        layout.addWidget(QLabel("Input value type"), 7, 3)
        self.cmb_valtype = QComboBox()
        self.cmb_valtype.addItems(["Depth (down +)", "Ellipsoidal bed height (up +)"])
        layout.addWidget(self.cmb_valtype, 8, 3)

        self.lbl_single_out_title = QLabel("New value")
        layout.addWidget(self.lbl_single_out_title, 9, 2)
        self.lbl_single_in_nick = QLabel("(      )")
        self.lbl_single_out_nick = QLabel("(      )")
        layout.addWidget(self.lbl_single_in_nick, 8, 5)
        layout.addWidget(self.lbl_single_out_nick, 10, 5)
        self.update_nick_labels()

        self.lbl_single_out_val = QLabel("—")
        layout.addWidget(self.lbl_single_out_val, 10, 2)

        btn_single = QPushButton("Transform >")
        btn_single.clicked.connect(self.do_single_transform)
        layout.addWidget(btn_single, 9, 3)

        # D) File transform
        lbl_file = QLabel("Import a file")
        lbl_file.setStyleSheet("font-weight:600;")
        layout.addWidget(lbl_file, 12, 0, Qt.AlignLeft)

        layout.addWidget(QLabel("Input file"), 14, 0)
        layout.addWidget(QLabel("Output directory"), 15, 0)
        layout.addWidget(QLabel("Output file"), 16, 1, Qt.AlignRight)

        self.ed_infile = QLineEdit()
        self.ed_outdir = QLineEdit()
        self.ed_outname = QLineEdit(); self.ed_outname.setPlaceholderText("output.xyz")
        layout.addWidget(self.ed_infile, 14, 1, 1, 3)
        layout.addWidget(self.ed_outdir, 15, 1, 1, 3)
        layout.addWidget(self.ed_outname, 16, 2, 1, 2)

        btn_browse_in = QPushButton("Browse")
        btn_browse_out = QPushButton("Browse")
        btn_browse_in.clicked.connect(self.pick_infile)
        btn_browse_out.clicked.connect(self.pick_outdir)
        layout.addWidget(btn_browse_in, 14, 4)
        layout.addWidget(btn_browse_out, 15, 4)

        self.btn_file_transform = QPushButton("Transform")
        self.btn_file_transform.clicked.connect(self.do_file_transform)
        layout.addWidget(self.btn_file_transform, 16, 4)

        self.lbl_file_status = QLabel("")
        layout.addWidget(self.lbl_file_status, 17, 4)

        # E) fig1.png（最底部，但在 Notes 上方）
        self.fig_label = QLabel()
        pix = QPixmap(FIG_PATH) if os.path.exists(FIG_PATH) else QPixmap()
        if not pix.isNull():
            w = 640
            h = int(pix.height() * (w / pix.width()))
            self.fig_label.setPixmap(pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(self.fig_label, 19, 0, 1, 6, Qt.AlignLeft)

        # F) Notes（頁面最後）
        note1 = "註1 : 水深值(垂直基準面至海床垂直距離)坐標軸向下為正"
        note2 = "註2 : 海床橢球高(橢球面至海床垂直距離)坐標軸向上為正"
        note3 = "註3 : Ellipsoid在海域上指的是海床橢球高，在陸域則是代表該點的橢球高"
        note4 = "註4 : 內陸橢球高進行正高轉換時，輸出值為負代表該點在geoid之上，"
        note5 = "         例如: 輸出值為-5公尺代表該點正高為5公尺"
        note6 = "Created by Mingyi Hsu"
        layout.addWidget(QLabel(note1), 21, 0, 1, 6)
        layout.addWidget(QLabel(note2), 22, 0, 1, 6)
        layout.addWidget(QLabel(note3), 23, 0, 1, 6)
        layout.addWidget(QLabel(note4), 24, 0, 1, 6)
        layout.addWidget(QLabel(note5), 25, 0, 1, 6)
        layout.addWidget(QLabel(note6), 26, 0, 1, 6)

    # --- Surface dropdown events ---
    def on_in_surface_changed(self, idx: int):
        self.input_surface_idx = idx
        if self.input_surface_idx == self.output_surface_idx:
            new_out = (self.output_surface_idx + 1) % len(Surface)
            self.cmb_out_surface.setCurrentIndex(new_out)
            self.output_surface_idx = new_out
        self.update_nick_labels()

    def on_out_surface_changed(self, idx: int):
        self.output_surface_idx = idx
        if self.input_surface_idx == self.output_surface_idx:
            new_in = (self.input_surface_idx + 1) % len(Surface)
            self.cmb_in_surface.setCurrentIndex(new_in)
            self.input_surface_idx = new_in
        self.update_nick_labels()

    def update_nick_labels(self):
        self.lbl_single_in_nick.setText(f"({Surface_nickname[self.input_surface_idx]:^7})")
        self.lbl_single_out_nick.setText(f"({Surface_nickname[self.output_surface_idx]:^7})")

    # --- Single transform ---
    def do_single_transform(self):
        lon_txt = self.ed_lon.text().strip()
        lat_txt = self.ed_lat.text().strip()
        val_txt = self.ed_val.text().strip()
        if not lon_txt or not lat_txt or not val_txt:
            QMessageBox.information(self, "Error", "Please fill in Longitude, Latitude, and Input value.")
            return
        try:
            lon = float(lon_txt); lat = float(lat_txt); val = float(val_txt)
        except ValueError:
            QMessageBox.information(self, "Error", "Please type in float format.")
            return
        if not (LON_MIN <= lon <= LON_MAX and LAT_MIN <= lat <= LAT_MAX):
            QMessageBox.information(
                self, "Error",
                f"The point must be inside the range:\n {LON_MIN}~{LON_MAX}E, {LAT_MIN}~{LAT_MAX}N."
            )
            return

        input_type = "DEPTH" if self.cmb_valtype.currentIndex() == 0 else "ELLI_BED"
        new_vals, _, _ = transform_values(
            self.input_surface_idx, self.output_surface_idx,
            np.array([lon]), np.array([lat]), np.array([val]), input_type
        )
        self.lbl_single_out_val.setText(f"{new_vals[0]:.4f}")
        self.update_nick_labels()

    # --- File transform ---
    def pick_infile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a file", os.getcwd(), "XYZ files (*.xyz);;All files (*.*)")
        if path:
            self.ed_infile.setText(path)

    def pick_outdir(self):
        path = QFileDialog.getExistingDirectory(self, "Select a directory", os.getcwd())
        if path:
            self.ed_outdir.setText(path)

    def do_file_transform(self):
        in_path = self.ed_infile.text().strip()
        out_dir = self.ed_outdir.text().strip()
        out_name = self.ed_outname.text().strip()
        if not in_path:
            QMessageBox.information(self, "Error", "Please select the input file"); return
        if not out_dir:
            QMessageBox.information(self, "Error", "Please choose a directory for output file"); return
        if not out_name:
            QMessageBox.information(self, "Error", "Please name the output file"); return

        input_type = "DEPTH" if self.cmb_valtype.currentIndex() == 0 else "ELLI_BED"
        self.btn_file_transform.setEnabled(False)
        self.lbl_file_status.setText("Running...")

        self.worker = FileTransformWorker(
            self.input_surface_idx, self.output_surface_idx,
            in_path, out_dir, out_name, input_type
        )
        self.worker.finished.connect(self.on_file_finished)
        self.worker.start()

    def on_file_finished(self, success: bool, message: str):
        self.btn_file_transform.setEnabled(True)
        self.lbl_file_status.setText(message)
        if not success:
            QMessageBox.warning(self, "Transform error", message)


# -----------------------------
# Entry
# -----------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

