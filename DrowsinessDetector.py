import math
import queue
import threading
import time
from collections import deque
try:
    import winsound  # Windows only
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QGridLayout,
    QWidget, QFrame, QSizePolicy,
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QFontMetrics,
    QLinearGradient, QPainterPath,
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, QSize

from slow_system import SlowSystem, SlowSystemConfig
from decision_fusion import DecisionFusion


# =========================================================================
#                   SENTINEL DMS — Brand system
#                   Apple / Tesla inspired dark theme
# =========================================================================
BRAND_NAME = "SENTINEL"
BRAND_SUB = "DRIVER MONITORING SYSTEM"
BRAND_VERSION = "v1.0"

# ---- Apple iOS 13+ dark-mode / Tesla cockpit palette ----
C_BG          = "#000000"   # true OLED black (Tesla + iOS)
C_BG_ALT      = "#0a0a0c"   # subtle lift under the main bg
C_CARD        = "#1c1c1e"   # iOS systemGray6 (dark) — secondary bg
C_CARD_2      = "#2c2c2e"   # iOS systemGray5 (dark) — elevated card
C_BORDER      = "#38383a"   # iOS separatorColor (dark)
C_BORDER_2    = "#48484a"   # iOS systemGray4 — stronger separator

C_TEXT        = "#ffffff"   # pure white primary
C_TEXT_DIM    = "#ebebf5"   # iOS secondary label (alpha via color)
C_TEXT_MUTED  = "#8e8e93"   # iOS systemGray — tertiary label
C_TEXT_FAINT  = "#636366"   # iOS systemGray2 — quaternary label

C_ACCENT      = "#0a84ff"   # iOS system blue (dark mode)
C_ACCENT_2    = "#5ac8fa"   # iOS system teal (dark mode)

C_OK          = "#30d158"   # iOS system green (dark)
C_WARN        = "#ffd60a"   # iOS system yellow (dark)
C_ORANGE      = "#ff9f0a"   # iOS system orange (dark)
C_DANGER      = "#ff453a"   # iOS system red (dark)
C_CRITICAL    = "#ff3b30"   # iOS system red (standard)

# Preferred font families. SF Pro is Apple's system font — not available on
# Linux, so Qt falls through to Ubuntu Sans (very clean modern sans, close
# in feel), Noto Sans, and finally the generic sans fallback. Chinese text
# auto-falls to Noto Sans CJK SC via Qt's per-glyph fallback.
FONT_STACK = (
    "'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', 'Inter', "
    "'Ubuntu Sans', 'Ubuntu', 'Noto Sans', 'Noto Sans CJK SC', "
    "'DejaVu Sans', sans-serif"
)
FONT_MONO_STACK = (
    "'SF Mono', 'JetBrains Mono', 'Fira Code', 'Ubuntu Mono', "
    "'DejaVu Sans Mono', monospace"
)


def _risk_color_hex(level) -> str:
    try:
        v = float(level)
    except (TypeError, ValueError):
        return C_TEXT_MUTED
    if v < 3:
        return C_OK
    if v < 5:
        return C_WARN
    if v < 7:
        return C_ORANGE
    return C_DANGER


def _action_color_hex(action: str) -> str:
    return {
        "none": C_OK,
        "verbal_warning": C_WARN,
        "alarm": C_ORANGE,
        "pull_over": C_DANGER,
    }.get(str(action).lower(), C_TEXT_MUTED)


def _product_font(size_pt: float, weight=QFont.Normal, letter_spacing=0.0):
    """Build a QFont from the Apple-leaning system stack."""
    f = QFont()
    f.setFamilies([
        "SF Pro Display", "SF Pro Text", "Helvetica Neue", "Inter",
        "Ubuntu Sans", "Ubuntu", "Noto Sans", "Noto Sans CJK SC",
        "DejaVu Sans",
    ])
    f.setPointSizeF(size_pt)
    f.setWeight(weight)
    if letter_spacing:
        f.setLetterSpacing(QFont.AbsoluteSpacing, letter_spacing)
    f.setHintingPreference(QFont.PreferFullHinting)
    f.setStyleStrategy(QFont.PreferAntialias)
    return f


# ----- color helpers for the multi-dimensional VLM panel ------------------

def _level_color(level) -> str:
    try:
        v = float(level)
    except (TypeError, ValueError):
        return "#9E9E9E"
    if v < 3:
        return "#2E7D32"   # green
    if v < 5:
        return "#FBC02D"   # yellow
    if v < 7:
        return "#F57C00"   # orange
    return "#C62828"       # red


def _severity_color(sev: str) -> str:
    return {
        "none": "#2E7D32",
        "low": "#FBC02D",
        "medium": "#F57C00",
        "high": "#C62828",
    }.get(str(sev).lower(), "#9E9E9E")


def _action_color(action: str) -> str:
    return {
        "none": "#2E7D32",
        "verbal_warning": "#FBC02D",
        "alarm": "#F57C00",
        "pull_over": "#C62828",
    }.get(str(action).lower(), "#9E9E9E")


def _action_label(action: str) -> str:
    return {
        "none": "无需动作",
        "verbal_warning": "语音提醒",
        "alarm": "报警",
        "pull_over": "立即靠边停车",
    }.get(str(action).lower(), str(action))


def _bool_dot(v: bool, true_color: str = "#C62828", false_color: str = "#2E7D32") -> str:
    return f"<span style='color:{true_color if v else false_color};font-size:14px;'>●</span>"


def _safe(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


# MediaPipe face-mesh landmark indices for the standard 6-point EAR computation
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [263, 387, 385, 362, 380, 373]


def _ear_from_landmarks(landmarks, idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]

    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    horiz = d(pts[0], pts[3])
    if horiz < 1e-6:
        return 0.0
    return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2.0 * horiz)


# =========================================================================
#                       Custom Qt widgets
# =========================================================================

class RiskGauge(QWidget):
    """
    Radial arc gauge for overall risk 0..10 — Apple/Tesla cockpit style.

    - Three minimal tick marks (0 / 5 / 10), no cluttering numbers
    - Center shows huge animated value (eased toward target) in Black weight
    - Bottom hosts the recommended-action banner with gradient fill
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 460)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._target = 0.0
        self._display = 0.0
        self._label = "INITIALIZING"
        self._color = QColor(C_TEXT_MUTED)
        self._action = "WAITING"
        self._action_color = QColor(C_TEXT_MUTED)
        self._fast_w = 1.0
        self._slow_w = 0.0

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._step)
        self._tick_timer.start(16)  # ~60 FPS animation

    def set_state(self, value, label, color_hex,
                  action_label, action_color_hex,
                  fast_w=1.0, slow_w=0.0):
        self._target = float(value)
        self._label = str(label).upper()
        self._color = QColor(color_hex)
        self._action = str(action_label)
        self._action_color = QColor(action_color_hex)
        self._fast_w = fast_w
        self._slow_w = slow_w

    def _step(self):
        d = self._target - self._display
        if abs(d) > 0.01:
            self._display += d * 0.18
            self.update()
        elif self._display != self._target:
            self._display = self._target
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        w, h = self.width(), self.height()
        banner_h = 72
        gauge_area_h = h - banner_h - 24

        side = min(w - 60, gauge_area_h - 20)
        side = max(side, 240)
        cx = w / 2.0
        cy = 24 + side / 2.0
        r = side / 2.0
        arc_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)

        start_angle = int(225 * 16)
        span_bg = int(-270 * 16)

        # --- background arc (iOS separator grey) ---
        p.setPen(QPen(QColor(C_BORDER), 20, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(arc_rect, start_angle, span_bg)

        # --- minimal tick marks at 0, 5, 10 only ---
        for i in (0, 5, 10):
            frac = i / 10.0
            angle_deg = 225.0 - frac * 270.0
            ar = math.radians(angle_deg)
            r_in = r - 28
            r_out = r + 8
            x1 = cx + r_in * math.cos(ar)
            y1 = cy - r_in * math.sin(ar)
            x2 = cx + r_out * math.cos(ar)
            y2 = cy - r_out * math.sin(ar)
            p.setPen(QPen(QColor(C_BORDER_2), 3))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # --- value arc ---
        val = max(0.0, min(10.0, self._display))
        sweep_deg = -(val / 10.0) * 270.0
        p.setPen(QPen(self._color, 20, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(arc_rect, start_angle, int(sweep_deg * 16))

        # --- tiny "RISK" caption above the number ---
        p.setPen(QColor(C_TEXT_MUTED))
        cap_font = _product_font(10, QFont.Black, letter_spacing=3)
        p.setFont(cap_font)
        cap_rect = QRectF(cx - r, cy - 92, 2 * r, 16)
        p.drawText(cap_rect, Qt.AlignCenter, "OVERALL RISK")

        # --- hero number (huge, Black weight) ---
        p.setPen(QColor(C_TEXT))
        num_font = _product_font(56, QFont.Black, letter_spacing=-2)
        p.setFont(num_font)
        num_rect = QRectF(cx - r, cy - 76, 2 * r, 120)
        p.drawText(num_rect, Qt.AlignCenter, f"{self._display:.1f}")

        # --- "/ 10" subtle below the number ---
        p.setPen(QColor(C_TEXT_FAINT))
        slash_font = _product_font(14, QFont.Bold, letter_spacing=2)
        p.setFont(slash_font)
        slash_rect = QRectF(cx - r, cy + 40, 2 * r, 18)
        p.drawText(slash_rect, Qt.AlignCenter, "OUT OF 10")

        # --- risk label (colored, large) ---
        p.setPen(self._color)
        label_font = _product_font(18, QFont.Black, letter_spacing=2)
        p.setFont(label_font)
        label_rect = QRectF(cx - r, cy + 60, 2 * r, 28)
        p.drawText(label_rect, Qt.AlignCenter, self._label)

        # --- fusion weight caption ---
        p.setPen(QColor(C_TEXT_MUTED))
        wf_font = _product_font(10, QFont.Bold, letter_spacing=2.5)
        p.setFont(wf_font)
        weight_rect = QRectF(cx - r, cy + 90, 2 * r, 16)
        p.drawText(weight_rect, Qt.AlignCenter,
                   f"FAST  {self._fast_w * 100:.0f}%      SLOW  {self._slow_w * 100:.0f}%")

        # --- recommended-action banner (iOS-style rounded rect) ---
        banner_rect = QRectF(20, h - banner_h - 4, w - 40, banner_h - 6)
        path = QPainterPath()
        path.addRoundedRect(banner_rect, 18, 18)

        grad = QLinearGradient(banner_rect.topLeft(), banner_rect.bottomLeft())
        grad.setColorAt(0.0, self._action_color.lighter(118))
        grad.setColorAt(1.0, self._action_color.darker(110))
        p.fillPath(path, QBrush(grad))

        # caption
        p.setPen(QColor(255, 255, 255, 220))
        banner_cap = _product_font(10, QFont.Black, letter_spacing=3)
        p.setFont(banner_cap)
        label_line = QRectF(banner_rect.x(), banner_rect.y() + 10,
                            banner_rect.width(), 16)
        p.drawText(label_line, Qt.AlignCenter, "⚠  RECOMMENDED ACTION")

        # value
        p.setPen(QColor("#ffffff"))
        banner_val = _product_font(20, QFont.Black, letter_spacing=1)
        p.setFont(banner_val)
        value_line = QRectF(banner_rect.x(), banner_rect.y() + 30,
                            banner_rect.width(), 32)
        p.drawText(value_line, Qt.AlignCenter, self._action)


class StatusChip(QWidget):
    """Apple-style rounded status pill — dot + bold caps text, optional blink."""

    _CHIP_FONT = None  # lazy cache

    def __init__(self, text, color_hex, blinking=False, parent=None):
        super().__init__(parent)
        self._text = text.upper()
        self._color_hex = color_hex
        self._blink = blinking
        self._on = True
        self.setFixedHeight(34)
        self._font = _product_font(11, QFont.Black, letter_spacing=2)
        self._reflow()
        if blinking:
            self._t = QTimer(self)
            self._t.timeout.connect(self._toggle)
            self._t.start(700)

    def _reflow(self):
        fm = QFontMetrics(self._font)
        w = fm.horizontalAdvance(self._text) + 52
        self.setFixedWidth(w)

    def set_text(self, text):
        self._text = text.upper()
        self._reflow()
        self.update()

    def set_color(self, color_hex):
        self._color_hex = color_hex
        self.update()

    def _toggle(self):
        self._on = not self._on
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, 17, 17)
        p.fillPath(path, QColor(C_CARD_2))

        edge = QColor(self._color_hex)
        edge.setAlpha(180)
        p.setPen(QPen(edge, 1.4))
        p.drawPath(path)

        # dot
        ds = 10
        dc = QColor(self._color_hex)
        if self._blink and not self._on:
            dc.setAlpha(60)
        p.setBrush(QBrush(dc))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(15, (self.height() - ds) / 2, ds, ds))

        p.setPen(QColor(C_TEXT))
        p.setFont(self._font)
        tr = QRectF(32, 0, self.width() - 38, self.height())
        p.drawText(tr, Qt.AlignLeft | Qt.AlignVCenter, self._text)


class HUDVideoLabel(QLabel):
    """Video display with tactical HUD corner markers overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QLabel {{ background: #000000; "
            f"border: 1px solid {C_BORDER}; border-radius: 12px; }}"
        )
        self.setAlignment(Qt.AlignCenter)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        m = 16
        L = 34
        p.setPen(QPen(QColor(C_ACCENT), 2.2))
        p.drawLine(m, m, m + L, m)
        p.drawLine(m, m, m, m + L)
        p.drawLine(w - m, m, w - m - L, m)
        p.drawLine(w - m, m, w - m, m + L)
        p.drawLine(m, h - m, m + L, h - m)
        p.drawLine(m, h - m, m, h - m - L)
        p.drawLine(w - m, h - m, w - m - L, h - m)
        p.drawLine(w - m, h - m, w - m, h - m - L)

        # tiny center crosshair
        p.setPen(QPen(QColor(C_ACCENT), 1))
        cx, cy = w / 2, h / 2
        p.drawLine(int(cx - 8), int(cy), int(cx - 3), int(cy))
        p.drawLine(int(cx + 3), int(cy), int(cx + 8), int(cy))
        p.drawLine(int(cx), int(cy - 8), int(cx), int(cy - 3))
        p.drawLine(int(cx), int(cy + 3), int(cx), int(cy + 8))


class Sparkline(QWidget):
    """Compact time-series sparkline over a deque of floats."""

    def __init__(self, maxlen=60, color_hex=C_ACCENT, parent=None):
        super().__init__(parent)
        self._data = deque(maxlen=maxlen)
        self._color = QColor(color_hex)
        self.setMinimumHeight(36)

    def push(self, value):
        self._data.append(float(value))
        self.update()

    def set_color(self, color_hex):
        self._color = QColor(color_hex)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()

        # background grid
        p.setPen(QPen(QColor(C_BORDER), 1, Qt.DotLine))
        for i in range(1, 4):
            y = h * i / 4
            p.drawLine(0, int(y), w, int(y))

        if len(self._data) < 2:
            return

        lo = min(self._data)
        hi = max(self._data)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        n = len(self._data)

        path = QPainterPath()
        for i, v in enumerate(self._data):
            x = i * w / max(n - 1, 1)
            y = h - (v - lo) / (hi - lo) * h
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        p.setPen(QPen(self._color, 2))
        p.drawPath(path)

        # filled area under line, faded
        fill_path = QPainterPath(path)
        fill_path.lineTo(w, h)
        fill_path.lineTo(0, h)
        fill_path.closeSubpath()
        fill_color = QColor(self._color)
        fill_color.setAlpha(40)
        p.fillPath(fill_path, QBrush(fill_color))


class SectionCard(QFrame):
    """
    Rounded card container — Apple-style: soft fill, subtle 1 px border,
    20 px radius. All right-column panels and the bottom VLM strip use
    this as their base.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"SectionCard {{ background: {C_CARD}; "
            f"border: 1px solid {C_BORDER}; border-radius: 20px; }}"
        )


class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        self.yawn_state = ''
        self.left_eye_state =''
        self.right_eye_state= ''
        self.alert_text = ''

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0 

        self.left_eye_still_closed = False  
        self.right_eye_still_closed = False 
        self.yawn_in_progress = False  
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        # ---- sliding-window state for Fast System metrics ----
        # ~10 s @ 30 fps for both PERCLOS and YOLO confidence
        self._eye_history = deque(maxlen=300)
        self._conf_history = deque(maxlen=300)
        self._ear_value = 0.0
        self._slow_submit_counter = 0

        # Fast / Slow / Fused state shared with the GUI
        self._fast_state = {
            "drowsiness_level": 0.0,
            "confidence": 0.0,
            "perclos": 0.0,
            "ear": 0.0,
            "microsleeps": 0.0,
            "yawns": 0,
            "yawn_duration": 0.0,
        }
        self._slow_state = None
        self._fusion = DecisionFusion(slow_max_age_s=30.0)
        self._fusion_result = None

        # ---- thread-safe UI data plumbing ----
        self._frame_lock = threading.Lock()
        self._latest_display_frame = None   # ndarray, most recent frame
        self._session_start = time.time()
        self._frame_stamps = deque(maxlen=60)  # for FPS
        self._perclos_series = deque(maxlen=120)

        # ================= SENTINEL DMS product UI =================
        self.setWindowTitle(f"{BRAND_NAME} DMS  ·  {BRAND_SUB}")
        self.setGeometry(30, 30, 1860, 1080)
        self.setStyleSheet(
            f"QMainWindow {{ background-color: {C_BG}; }}"
            f"QWidget {{ background-color: {C_BG}; color: {C_TEXT}; }}"
        )

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        root_layout = QVBoxLayout(self.central_widget)
        root_layout.setContentsMargins(26, 18, 26, 14)
        root_layout.setSpacing(14)

        # ================================================================
        #                        TOP BRAND BAR
        # ================================================================
        top_bar = QWidget()
        top_bar.setFixedHeight(72)
        top_bar.setStyleSheet(f"background: transparent;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 12, 10, 12)
        top_layout.setSpacing(16)

        self.brand_label = QLabel(
            f"<span style='color:{C_ACCENT};font-size:32px;font-weight:900;'>◈</span>"
            f"<span style='color:{C_TEXT};font-size:28px;font-weight:900;"
            f"letter-spacing:2px;'>  {BRAND_NAME}</span>"
            f"<span style='color:{C_TEXT_MUTED};font-size:13px;"
            f"font-weight:bold;letter-spacing:2.5px;'>"
            f"   ·   {BRAND_SUB}</span>"
        )
        self.brand_label.setStyleSheet("background: transparent;")
        top_layout.addWidget(self.brand_label)

        top_layout.addStretch()

        self.chip_live = StatusChip("LIVE", C_DANGER, blinking=True)
        self.chip_fast = StatusChip("FAST  30 FPS", C_ACCENT)
        self.chip_slow = StatusChip("VLM STANDBY", C_TEXT_MUTED)
        top_layout.addWidget(self.chip_live)
        top_layout.addWidget(self.chip_fast)
        top_layout.addWidget(self.chip_slow)

        top_layout.addSpacing(18)

        self.clock_label = QLabel()
        self.clock_label.setStyleSheet(
            f"color: {C_TEXT}; background: transparent;"
        )
        self.clock_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.clock_label.setMinimumWidth(300)
        top_layout.addWidget(self.clock_label)

        root_layout.addWidget(top_bar)

        # Thin accent divider under top bar
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {C_BORDER}; border: 0;")
        root_layout.addWidget(divider)

        # ================================================================
        #                       MAIN CONTENT ROW
        # ================================================================
        content_row = QWidget()
        content_row.setStyleSheet("background: transparent;")
        content_layout = QHBoxLayout(content_row)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        # ----------------- LEFT: video + caption ----------------
        left_col = QWidget()
        left_col.setStyleSheet("background: transparent;")
        left_v = QVBoxLayout(left_col)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(12)

        self.video_label = HUDVideoLabel()
        self.video_label.setFixedSize(900, 676)
        left_v.addWidget(self.video_label, 0, Qt.AlignHCenter)

        # Caption bar under video
        video_caption = QLabel(
            f"<span style='color:{C_TEXT_DIM};font-size:12px;"
            f"font-weight:900;letter-spacing:2.2px;'>"
            f"LIVE CAMERA FEED</span>"
            f"<span style='color:{C_TEXT_MUTED};font-size:11px;"
            f"font-weight:bold;letter-spacing:2px;'>"
            f"     /dev/video0     MEDIAPIPE + YOLOv8</span>"
        )
        video_caption.setStyleSheet("background: transparent;")
        video_caption.setAlignment(Qt.AlignCenter)
        left_v.addWidget(video_caption)

        content_layout.addWidget(left_col, 0)

        # ----------------- RIGHT: stacked cards ----------------
        right_col = QWidget()
        right_col.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)

        def _make_section_header(text):
            lbl = QLabel(
                f"<span style='color:{C_TEXT_DIM};font-size:12px;"
                f"font-weight:900;letter-spacing:2.5px;'>{text}</span>"
            )
            lbl.setStyleSheet("background: transparent;")
            return lbl

        # --- Gauge card ---
        gauge_card = SectionCard()
        gauge_card_layout = QVBoxLayout(gauge_card)
        gauge_card_layout.setContentsMargins(24, 18, 24, 18)
        gauge_card_layout.setSpacing(8)

        gauge_card_layout.addWidget(_make_section_header(
            "◆  FUSION OUTPUT     FAST + SLOW AGGREGATED"
        ))

        self.risk_gauge = RiskGauge()
        gauge_card_layout.addWidget(self.risk_gauge, 1)

        right_layout.addWidget(gauge_card, 6)

        # --- Fast metrics card ---
        fast_card = SectionCard()
        fast_card_layout = QVBoxLayout(fast_card)
        fast_card_layout.setContentsMargins(26, 18, 26, 18)
        fast_card_layout.setSpacing(10)

        fast_card_layout.addWidget(_make_section_header(
            "●  FAST SYSTEM     REAL-TIME DETECTION"
        ))

        self.fast_label = QLabel()
        self.fast_label.setStyleSheet(
            "QLabel { background: transparent; border: none; }"
        )
        self.fast_label.setWordWrap(True)
        self.fast_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        fast_card_layout.addWidget(self.fast_label)

        # sparkline for PERCLOS history
        self.sparkline = Sparkline(maxlen=120, color_hex=C_ACCENT)
        fast_card_layout.addWidget(self.sparkline)

        right_layout.addWidget(fast_card, 3)

        # --- Slow (VLM) multi-dimension card ---
        slow_card = SectionCard()
        slow_card_layout = QVBoxLayout(slow_card)
        slow_card_layout.setContentsMargins(26, 18, 26, 18)
        slow_card_layout.setSpacing(10)

        slow_card_layout.addWidget(_make_section_header(
            "◆  SLOW SYSTEM     VLM MULTI-DIMENSIONAL ANALYSIS"
        ))

        self.slow_label = QLabel()
        self.slow_label.setStyleSheet(
            "QLabel { background: transparent; border: none; }"
        )
        self.slow_label.setWordWrap(True)
        self.slow_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        slow_card_layout.addWidget(self.slow_label, 1)

        right_layout.addWidget(slow_card, 4)

        content_layout.addWidget(right_col, 1)
        root_layout.addWidget(content_row, 1)

        # ================================================================
        #                 BOTTOM VLM EXPLANATION STRIP
        # ================================================================
        bottom_card = SectionCard()
        bottom_layout = QVBoxLayout(bottom_card)
        bottom_layout.setContentsMargins(32, 20, 32, 20)
        bottom_layout.setSpacing(10)

        bottom_header = QLabel(
            f"<span style='color:{C_TEXT_DIM};font-size:12px;"
            f"font-weight:900;letter-spacing:2.5px;'>"
            f"📄  VLM ANALYSIS REPORT     REAL-TIME NATURAL LANGUAGE</span>"
        )
        bottom_header.setStyleSheet("background: transparent;")
        bottom_layout.addWidget(bottom_header)

        self.final_label = QLabel()
        self.final_label.setStyleSheet(
            "QLabel { background: transparent; border: none; }"
        )
        self.final_label.setWordWrap(True)
        self.final_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.final_label.setMinimumHeight(90)
        bottom_layout.addWidget(self.final_label)

        root_layout.addWidget(bottom_card)

        # ================================================================
        #                      FOOTER STATUS BAR
        # ================================================================
        footer = QWidget()
        footer.setFixedHeight(36)
        footer.setStyleSheet("background: transparent;")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 4, 10, 4)

        self.footer_left = QLabel()
        self.footer_left.setStyleSheet(
            f"color: {C_TEXT_MUTED}; background: transparent;"
        )
        footer_layout.addWidget(self.footer_left)
        footer_layout.addStretch()

        self.footer_right = QLabel(
            f"<span style='color:{C_ACCENT};font-size:12px;font-weight:900;"
            f"letter-spacing:2.5px;'>◈ {BRAND_NAME}</span>"
            f"<span style='color:{C_TEXT_MUTED};font-size:12px;font-weight:900;"
            f"letter-spacing:2.5px;'>   DMS  {BRAND_VERSION}</span>"
        )
        self.footer_right.setStyleSheet("background: transparent;")
        footer_layout.addWidget(self.footer_right)

        root_layout.addWidget(footer)

        self.update_info()

        self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
        self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")

        # Slow System (VLM) — DashScope OpenAI-compatible Qwen-Omni.
        # Credentials MUST be supplied via env vars (no hardcoded keys):
        #   DASHSCOPE_API_KEY   required
        #   DASHSCOPE_BASE_URL  optional, defaults to DashScope compat endpoint
        #   DASHSCOPE_MODEL     optional, defaults to qwen3.5-omni-plus
        # If DASHSCOPE_API_KEY is unset the SlowSystem falls back to mock mode
        # so the UI still exercises the full pipeline without an API key.
        import os
        _api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        # SLOW_INTERVAL_SECONDS <= 0 → fire back-to-back, bounded only by
        # VLM latency (max throughput). Default: 0 (= as fast as possible).
        _interval = float(os.environ.get("SLOW_INTERVAL_SECONDS", "0"))
        _image_max = int(os.environ.get("SLOW_IMAGE_MAX_SIDE", "480"))
        self._slow_system = SlowSystem(
            SlowSystemConfig(
                interval_seconds=_interval,
                mock_mode=(_api_key == ""),
                base_url=os.environ.get(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                api_key=_api_key,
                model_name=os.environ.get(
                    "DASHSCOPE_MODEL", "qwen3.5-omni-flash"
                ),
                request_timeout=40.0,
                image_max_side=_image_max,
                jpeg_quality=80,
            )
        )
        self._slow_system.start()

        self.cap = cv2.VideoCapture(0)
        time.sleep(1.000)

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()

        # Main-thread UI tick — pulls shared state and repaints every ~33 ms.
        # Widget access must stay on the GUI thread, so worker threads only
        # update state dicts + latest frame under _frame_lock.
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._ui_tick)
        self._ui_timer.start(33)

    def _ui_tick(self):
        """Runs on the main thread; renders shared state into widgets."""
        with self._frame_lock:
            frame = self._latest_display_frame
        if frame is not None:
            self.display_frame(frame)
        try:
            self.update_info()
        except Exception as exc:
            print(f"[ui_tick] update_info error: {exc}")

    # ------------------------------------------------------------------
    # update_info — drive all product-grade widgets from shared state
    # ------------------------------------------------------------------
    def update_info(self):
        fast = self._fast_state
        slow = self._slow_state
        fused = self._fusion_result

        risk_label_map = {
            "正常": (C_OK, "NORMAL"),
            "轻度疲劳": (C_WARN, "MILD FATIGUE"),
            "中度疲劳": "#f97316",  # overwritten below — placeholder
            "严重疲劳": C_DANGER,
        }

        def risk_label_pair(zh: str):
            return {
                "正常": (C_OK, "NORMAL"),
                "轻度疲劳": (C_WARN, "MILD FATIGUE"),
                "中度疲劳": ("#f97316", "MODERATE FATIGUE"),
                "严重疲劳": (C_DANGER, "SEVERE FATIGUE"),
            }.get(zh, (C_TEXT_MUTED, "INITIALIZING"))

        # ===== overall risk value (prefer VLM overall_risk, else fused drowsiness) =====
        if slow is not None:
            overall_val = float(_safe(slow, "overall_risk", default=0) or 0)
        elif fused is not None:
            overall_val = float(fused.drowsiness_level)
        else:
            overall_val = 0.0

        risk_zh = fused.risk_label if fused is not None else "初始化中"
        risk_color, risk_en = risk_label_pair(risk_zh)
        risk_text = f"{risk_en}  ·  {risk_zh}"

        action = _safe(slow, "recommended_action", default="none") or "none"
        action_color = _action_color_hex(action)
        action_text = {
            "none": "NO ACTION",
            "verbal_warning": "VERBAL WARNING",
            "alarm": "AUDIBLE ALARM",
            "pull_over": "PULL OVER IMMEDIATELY",
        }.get(str(action).lower(), str(action).upper())

        fw = fused.fast_weight if fused else 1.0
        sw = fused.slow_weight if fused else 0.0

        # ---- drive the risk gauge widget ----
        self.risk_gauge.set_state(
            value=overall_val,
            label=risk_text,
            color_hex=risk_color,
            action_label=action_text,
            action_color_hex=action_color,
            fast_w=fw,
            slow_w=sw,
        )

        # ===== FAST card content =====
        fast_drowsy_color = _risk_color_hex(fast["drowsiness_level"])
        perclos_color = _risk_color_hex(fast["perclos"] * 12)

        def metric(label_en, value_html):
            return (
                f"<td width='25%' valign='top'>"
                f"<div style='color:{C_TEXT_MUTED};font-size:11px;"
                f"font-weight:900;letter-spacing:2px;'>{label_en}</div>"
                f"<div style='margin-top:6px;'>{value_html}</div>"
                "</td>"
            )

        def big(value, color=C_TEXT, suffix="", suffix_color=C_TEXT_MUTED):
            out = (f"<span style='color:{color};font-size:34px;"
                   f"font-weight:900;letter-spacing:-1.5px;'>{value}</span>")
            if suffix:
                out += (f"<span style='color:{suffix_color};font-size:15px;"
                        f"font-weight:900;'>  {suffix}</span>")
            return out

        fast_html = (
            "<table width='100%' cellpadding='0' cellspacing='0' "
            "style='margin-top:8px;'>"
            "<tr>"
            + metric("BLINKS", big(self.blinks))
            + metric("MICROSLEEP",
                     big(f"{round(self.microsleeps, 2):.2f}", suffix="s"))
            + metric("YAWNS", big(self.yawns))
            + metric("PERCLOS",
                     big(f"{fast['perclos'] * 100:.1f}",
                         color=perclos_color, suffix="%"))
            + "</tr>"
            "<tr><td height='20'></td></tr>"
            "<tr>"
            + metric("EAR",
                     big(f"{fast['ear']:.3f}"))
            + metric("YAWN DURATION",
                     big(f"{round(self.yawn_duration, 2):.2f}", suffix="s"))
            + metric("FAST DROWSINESS",
                     big(f"{fast['drowsiness_level']:.1f}",
                         color=fast_drowsy_color, suffix="/ 10"))
            + metric("FAST CONFIDENCE",
                     big(f"{fast['confidence']:.2f}"))
            + "</tr>"
            "</table>"
        )
        self.fast_label.setText(fast_html)

        # feed sparkline with latest PERCLOS
        if self._perclos_series:
            self.sparkline.push(self._perclos_series[-1])
            self.sparkline.set_color(perclos_color)

        # ===== SLOW (VLM multi-dim) card content =====
        if slow is None:
            slow_html = (
                "<table width='100%' height='220' cellpadding='0' cellspacing='0'>"
                "<tr><td align='center' valign='middle'>"
                f"<span style='color:{C_TEXT_MUTED};font-size:16px;"
                f"font-weight:900;letter-spacing:2px;'>"
                f"⏳     AWAITING FIRST VLM SAMPLE"
                f"</span>"
                "</td></tr></table>"
            )
        else:
            age = max(0.0, time.time() - float(slow.get("timestamp", time.time())))

            d_level = _safe(slow, "drowsiness", "level", default=0) or 0
            d_conf = _safe(slow, "drowsiness", "confidence", default=0.0) or 0.0
            d_color = _risk_color_hex(d_level)

            di_det = bool(_safe(slow, "distraction", "detected", default=False))
            di_type = _safe(slow, "distraction", "type", default="none") or "none"
            di_conf = float(_safe(slow, "distraction", "confidence", default=0.0) or 0.0)
            di_color = C_DANGER if di_det else C_OK
            di_text = di_type.upper() if di_det else "NONE"

            an_det = bool(_safe(slow, "anomaly", "detected", default=False))
            an_sev = _safe(slow, "anomaly", "severity", default="none") or "none"
            an_desc = _safe(slow, "anomaly", "description", default="") or ""
            an_color = {
                "none": C_OK, "low": C_WARN,
                "medium": "#f97316", "high": C_DANGER,
            }.get(str(an_sev).lower(), C_TEXT_MUTED)
            if an_det and an_desc:
                an_text = an_desc
            elif an_det:
                an_text = str(an_sev).upper()
            else:
                an_text = "CLEAR"

            occ_types = _safe(slow, "occlusion", "type", default=["none"]) or ["none"]
            occ_impact = float(_safe(slow, "occlusion", "impact_on_reliability",
                                     default=0.0) or 0.0)
            occ_text = ", ".join([t.upper() for t in occ_types]) if occ_types else "NONE"
            occ_color = _risk_color_hex(occ_impact * 10)

            lighting = _safe(slow, "context", "lighting", default="good") or "good"
            light_color = {
                "good": C_OK, "dim": C_WARN, "dark": C_DANGER,
            }.get(str(lighting).lower(), C_TEXT_MUTED)
            passengers = bool(_safe(slow, "context", "passengers_detected", default=False))

            # Unicode block progress bar, bigger and more opaque
            def bar(fraction, color_hex, width=12):
                fraction = max(0.0, min(1.0, fraction))
                filled = int(round(fraction * width))
                empty = width - filled
                return (
                    f"<span style='color:{color_hex};"
                    f"font-family:{FONT_MONO_STACK};"
                    f"font-size:17px;letter-spacing:-2px;'>"
                    f"{'█' * filled}</span>"
                    f"<span style='color:{C_BORDER};"
                    f"font-family:{FONT_MONO_STACK};"
                    f"font-size:17px;letter-spacing:-2px;'>"
                    f"{'█' * empty}</span>"
                )

            di_bar = bar(di_conf if di_det else 0, di_color)
            an_bar = bar({"none": 0, "low": 0.33, "medium": 0.66,
                          "high": 1.0}.get(str(an_sev).lower(), 0), an_color)
            occ_bar = bar(occ_impact, occ_color)

            def row(key, value, color_hex, bar_html=""):
                return (
                    "<tr>"
                    f"<td width='150' style='color:{C_TEXT_MUTED};"
                    f"font-size:11px;font-weight:900;letter-spacing:2px;"
                    f"padding-top:4px;'>"
                    f"{key}"
                    "</td>"
                    f"<td style='color:{C_TEXT};font-size:20px;font-weight:900;"
                    f"padding-top:4px;'>"
                    f"<span style='color:{color_hex};'>{value}</span>"
                    "</td>"
                    f"<td align='right' width='180' style='padding-top:6px;'>{bar_html}</td>"
                    "</tr>"
                )

            d_label = (
                f"{d_level}<span style='color:{C_TEXT_MUTED};"
                f"font-size:15px;font-weight:900;'> / 10</span>"
                f"  <span style='color:{C_TEXT_MUTED};font-size:13px;"
                f"font-weight:bold;'>conf {d_conf}</span>"
            )
            ctx_label = (
                f"{str(lighting).upper()} LIGHT    "
                f"{'WITH PASSENGERS' if passengers else 'SOLO'}"
            )

            rows = (
                row("DROWSINESS", d_label, d_color,
                    bar(float(d_level) / 10, d_color))
                + row("DISTRACTION",
                      f"{di_text}"
                      + (f"  <span style='color:{C_TEXT_MUTED};font-size:13px;"
                         f"font-weight:bold;'>conf {di_conf:.2f}</span>"
                         if di_det else ""),
                      di_color, di_bar)
                + row("ANOMALY", an_text, an_color, an_bar)
                + row("OCCLUSION",
                      f"{occ_text}"
                      f"  <span style='color:{C_TEXT_MUTED};font-size:13px;"
                      f"font-weight:bold;'>impact {occ_impact:.2f}</span>",
                      occ_color, occ_bar)
                + row("CONTEXT", ctx_label, light_color, "")
            )

            slow_html = (
                "<table width='100%' cellpadding='10' cellspacing='0' "
                "style='margin-top:8px;'>"
                f"{rows}"
                "</table>"
                f"<div style='color:{C_TEXT_MUTED};font-size:11px;"
                f"letter-spacing:1.8px;margin-top:14px;font-weight:900;'>"
                f"MODEL  {slow.get('source','?')}       "
                f"AGE  {age:.1f}s       "
                f"LATENCY  {slow.get('latency_s', 0)}s"
                "</div>"
            )
        self.slow_label.setText(slow_html)

        # ===== Bottom explanation strip =====
        full_explanation = _safe(slow, "explanation", default="") or ""
        if not full_explanation:
            final_html = (
                f"<span style='color:{C_TEXT_MUTED};font-size:16px;'>"
                f"Awaiting first VLM analysis cycle…"
                "</span>"
            )
        else:
            final_html = (
                f"<div style='color:{C_TEXT};font-size:19px;"
                f"line-height:1.7;font-weight:500;'>{full_explanation}</div>"
            )
        self.final_label.setText(final_html)

        # ===== Header chips =====
        # FPS
        stamps = list(self._frame_stamps)
        if len(stamps) >= 2:
            fps = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-6)
        else:
            fps = 0.0
        self.chip_fast.set_text(f"FAST  {fps:.0f} FPS")

        # VLM chip
        if slow is None:
            self.chip_slow.set_text("VLM STANDBY")
            self.chip_slow.set_color(C_TEXT_MUTED)
        else:
            age = max(0.0, time.time() - float(slow.get("timestamp", time.time())))
            source = slow.get("source", "VLM")
            if source == "error":
                self.chip_slow.set_text("VLM ERROR")
                self.chip_slow.set_color(C_DANGER)
            elif age > 30:
                self.chip_slow.set_text("VLM STALE")
                self.chip_slow.set_color(C_WARN)
            else:
                self.chip_slow.set_text(f"VLM  {age:.0f}S AGO")
                self.chip_slow.set_color(C_ACCENT)

        # Clock
        now = datetime.now()
        self.clock_label.setText(
            f"<span style='color:{C_TEXT_MUTED};font-size:11px;"
            f"font-weight:900;letter-spacing:2.5px;'>SYSTEM TIME</span>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;"
            f"<span style='color:{C_TEXT};font-size:20px;"
            f"font-weight:900;letter-spacing:1px;'>"
            f"{now.strftime('%H:%M:%S')}</span>"
        )

        # Footer
        session_s = int(time.time() - self._session_start)
        hh, rem = divmod(session_s, 3600)
        mm, ss = divmod(rem, 60)
        model_name = _safe(slow, "source", default="qwen3.5-omni-flash")
        self.footer_left.setText(
            f"<span style='color:{C_TEXT_MUTED};font-size:12px;"
            f"font-weight:900;letter-spacing:2px;'>"
            f"SESSION  {hh:02d}:{mm:02d}:{ss:02d}"
            f"       VLM  {model_name}"
            f"       FAST  {fps:.0f} FPS"
            f"       GPU  RTX 5070 TI"
            f"</span>"
        )


    def predict_eye(self, eye_frame, eye_state):
        results_eye = self.detecteye.predict(eye_frame, verbose=False)
        boxes = results_eye[0].boxes
        if len(boxes) == 0:
            return eye_state, 0.0

        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        max_confidence_index = int(np.argmax(confidences))
        class_id = int(class_ids[max_confidence_index])
        conf = float(confidences[max_confidence_index])

        if class_id == 1:
            eye_state = "Close Eye"
        elif class_id == 0 and conf > 0.30:
            eye_state = "Open Eye"

        return eye_state, conf

    def predict_yawn(self, yawn_frame):
        results_yawn = self.detectyawn.predict(yawn_frame, verbose=False)
        boxes = results_yawn[0].boxes

        if len(boxes) == 0:
            return self.yawn_state

        confidences = boxes.conf.cpu().numpy()  
        class_ids = boxes.cls.cpu().numpy()  
        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])

        if class_id == 0:
            self.yawn_state = "Yawn"
        elif class_id == 1 and confidences[max_confidence_index] > 0.50 :
            self.yawn_state = "No Yawn"
                            

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
            else:
                break

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Push every 3rd frame (~10 Hz) so the VLM worker always has a
            # very fresh sample when it wakes. submit_frame() is a cheap
            # locked-memcopy, so this is much lighter than the actual VLM
            # round-trip.
            self._slow_submit_counter = (self._slow_submit_counter + 1) % 3
            if self._slow_submit_counter == 0:
                self._slow_system.submit_frame(frame)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            conf_l = conf_r = 0.0
            face_seen = False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_seen = True
                    ih, iw, _ = frame.shape

                    # ---- EAR from MediaPipe landmarks ----
                    self._ear_value = (
                        _ear_from_landmarks(face_landmarks.landmark, LEFT_EYE_EAR_IDX, iw, ih)
                        + _ear_from_landmarks(face_landmarks.landmark, RIGHT_EYE_EAR_IDX, iw, ih)
                    ) / 2.0

                    points = []
                    for point_id in self.points_ids:
                        lm = face_landmarks.landmark[point_id]
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        points.append((x, y))

                    if len(points) != 0:
                        x1, y1 = points[0]
                        x2, _ = points[1]
                        _, y3 = points[2]

                        x4, y4 = points[3]
                        x5, y5 = points[4]

                        x6, y6 = points[5]
                        x7, y7 = points[6]

                        x6, x7 = min(x6, x7), max(x6, x7)
                        y6, y7 = min(y6, y7), max(y6, y7)

                        mouth_roi = frame[y1:y3, x1:x2]
                        right_eye_roi = frame[y4:y5, x4:x5]
                        left_eye_roi = frame[y6:y7, x6:x7]

                        try:
                            self.left_eye_state, conf_l = self.predict_eye(
                                left_eye_roi, self.left_eye_state
                            )
                            self.right_eye_state, conf_r = self.predict_eye(
                                right_eye_roi, self.right_eye_state
                            )
                            self.predict_yawn(mouth_roi)
                        except Exception as e:
                            print(f"Error al realizar la predicción: {e}")

                        # ---- update sliding-window state for PERCLOS / fast conf ----
                        both_closed = (
                            self.left_eye_state == "Close Eye"
                            and self.right_eye_state == "Close Eye"
                        )
                        self._eye_history.append(both_closed)
                        self._conf_history.append(max(conf_l, conf_r))

                        if both_closed:
                            if not self.left_eye_still_closed and not self.right_eye_still_closed:
                                self.left_eye_still_closed, self.right_eye_still_closed = True, True
                                self.blinks += 1
                            self.microsleeps += 45 / 1000
                        else:
                            if self.left_eye_still_closed and self.right_eye_still_closed:
                                self.left_eye_still_closed, self.right_eye_still_closed = False, False
                            self.microsleeps = 0

                        if self.yawn_state == "Yawn":
                            if not self.yawn_in_progress:
                                self.yawn_in_progress = True
                                self.yawns += 1
                            self.yawn_duration += 45 / 1000
                        else:
                            if self.yawn_in_progress:
                                self.yawn_in_progress = False
                                self.yawn_duration = 0

            # Always run fast-state aggregation + fusion, even when no face
            # is detected, so the VLM panel stays responsive. UI rendering
            # is done by QTimer on the main thread, not here.
            self._update_fast_state(face_seen)
            self._poll_and_fuse()

            # track FPS and PERCLOS history
            self._frame_stamps.append(time.time())
            self._perclos_series.append(self._fast_state.get("perclos", 0.0) * 100)

            with self._frame_lock:
                self._latest_display_frame = frame

    # ------------------------------------------------------------------
    # Fast state aggregation + fusion
    # ------------------------------------------------------------------
    def _update_fast_state(self, face_seen: bool):
        n = max(len(self._eye_history), 1)
        perclos = sum(self._eye_history) / n
        avg_conf = (
            sum(self._conf_history) / max(len(self._conf_history), 1)
            if self._conf_history else 0.0
        )
        # If no face this frame, the YOLO confidence drops sharply for the
        # current sample — represent that by a small penalty so fusion learns
        # to lean on the slow system.
        if not face_seen:
            avg_conf *= 0.5

        # Fast drowsiness level 0..10 from PERCLOS, microsleeps, yawn duration
        level = 0.0
        level += min(perclos * 12.0, 5.0)            # 0..5 from PERCLOS
        level += min(self.microsleeps * 1.5, 3.0)    # 0..3 from microsleeps (s)
        level += min(self.yawn_duration * 0.5, 2.0)  # 0..2 from yawn duration (s)
        level = min(level, 10.0)

        self._fast_state.update({
            "drowsiness_level": level,
            "confidence": float(avg_conf),
            "perclos": float(perclos),
            "ear": float(self._ear_value),
            "microsleeps": float(self.microsleeps),
            "yawns": int(self.yawns),
            "yawn_duration": float(self.yawn_duration),
        })

    def _poll_and_fuse(self):
        self._slow_state = self._slow_system.poll_result()
        self._fusion_result = self._fusion.fuse(self._fast_state, self._slow_state)

    # ------------------------------------------------------------------
    # Qt
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.stop_event.set()
        try:
            self._ui_timer.stop()
        except Exception:
            pass
        try:
            self._slow_system.stop()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        super().closeEvent(event)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        # HUDVideoLabel is 900x676; leave ~20 px margin so the cyan corner
        # markers and center crosshair paint cleanly on top.
        p = qimg.scaled(
            872, 648, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def play_alert_sound(self):
            frequency = 1000
            duration = 500
            if _HAS_WINSOUND:
                winsound.Beep(frequency, duration)
            else:
                # Linux fallback: terminal bell + best-effort beep
                try:
                    import subprocess
                    subprocess.run(
                        ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                        check=False, timeout=1,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    print("\a", end="", flush=True)

    def play_sound_in_thread(self):
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()
        
    def show_alert_on_frame(self, frame, text="Alerta!"):
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)
        font_scale = 1
        font_color = (0, 0, 255) 
        line_type = 2

        cv2.putText(frame, text, position, font, font_scale, font_color, line_type)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Global system font — Apple SF Pro fallback stack (Ubuntu Sans on
    # Linux). This propagates to every widget / QLabel / QFrame and into
    # their rich-text content. Qt auto-falls back per-glyph to Noto Sans
    # CJK SC for Chinese characters.
    app_font = QFont()
    app_font.setFamilies([
        "SF Pro Display", "SF Pro Text", "Helvetica Neue", "Inter",
        "Ubuntu Sans", "Ubuntu", "Noto Sans", "Noto Sans CJK SC",
        "DejaVu Sans",
    ])
    app_font.setPointSize(11)
    app_font.setHintingPreference(QFont.PreferFullHinting)
    app_font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(app_font)

    # Global stylesheet: default bg + font-family inheritance for HTML
    # content inside QLabels.
    app.setStyleSheet(
        f"* {{ font-family: {FONT_STACK}; }}"
        f"QMainWindow, QWidget {{ background-color: {C_BG}; color: {C_TEXT}; }}"
        f"QToolTip {{ background: {C_CARD_2}; color: {C_TEXT}; "
        f"border: 1px solid {C_BORDER_2}; padding: 6px 10px; }}"
    )

    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())