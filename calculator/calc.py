# Hand-Gesture Virtual Calculator (Neon UI)
# Requirements:
#   pip install opencv-python mediapipe numpy

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math
import re

# ------------------- Settings -------------------
# Click mode: "pinch" (thumb + index) or "hover"
CLICK_MODE = "pinch"            # change to "hover" if you prefer dwell clicking
HOVER_HOLD_MS = 350             # how long to hover to click (ms)
PINCH_DIST_THRESH = 0.055       # normalized (0..1) distance threshold for pinch click
SMOOTH_WIN = 5                  # fingertip smoothing window
PRESS_COOLDOWN = 280            # ms between accepted clicks

# Camera
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720    # try 640x480 if your webcam struggles

# UI colors (BGR)
C_BG      = (0, 0, 0)           # pure black background
C_GLASS   = (25, 25, 25)        # panel background
C_NEON    = (255, 0, 255)       # neon pink
C_ACCENT  = (255, 255, 0)       # neon yellow
C_TEXT    = (240, 240, 240)
C_BTN     = (10, 10, 10)
C_BTN_HI  = (40, 0, 60)
C_BTN_ON  = (120, 0, 180)
C_GRID    = (80, 0, 120)

# Calculator layout
BTN_ROWS = [
    ["C", "(", ")", "DEL"],
    ["7", "8", "9", "/"],
    ["4", "5", "6", "*"],
    ["1", "2", "3", "-"],
    ["0", ".", "=", "+"],
]
PAD_LEFT   = 760
PAD_TOP    = 120
BTN_W      = 110
BTN_H      = 72
BTN_GAP_X  = 16
BTN_GAP_Y  = 16

DISPLAY_X  = 60
DISPLAY_Y  = 60
DISPLAY_W  = 1160
DISPLAY_H  = 44

# ------------------- Mediapipe -------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ------------------- Helpers ---------------------
class Button:
    def __init__(self, label, x, y, w, h):
        self.label = label
        self.x, self.y, self.w, self.h = x, y, w, h
        self.hover_t0 = 0
        self.last_flash_t = 0
        self.flash = False

    def rect(self):
        return (self.x, self.y, self.w, self.h)

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

def make_buttons():
    buttons = []
    y = PAD_TOP
    for row in BTN_ROWS:
        x = PAD_LEFT
        for lab in row:
            buttons.append(Button(lab, x, y, BTN_W, BTN_H))
            x += BTN_W + BTN_GAP_X
        y += BTN_H + BTN_GAP_Y
    return buttons

BUTTONS = make_buttons()

def draw_glow_rect(img, p1, p2, color, thickness=2, glow=10):
    x1, y1 = p1
    x2, y2 = p2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    for i in range(glow):
        alpha = max(0, 0.22 - i * 0.02)
        cv2.rectangle(overlay, (x1 - i, y1 - i), (x2 + i, y2 + i), color, 1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def draw_panel(img, x, y, w, h, fill=C_GLASS, border=C_NEON):
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
    draw_glow_rect(img, (x, y), (x + w, y + h), border, 1, 12)

def draw_button(img, btn: Button, hovered=False, pressed=False):
    x, y, w, h = btn.rect()
    base = C_BTN_ON if pressed else (C_BTN_HI if hovered else C_BTN)
    cv2.rectangle(img, (x, y), (x + w, y + h), base, -1, cv2.LINE_AA)
    draw_glow_rect(img, (x, y), (x + w, y + h), C_GRID, 1, 6)
    # bottom neon accent
    cv2.line(img, (x, y + h - 2), (x + w, y + h - 2), C_NEON, 2, cv2.LINE_AA)
    # label
    font_scale = 1.05 if len(btn.label) <= 2 else 0.9
    (tw, th), _ = cv2.getTextSize(btn.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2 - 3
    cv2.putText(img, btn.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, C_TEXT, 2, cv2.LINE_AA)

def safe_eval(expr: str) -> str:
    # allow only digits, operators, dot and parentheses
    if not re.fullmatch(r"[0-9\+\-\*/\.\(\)\s]+", expr):
        return "Error"
    try:
        # Evaluate with limited globals/locals
        res = eval(expr, {"__builtins__": None}, {})
        # avoid long floats
        if isinstance(res, float):
            res = round(res, 8)
        return str(res)
    except Exception:
        return "Error"

def get_landmark_xy(lm, w, h, idx):
    return int(lm[idx].x * w), int(lm[idx].y * h)

def norm_dist(lm, i, j):
    dx = lm[i].x - lm[j].x
    dy = lm[i].y - lm[j].y
    return math.hypot(dx, dy)

# ------------------- State -----------------------
expr = ""
last_press_ms = 0
hover_target = None
finger_hist = deque(maxlen=SMOOTH_WIN)  # for smoothing index fingertip
click_armed = False  # for pinch mode

# ------------------- Video -----------------------
cap = cv2.VideoCapture(CAM_INDEX)
if FRAME_W and FRAME_H:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Warm up one frame to get actual size
ok, frame0 = cap.read()
if not ok:
    raise RuntimeError("Camera frame read failed")
H, W = frame0.shape[:2]

# ------------------- Main Loop -------------------
try:
    # ------------------- In Main Loop -------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        # Create canvas
        canvas = np.full_like(frame, 0)
        canvas[:] = C_BG

        # Place webcam feed into the left side (black blank area)
        cam_resized = cv2.resize(frame, (PAD_LEFT - 80, FRAME_H - 100))
        canvas[80:80+cam_resized.shape[0], 40:40+cam_resized.shape[1]] = cam_resized

        # Panels
        draw_panel(canvas, 40, 30, DISPLAY_W, DISPLAY_H, fill=(14, 14, 14))  # top neon box
        draw_panel(canvas, PAD_LEFT - 30, PAD_TOP - 30,
                   4 * BTN_W + 3 * BTN_GAP_X + 60,
                   5 * BTN_H + 4 * BTN_GAP_Y + 60,
                   fill=(12, 12, 16))

        # Expression shown INSIDE the neon box
        show_expr = expr[-32:] if len(expr) > 32 else expr
        cv2.putText(canvas, show_expr if show_expr else " ",
                    (DISPLAY_X + 18, DISPLAY_Y + DISPLAY_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_ACCENT, 2, cv2.LINE_AA)

        # Title (moved slightly down to avoid overlap with neon box)
        cv2.putText(canvas, "Neon Gesture Calculator",
                    (PAD_LEFT - 20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_ACCENT, 2, cv2.LINE_AA)

        # Draw all buttons
        for b in BUTTONS:
            draw_button(canvas, b, hovered=False, pressed=False)

        # Hand tracking
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        now_ms = int(time.time() * 1000)
        pressed_btn = None
        cursor = None

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = hand.landmark

            # Index fingertip + thumb tip
            ix, iy = get_landmark_xy(lm, W, H, 8)
            tx, ty = get_landmark_xy(lm, W, H, 4)
            finger_hist.append((ix, iy))
            sx = int(np.mean([p[0] for p in finger_hist]))
            sy = int(np.mean([p[1] for p in finger_hist]))
            cursor = (sx, sy)

            # Draw hand mesh faintly
            mp_draw.draw_landmarks(canvas, hand, mp_hands.HAND_CONNECTIONS,
                                   landmark_drawing_spec=mp_draw.DrawingSpec(color=(140, 80, 255), thickness=1, circle_radius=1),
                                   connection_drawing_spec=mp_draw.DrawingSpec(color=(70, 0, 110), thickness=1))

            # Neon cursor
            cv2.circle(canvas, cursor, 10, C_NEON, 2, cv2.LINE_AA)
            cv2.circle(canvas, cursor, 4, C_ACCENT, -1, cv2.LINE_AA)

            # Detect which button is hovered
            hovered_btn = None
            for b in BUTTONS:
                if b.contains(sx, sy):
                    hovered_btn = b
                    break

            # Visual hover
            if hovered_btn:
                draw_button(canvas, hovered_btn, hovered=True, pressed=False)

            # Determine click by mode
            clicked = False

            if CLICK_MODE == "pinch":
                d = norm_dist(lm, 4, 8)  # thumb tip to index tip
                # "armed" when hands are apart; "fire" when pinch below threshold
                if d > PINCH_DIST_THRESH * 1.3:
                    click_armed = True
                if click_armed and d < PINCH_DIST_THRESH and (now_ms - last_press_ms) > PRESS_COOLDOWN:
                    clicked = True
                    click_armed = False

            else:  # hover click
                if hovered_btn:
                    if hover_target is not hovered_btn:
                        hover_target = hovered_btn
                        hovered_btn.hover_t0 = now_ms
                    else:
                        dwell = now_ms - hovered_btn.hover_t0
                        if dwell >= HOVER_HOLD_MS and (now_ms - last_press_ms) > PRESS_COOLDOWN:
                            clicked = True
                            hovered_btn.hover_t0 = now_ms  # reset dwell
                else:
                    hover_target = None

            # If clicked, determine target
            if clicked:
                target = hovered_btn
                if target:
                    pressed_btn = target
                    draw_button(canvas, target, hovered=True, pressed=True)
                    last_press_ms = now_ms

                    # --- Button actions ---
                    lab = target.label
                    if lab == "C":
                        expr = ""
                    elif lab == "DEL":
                        expr = expr[:-1]
                    elif lab == "=":
                        expr = safe_eval(expr)
                    else:
                        expr += lab

        else:
            finger_hist.clear()
            hover_target = None
            click_armed = False

        # Subtle button flash after press
        if pressed_btn:
            x, y, w, h = pressed_btn.rect()
            cv2.rectangle(canvas, (x, y), (x + w, y + h), C_NEON, 2, cv2.LINE_AA)

        # Render
        cv2.imshow("Gesture Calculator (OpenCV + MediaPipe)", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        # quick toggles (optional)
        if key == ord('h'):
            CLICK_MODE = "hover"
        if key == ord('p'):
            CLICK_MODE = "pinch"

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
