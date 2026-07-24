import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"

import cv2
import mediapipe as mp
import pyttsx3
import math
import time
import json
from datetime import datetime

print("=" * 50)
print("SIGNSPEAK PRO - DELUXE EDITION")
print("=" * 50)

# Try to open camera with DirectShow
print("Attempting to open camera with DirectShow...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Failed with DirectShow, trying default...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
    print("\nPlease check your camera connection")
    input("\nPress Enter to exit...")
    exit()

print("Camera opened successfully!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# SETUP MEDIAPIPE
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2)
    detector = HandLandmarker.create_from_options(options)
    print("Hand landmark model loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    print("   Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    input("Press Enter to exit...")
    exit()

try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    print("Text-to-speech initialized")
except:
    print("Text-to-speech failed, continuing without voice")
    engine = None

# MENU PRICES
menu = {
    "Burger": 5.99,
    "Pizza": 8.99,
    "Regular Water": 1.99,
    "Distilled Water": 2.49,
    "Fries": 3.49,
    "Ice Cream": 3.99,
    "Salad": 4.99,
    "Cookie": 1.99,
    "Donut": 2.49,
    "Coke": 2.49,
    "Pepsi": 2.49,
    "Sprite": 2.49
}

# COMBO DEALS
combos = {
    "Burger Combo": {
        "items": ["Burger", "Fries", "Regular Water"],
        "price": 10.99,
        "savings": 2.48
    },
    "Pizza Combo": {
        "items": ["Pizza", "Salad", "Coke"],
        "price": 13.99,
        "savings": 2.48
    },
    "Dessert Combo": {
        "items": ["Ice Cream", "Cookie"],
        "price": 4.99,
        "savings": 0.99
    },
    "Drink Combo": {
        "items": ["Coke", "Fries"],
        "price": 4.99,
        "savings": 0.99
    }
}

# STATE VARIABLES
order_items = []
order_list = []
last_spoken = ""
z_path = []
COOLDOWN = 1.5
last_item_time = 0
last_remove_time = 0
frame_count = 0
total_price = 0
burger_hands_detected = False
burger_start_time = 0
TAX_RATE = 0.08

waiting_for_selection = False
selection_type = None
selection_start_time = 0
pending_item = None
quantity_count = 0
order_history = []
history_file = "order_history.json"

cross_detected = False
cross_start_time = 0
cross_removed = False

z_stage = 0
z_completed = False

# LANDMARK INDEXES
INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
MIDDLE_TIP = 12
MIDDLE_PIP = 10
MIDDLE_MCP = 9
RING_TIP = 16
RING_PIP = 14
RING_MCP = 13
PINKY_TIP = 20
PINKY_PIP = 18
PINKY_MCP = 17
WRIST = 0

# ─────────────────────────────────────────────
#  IMPROVED GESTURE HELPERS
# ─────────────────────────────────────────────

def finger_extended(tip, pip, mcp):
    """
    A finger is extended when its tip is further from the wrist than its MCP joint.
    Uses 3D-ish distance along the finger axis for robustness to hand tilt.
    """
    # Compare tip-to-MCP distance vs pip-to-MTC: tip must be clearly beyond pip
    tip_to_mcp = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
    pip_to_mcp = math.sqrt((pip.x - mcp.x)**2 + (pip.y - mcp.y)**2)
    return tip_to_mcp > pip_to_mcp * 1.2   # 20% clearance buffer


def finger_curled(tip, pip, mcp):
    return not finger_extended(tip, pip, mcp)


def thumb_extended(hand_lms, handedness_label):
    """
    Thumb extended = tip is to the LEFT of IP for RIGHT hand,
    to the RIGHT of IP for LEFT hand.
    Uses the handedness label from MediaPipe.
    """
    tip = hand_lms[THUMB_TIP]
    ip  = hand_lms[THUMB_IP]
    if handedness_label == "Right":
        return tip.x < ip.x   # mirror-flipped because we flip the frame
    else:
        return tip.x > ip.x


def get_finger_states(lms, handedness_label):
    """Returns (thumb, index, middle, ring, pinky) booleans."""
    t  = thumb_extended(lms, handedness_label)
    i  = finger_extended(lms[INDEX_TIP],  lms[INDEX_PIP],  lms[INDEX_MCP])
    m  = finger_extended(lms[MIDDLE_TIP], lms[MIDDLE_PIP], lms[MIDDLE_MCP])
    r  = finger_extended(lms[RING_TIP],   lms[RING_PIP],   lms[RING_MCP])
    p  = finger_extended(lms[PINKY_TIP],  lms[PINKY_PIP],  lms[PINKY_MCP])
    return t, i, m, r, p


def get_finger_count(lms, handedness_label):
    """Count extended fingers (thumb NOT included)."""
    _, i, m, r, p = get_finger_states(lms, handedness_label)
    return sum([i, m, r, p])


def thumb_index_distance(lms):
    return math.sqrt(
        (lms[THUMB_TIP].x - lms[INDEX_TIP].x)**2 +
        (lms[THUMB_TIP].y - lms[INDEX_TIP].y)**2
    )


def get_gesture_name(lms, handedness_label):
    """
    Detect gesture. More-specific patterns are checked FIRST to avoid
    ambiguous early returns.
    """
    t, i, m, r, p = get_finger_states(lms, handedness_label)
    dist = thumb_index_distance(lms)

    # ── SALAD: all 5 fingers open ──────────────────────────────────────────
    if t and i and m and r and p:
        return "Salad"

    # ── COMBO: thumbs-up (thumb only, all others curled) ──────────────────
    if t and not i and not m and not r and not p:
        return "Combo"

    # ── ICE CREAM: L-shape (thumb + index, others curled) ─────────────────
    if t and i and not m and not r and not p:
        return "Ice Cream"

    # ── PIZZA: peace sign WITH thumb curled (checked before Fries) ────────
    # Thumb must be clearly curled inward
    if not t and i and m and not r and not p:
        # distinguish pizza (thumb curled) from fries (thumb free)
        # use thumb tip vs thumb MCP: if tip is close to palm, it's curled
        thumb_tip = lms[THUMB_TIP]
        index_mcp = lms[INDEX_MCP]
        thumb_palm_dist = math.sqrt(
            (thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2
        )
        if thumb_palm_dist < 0.12:
            return "Pizza"
        else:
            return "Fries"

    # ── SODA: 4 fingers (no thumb) ────────────────────────────────────────
    if not t and i and m and r and p:
        return "Soda"

    # ── WATER: 3 fingers (index+middle+ring, no thumb, no pinky) ──────────
    if not t and i and m and r and not p:
        return "Water"

    # ── COOKIE: finger-gun / circle — thumb+index close, others curled ────
    if not i and not m and not r and not p:
        if dist < 0.07:
            return "Cookie"

    # ── DONUT: O shape — thumb+index close-ish, others curled ────────────
    if not m and not r and not p:
        if 0.04 < dist < 0.13:
            return "Donut"

    return None


def is_c_shape_hand(lms, handedness_label):
    """All four fingers curled, thumb free (C shape)."""
    _, i, m, r, p = get_finger_states(lms, handedness_label)
    return not i and not m and not r and not p


def is_burger_gesture(h1, h2, hl1, hl2):
    return is_c_shape_hand(h1, hl1) and is_c_shape_hand(h2, hl2)


def are_fingers_crossed(h1, h2):
    """Both index fingers extended AND tips are very close together."""
    i1 = finger_extended(h1[INDEX_TIP], h1[INDEX_PIP], h1[INDEX_MCP])
    i2 = finger_extended(h2[INDEX_TIP], h2[INDEX_PIP], h2[INDEX_MCP])
    if not (i1 and i2):
        return False
    t1 = h1[INDEX_TIP]
    t2 = h2[INDEX_TIP]
    dist = math.sqrt((t1.x - t2.x)**2 + (t1.y - t2.y)**2)
    return dist < 0.12

# ─────────────────────────────────────────────
#  Z-DRAWING HELPERS  (unchanged logic, same as before)
# ─────────────────────────────────────────────

def detect_z_progress(path):
    global z_stage
    if len(path) < 5:
        return 0

    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)

    if x_range < 50 or y_range < 40:
        return z_stage

    if z_stage == 0 and len(path) > 8:
        first_part = path[:min(12, len(path))]
        y_variance = max(p[1] for p in first_part) - min(p[1] for p in first_part)
        x_progress = abs(first_part[-1][0] - first_part[0][0])
        if y_variance < 30 and x_progress > 50:
            z_stage = 1
            print("Z Stage 1: First horizontal line detected!")

    elif z_stage == 1 and len(path) > 16:
        mid_part = path[8:min(24, len(path))]
        if len(mid_part) > 5:
            x_change = abs(mid_part[-1][0] - mid_part[0][0])
            y_change = abs(mid_part[-1][1] - mid_part[0][1])
            if x_change > 30 and y_change > 40:
                z_stage = 2
                print("Z Stage 2: Diagonal detected!")

    elif z_stage == 2 and len(path) > 25:
        last_part = path[-12:]
        y_variance = max(p[1] for p in last_part) - min(p[1] for p in last_part)
        if y_variance < 30:
            z_stage = 3
            print("Z Stage 3: Z complete!")
            return 3

    return z_stage


def reset_z():
    global z_path, z_stage, z_completed
    z_path = []
    z_stage = 0
    z_completed = False

# ─────────────────────────────────────────────
#  ORDER HELPERS
# ─────────────────────────────────────────────

def update_total():
    global total_price
    total_price = sum(price for _, price in order_items)

def calculate_tax():
    return total_price * TAX_RATE

def calculate_grand_total():
    return total_price + calculate_tax()

def add_to_order(item_name, quantity=1):
    global last_item_time, order_items, order_list, total_price
    if item_name in menu:
        for _ in range(quantity):
            order_items.append((item_name, menu[item_name]))
            order_list.append(item_name)
        update_total()
        print(f"Added: {quantity}x {item_name} - Total: ${total_price:.2f}")
        return True
    return False

def add_combo(combo_name):
    global last_item_time, order_items, order_list, total_price
    if combo_name in combos:
        combo = combos[combo_name]
        for item in combo["items"]:
            order_items.append((item, menu[item]))
            order_list.append(item)
        update_total()
        print(f"Added Combo: {combo_name} - Saved ${combo['savings']:.2f}!")
        return True
    return False

def remove_last_item():
    global order_items, order_list, total_price
    if order_items:
        removed = order_items.pop()
        order_list.pop()
        update_total()
        print(f"Removed: {removed[0]} - Total: ${total_price:.2f}")
        return removed[0]
    return None

def clear_order():
    global order_items, order_list, total_price, waiting_for_selection
    order_items = []
    order_list = []
    total_price = 0
    waiting_for_selection = False
    print("Order cleared!")

def save_order():
    if not order_items:
        return False
    order_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "items": order_list.copy(),
        "subtotal": total_price,
        "tax": calculate_tax(),
        "total": calculate_grand_total()
    }
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except:
        history = []
    history.append(order_record)
    if len(history) > 50:
        history = history[-50:]
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print("Order saved to history!")
        return True
    except:
        print("Failed to save order")
        return False

def load_order_history():
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except:
        return []

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

print("Ready!")
print("=" * 50)
print("GESTURE GUIDE:")
print("  Fries      : Peace sign (index + middle, thumb free)")
print("  Pizza      : Peace sign + thumb CURLED IN, then draw Z")
print("  Water      : 3 fingers (index+middle+ring)")
print("  Soda       : 4 fingers (all except thumb)")
print("  Combo      : Thumbs up only")
print("  Ice Cream  : L-shape (thumb + index)")
print("  Salad      : Open palm (all 5)")
print("  Cookie     : Thumb+index touching circle")
print("  Donut      : O-shape (thumb+index close)")
print("  Burger     : Both hands C-shape")
print("  Remove     : Cross index fingers from both hands")
print("  c = clear | s = save | h = history | q = quit")
print("=" * 50)

cv2.namedWindow("SignSpeak Pro", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SignSpeak Pro", 800, 600)
cv2.moveWindow("SignSpeak Pro", 100, 100)

show_history = False
history_data = []

while cap.isOpened():
    frame_count += 1
    success, frame = cap.read()
    if not success:
        print("Lost camera connection")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    try:
        result = detector.detect_for_video(mp_image, int(time.time() * 1000))
    except Exception:
        continue

    message = ""
    curr_time = time.time()

    if show_history:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "ORDER HISTORY", (w//2-100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        history_data = load_order_history()
        y_offset = 100
        for order in reversed(history_data[-10:]):
            cv2.putText(frame, f"{order['timestamp']}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Total: ${order['total']:.2f}", (w-150, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 25
            for item in order['items'][-3:]:
                cv2.putText(frame, f"  {item}", (30, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
                y_offset += 20
            y_offset += 10
        cv2.putText(frame, "Press 'h' to close history", (w//2-120, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    elif result.hand_landmarks:
        num_hands = len(result.hand_landmarks)

        # Build handedness labels safely
        hand_labels = []
        if result.handedness:
            for handedness in result.handedness:
                hand_labels.append(handedness[0].display_name)
        else:
            hand_labels = ["Right"] * num_hands

        # Draw landmarks
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        if num_hands == 2:
            h1, h2 = result.hand_landmarks[0], result.hand_landmarks[1]
            hl1 = hand_labels[0] if len(hand_labels) > 0 else "Right"
            hl2 = hand_labels[1] if len(hand_labels) > 1 else "Right"

            # REMOVE GESTURE
            if are_fingers_crossed(h1, h2):
                p1 = (int(h1[INDEX_TIP].x * w), int(h1[INDEX_TIP].y * h))
                p2 = (int(h2[INDEX_TIP].x * w), int(h2[INDEX_TIP].y * h))
                cv2.line(frame, p1, p2, (0, 0, 255), 5)
                cv2.putText(frame, "CROSSED - Uncross to remove", (w//2-150, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if not cross_detected:
                    cross_detected = True
                    cross_start_time = curr_time
                    cross_removed = False
            else:
                if cross_detected and not cross_removed:
                    if (curr_time - cross_start_time) > 0.2:
                        removed = remove_last_item()
                        if removed:
                            message = f"Removed {removed}"
                            cross_removed = True
                cross_detected = False

            # BURGER GESTURE
            if is_burger_gesture(h1, h2, hl1, hl2) and not waiting_for_selection:
                for hand in [h1, h2]:
                    wrist = hand[WRIST]
                    cv2.circle(frame, (int(wrist.x * w), int(wrist.y * h)), 50, (0, 255, 255), 2)
                cv2.putText(frame, "BURGER - Hold to add", (w//2-120, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
                if not burger_hands_detected:
                    burger_hands_detected = True
                    burger_start_time = curr_time
                elif (curr_time - burger_start_time) > 0.8:
                    if (curr_time - last_item_time) > COOLDOWN:
                        if add_to_order("Burger"):
                            message = "Burger added"
                            last_item_time = curr_time
                            burger_hands_detected = False
            else:
                burger_hands_detected = False

        elif num_hands == 1:
            lms = result.hand_landmarks[0]
            hl = hand_labels[0] if hand_labels else "Right"

            detected_gesture = get_gesture_name(lms, hl)
            finger_count = get_finger_count(lms, hl)

            # Debug display
            cv2.putText(frame, f"Fingers: {finger_count}  Gesture: {detected_gesture or '---'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # ── QUANTITY SELECTION ─────────────────────────────────────────
            if waiting_for_selection == "quantity":
                cv2.rectangle(frame, (w//2-160, h//2-90), (w//2+160, h//2+60), (20, 20, 20), -1)
                cv2.putText(frame, f"QUANTITY FOR: {pending_item}", (w//2-140, h//2-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, "Show 1-5 fingers", (w//2-130, h//2-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Current: {finger_count}", (w//2-130, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                timeout_left = max(0, 5 - int(curr_time - selection_start_time))
                cv2.putText(frame, f"Timeout: {timeout_left}s", (w//2-130, h//2+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
                if (curr_time - selection_start_time) > 5:
                    waiting_for_selection = False
                    message = "Cancelled"
                elif 1 <= finger_count <= 5:
                    if add_to_order(pending_item, finger_count):
                        message = f"Added {finger_count}x {pending_item}"
                        waiting_for_selection = False
                        last_item_time = curr_time

            # ── COMBO SELECTION ────────────────────────────────────────────
            elif waiting_for_selection == "combo":
                combo_list = list(combos.keys())
                cv2.rectangle(frame, (w//2-160, h//2-90), (w//2+160, h//2+90), (20, 20, 20), -1)
                cv2.putText(frame, "SELECT COMBO:", (w//2-70, h//2-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                for i, cname in enumerate(combo_list):
                    color = (0, 255, 0) if (i + 1) == finger_count else (150, 150, 150)
                    cv2.putText(frame,
                                f"{i+1}. {cname}  ${combos[cname]['price']:.2f}  (save ${combos[cname]['savings']:.2f})",
                                (w//2-140, h//2-20 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                if (curr_time - selection_start_time) > 5:
                    waiting_for_selection = False
                    message = "Cancelled"
                elif 1 <= finger_count <= len(combo_list):
                    selected_combo = combo_list[finger_count - 1]
                    if add_combo(selected_combo):
                        message = f"Added {selected_combo}!"
                        waiting_for_selection = False
                        last_item_time = curr_time

            # ── WATER SELECTION ────────────────────────────────────────────
            elif waiting_for_selection == "water":
                cv2.rectangle(frame, (w//2-160, h//2-80), (w//2+160, h//2+60), (20, 20, 20), -1)
                cv2.putText(frame, "SELECT WATER TYPE:", (w//2-120, h//2-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                col1 = (0, 255, 0) if finger_count == 1 else (150, 150, 150)
                col2 = (0, 255, 0) if finger_count == 2 else (150, 150, 150)
                cv2.putText(frame, "1 finger  = Regular Water  $1.99", (w//2-140, h//2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col1, 1)
                cv2.putText(frame, "2 fingers = Distilled Water $2.49", (w//2-140, h//2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col2, 1)
                cv2.putText(frame, f"Showing: {finger_count} finger(s)", (w//2-140, h//2+48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                if (curr_time - selection_start_time) > 5:
                    waiting_for_selection = False
                    message = "Cancelled"
                elif finger_count == 1:
                    add_to_order("Regular Water")
                    message = "Regular Water added"
                    waiting_for_selection = False
                    last_item_time = curr_time
                elif finger_count == 2:
                    add_to_order("Distilled Water")
                    message = "Distilled Water added"
                    waiting_for_selection = False
                    last_item_time = curr_time

            # ── SODA SELECTION ─────────────────────────────────────────────
            elif waiting_for_selection == "soda":
                soda_options = ["Coke", "Pepsi", "Sprite"]
                cv2.rectangle(frame, (w//2-160, h//2-80), (w//2+160, h//2+80), (20, 20, 20), -1)
                cv2.putText(frame, "SELECT SODA:", (w//2-70, h//2-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                for i, soda in enumerate(soda_options):
                    color = (0, 255, 0) if (i + 1) == finger_count else (150, 150, 150)
                    cv2.putText(frame, f"{i+1}. {soda}  ${menu[soda]:.2f}",
                                (w//2-130, h//2-10 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if (curr_time - selection_start_time) > 5:
                    waiting_for_selection = False
                    message = "Cancelled"
                elif 1 <= finger_count <= 3:
                    sel = soda_options[finger_count - 1]
                    add_to_order(sel)
                    message = f"{sel} added"
                    waiting_for_selection = False
                    last_item_time = curr_time

            # ── PIZZA GESTURE (Z drawing) ──────────────────────────────────
            elif detected_gesture == "Pizza" and not waiting_for_selection:
                curr_p = (int(lms[INDEX_TIP].x * w), int(lms[INDEX_TIP].y * h))
                z_path.append(curr_p)
                if len(z_path) > 60:
                    z_path.pop(0)

                for i in range(1, len(z_path)):
                    color = [(0, 255, 255), (255, 0, 255), (0, 255, 255), (0, 255, 0)][z_stage]
                    cv2.line(frame, z_path[i-1], z_path[i], color, 3)

                detect_z_progress(z_path)

                stage_labels = [
                    "1. Draw HORIZONTAL  →",
                    "2. Draw DIAGONAL    ↘",
                    "3. Draw HORIZONTAL  →",
                    "Z COMPLETE!  Adding pizza..."
                ]
                cv2.putText(frame, "PIZZA - Draw Z shape!", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, stage_labels[z_stage], (50, 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if z_stage == 3 else (0, 255, 255), 2)

                if z_stage == 3 and not z_completed:
                    if (curr_time - last_item_time) > COOLDOWN:
                        pending_item = "Pizza"
                        waiting_for_selection = "quantity"
                        selection_start_time = curr_time
                        z_completed = True
            else:
                if z_path:
                    reset_z()

            # ── SINGLE-GESTURE ITEMS ───────────────────────────────────────
            single_items = ["Fries", "Ice Cream", "Salad", "Cookie", "Donut"]
            if not waiting_for_selection and detected_gesture in single_items:
                cv2.putText(frame, f"{detected_gesture} - Show 1-5 fingers for quantity",
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                if (curr_time - last_item_time) > COOLDOWN:
                    pending_item = detected_gesture
                    waiting_for_selection = "quantity"
                    selection_start_time = curr_time

            # ── WATER TRIGGER ──────────────────────────────────────────────
            elif not waiting_for_selection and detected_gesture == "Water":
                cv2.putText(frame, "WATER - Show 1 or 2 fingers",
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 150, 255), 2)
                if (curr_time - last_item_time) > COOLDOWN:
                    waiting_for_selection = "water"
                    selection_start_time = curr_time

            # ── SODA TRIGGER ───────────────────────────────────────────────
            elif not waiting_for_selection and detected_gesture == "Soda":
                cv2.putText(frame, "SODA - Choose 1-3",
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 120, 0), 2)
                if (curr_time - last_item_time) > COOLDOWN:
                    waiting_for_selection = "soda"
                    selection_start_time = curr_time

            # ── COMBO TRIGGER ──────────────────────────────────────────────
            elif not waiting_for_selection and detected_gesture == "Combo":
                cv2.putText(frame, "COMBO DEALS - Show 1-4 fingers",
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
                if (curr_time - last_item_time) > COOLDOWN:
                    waiting_for_selection = "combo"
                    selection_start_time = curr_time

    # ── UI OVERLAY ─────────────────────────────────────────────────────────
    if not show_history:
        # Right panel - order
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-285, 5), (w-5, h-5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        cv2.putText(frame, "YOUR ORDER", (w-270, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.line(frame, (w-270, 48), (w-18, 48), (255, 255, 0), 1)

        y_offset = 78
        if order_list:
            item_counts = {}
            for item in order_list:
                item_counts[item] = item_counts.get(item, 0) + 1
            for idx, (item, count) in enumerate(item_counts.items()):
                price = menu.get(item, 0)
                line = f"{idx+1}. {item} x{count}  ${price*count:.2f}"
                cv2.putText(frame, line, (w-270, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
                y_offset += 26
                if y_offset > h - 130:
                    break
            cv2.line(frame, (w-270, y_offset+4), (w-18, y_offset+4), (255, 255, 0), 1)
            cv2.putText(frame, f"Subtotal: ${total_price:.2f}", (w-270, y_offset+24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, f"Tax (8%): ${calculate_tax():.2f}", (w-270, y_offset+44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, f"TOTAL: ${calculate_grand_total():.2f}", (w-270, y_offset+68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            cv2.putText(frame, f"Items: {len(order_list)}", (w-270, y_offset+90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "No items yet", (w-270, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        # Left panel - gesture guide
        cv2.rectangle(frame, (8, 8), (275, 310), (0, 0, 0), -1)
        cv2.putText(frame, "GESTURE GUIDE:", (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        guide = [
            ("Fries",      "Peace sign"),
            ("Pizza",      "Peace + thumb in + Z"),
            ("Water",      "3 fingers -> 1 or 2"),
            ("Soda",       "4 fingers -> 1,2,3"),
            ("Combo",      "Thumbs up -> 1-4"),
            ("Ice Cream",  "L-shape"),
            ("Salad",      "Open palm"),
            ("Cookie",     "Pinch circle"),
            ("Donut",      "O-shape"),
            ("Burger",     "Both hands C-shape"),
            ("Remove",     "Cross index fingers"),
            ("c/s/h/q",    "clear/save/hist/quit"),
        ]
        yg = 52
        for label, desc in guide:
            color = (0, 0, 230) if label == "Remove" else (190, 190, 190)
            cv2.putText(frame, f"{label}: {desc}", (18, yg),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)
            yg += 19

        # Status message
        if message:
            cv2.putText(frame, f">> {message}", (w//2-120, h-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        cv2.putText(frame, "c=clear  s=save  h=history  q=quit",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    cv2.imshow("SignSpeak Pro", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('c'):
        clear_order()
        if engine:
            engine.say("Order cleared")
            engine.runAndWait()
    elif key == ord('s'):
        if save_order():
            message = "Order saved!"
            if engine:
                engine.say("Order saved")
                engine.runAndWait()
    elif key == ord('h'):
        show_history = not show_history
        if show_history:
            history_data = load_order_history()
            print(f"Showing {len(history_data)} past orders")

# CLEANUP
print("\n" + "=" * 50)
print("FINAL ORDER SUMMARY:")
print("=" * 50)
if order_list:
    item_counts = {}
    for item in order_list:
        item_counts[item] = item_counts.get(item, 0) + 1
    for item, count in item_counts.items():
        print(f"  {count}x {item}  -  ${menu.get(item,0)*count:.2f}")
    print("-" * 50)
    print(f"Subtotal : ${total_price:.2f}")
    print(f"Tax (8%) : ${calculate_tax():.2f}")
    print(f"TOTAL    : ${calculate_grand_total():.2f}")
    print(f"Items    : {len(order_list)}")
else:
    print("No items ordered.")
print("=" * 50)

if 'detector' in locals():
    detector.close()
cap.release()
cv2.destroyAllWindows()
print("SignSpeak Pro closed.")