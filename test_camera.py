import cv2
import mediapipe as mp
import math
import time

print("=" * 60)
print("SIGNSPEAK PRO - WITH SUB-MENUS")
print("=" * 60)

# ============================================
# CAMERA SETUP
# ============================================
print("Opening camera...")

# Try multiple methods to open camera
cap = None

# Try different backends
for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF]:
    cap = cv2.VideoCapture(0, backend)
    if cap and cap.isOpened():
        print("✅ Camera opened!")
        break
    cap = None

# Try different indexes
if not cap:
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap and cap.isOpened():
            print(f"✅ Camera {i} opened!")
            break
        cap = None

if not cap or not cap.isOpened():
    print("\n❌ Camera not found!")
    print("\nQUICK FIXES:")
    print("1. Close Zoom, Teams, Skype")
    print("2. Check Windows Settings > Privacy > Camera")
    print("3. Restart your computer")
    print("\nPress Enter to exit...")
    input()
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("✅ Camera ready!")

# ============================================
# MEDIAPIPE SETUP
# ============================================
print("Loading hand detector...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
print("✅ Hand detector ready!")

# ============================================
# MENU WITH SUB-OPTIONS
# ============================================
menu = {
    "🍔 Burger": 5.99,
    "🍟 Fries": 3.49,
    "🍕 Pizza": 8.99,
    "🍦 Ice Cream": 3.99,
    "🥗 Salad": 4.99,
    "🍪 Cookie": 1.99,
    "🍩 Donut": 2.49
}

# Sub-menu options
sub_menus = {
    "💧 Water": {
        "1️⃣ Regular": 1.99,
        "2️⃣ Distilled": 2.49,
        "3️⃣ Sparkling": 2.99
    },
    "🥤 Soda": {
        "1️⃣ Coke": 2.49,
        "2️⃣ Pepsi": 2.49,
        "3️⃣ Sprite": 2.49,
        "4️⃣ Fanta": 2.49
    },
    "🍔 Burger": {
        "1️⃣ Classic": 5.99,
        "2️⃣ Cheese": 6.99,
        "3️⃣ Double": 8.99,
        "4️⃣ Veggie": 5.99
    },
    "🍕 Pizza": {
        "1️⃣ Margherita": 8.99,
        "2️⃣ Pepperoni": 10.99,
        "3️⃣ Veggie": 9.99,
        "4️⃣ Hawaiian": 10.99
    },
    "🍦 Ice Cream": {
        "1️⃣ Vanilla": 3.99,
        "2️⃣ Chocolate": 3.99,
        "3️⃣ Strawberry": 3.99,
        "4️⃣ Mint": 4.49
    }
}

# Order tracking
order = []
total = 0
last_add_time = 0
COOLDOWN = 1.5

# Sub-menu state
waiting_for_selection = False
selection_type = None
selection_start_time = 0
pending_category = None

# ============================================
# IMPROVED GESTURE DETECTION
# ============================================
def get_finger_states(hand):
    """Get which fingers are up"""
    thumb_tip = 4
    index_tip = 8
    middle_tip = 12
    ring_tip = 16
    pinky_tip = 20
    
    index_pip = 6
    middle_pip = 10
    ring_pip = 14
    pinky_pip = 18
    
    index_up = hand.landmark[index_tip].y < hand.landmark[index_pip].y
    middle_up = hand.landmark[middle_tip].y < hand.landmark[middle_pip].y
    ring_up = hand.landmark[ring_tip].y < hand.landmark[ring_pip].y
    pinky_up = hand.landmark[pinky_tip].y < hand.landmark[pinky_pip].y
    
    thumb_up = hand.landmark[thumb_tip].x < hand.landmark[thumb_tip - 1].x
    
    return index_up, middle_up, ring_up, pinky_up, thumb_up

def get_finger_count(hand):
    """Count how many fingers are up"""
    index_up, middle_up, ring_up, pinky_up, _ = get_finger_states(hand)
    return sum([index_up, middle_up, ring_up, pinky_up])

def get_gesture(hand):
    """Detect which gesture is being made"""
    index_up, middle_up, ring_up, pinky_up, thumb_up = get_finger_states(hand)
    
    # FRIES = Peace sign (index + middle up)
    if index_up and middle_up and not ring_up and not pinky_up:
        return "🍟 Fries"
    
    # ICE CREAM = L shape (thumb up + index up)
    if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return "🍦 Ice Cream"
    
    # WATER = Three fingers (index, middle, ring up)
    if index_up and middle_up and ring_up and not pinky_up:
        return "💧 Water"
    
    # SODA = Four fingers (all 4 fingers up)
    if index_up and middle_up and ring_up and pinky_up:
        return "🥤 Soda"
    
    # SALAD = Open palm (thumb + all fingers up)
    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "🥗 Salad"
    
    # COOKIE = Circle (thumb and index touching)
    dist = math.sqrt((hand.landmark[4].x - hand.landmark[8].x)**2 + 
                     (hand.landmark[4].y - hand.landmark[8].y)**2)
    if dist < 0.05 and not middle_up and not ring_up and not pinky_up:
        return "🍪 Cookie"
    
    # DONUT = O shape (thumb and index apart)
    if 0.05 < dist < 0.12 and not middle_up and not ring_up and not pinky_up:
        return "🍩 Donut"
    
    # BURGER = C shape (all fingers curled)
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "🍔 Burger"
    
    # Number selection (for sub-menus)
    finger_count = get_finger_count(hand)
    if finger_count > 0:
        return f"{finger_count}️⃣"
    
    return None

def is_burger_gesture(h1, h2):
    """Check if both hands are in C shape"""
    def is_c_shape(hand):
        index_up, middle_up, ring_up, pinky_up, _ = get_finger_states(hand)
        return not index_up and not middle_up and not ring_up and not pinky_up
    return is_c_shape(h1) and is_c_shape(h2)

def are_fingers_crossed(h1, h2):
    """Check if index fingers are crossed"""
    h1_index = h1.landmark[8].y < h1.landmark[6].y
    h2_index = h2.landmark[8].y < h2.landmark[6].y
    
    if not (h1_index and h2_index):
        return False
    
    dist = math.sqrt((h1.landmark[8].x - h2.landmark[8].x)**2 +
                     (h1.landmark[8].y - h2.landmark[8].y)**2)
    return dist < 0.12

# ============================================
# ORDER FUNCTIONS
# ============================================
def add_item(item_name, price):
    global total, last_add_time
    order.append((item_name, price))
    total += price
    last_add_time = time.time()
    print(f"✅ Added: {item_name} - ${price:.2f} - Total: ${total:.2f}")
    return True

def remove_last_item():
    global total
    if order:
        removed, price = order.pop()
        total -= price
        print(f"❌ Removed: {removed} - Total: ${total:.2f}")
        return True
    return False

def clear_order():
    global order, total, waiting_for_selection
    order = []
    total = 0
    waiting_for_selection = False
    print("🗑️ Order cleared!")

# ============================================
# PRESENTATION READY
# ============================================
print("\n" + "=" * 60)
print("🎯 READY! GESTURES WITH SUB-MENUS")
print("=" * 60)
print("\n📖 MAIN GESTURES:")
print("   ✌️  Peace      = 🍟 Fries (adds directly)")
print("   👍  L shape    = 🍦 Ice Cream → choose flavor")
print("   ✋  3 fingers  = 💧 Water → choose type")
print("   🖐️  4 fingers  = 🥤 Soda → choose brand")
print("   🖐️  Open palm  = 🥗 Salad (adds directly)")
print("   👌  Circle     = 🍪 Cookie (adds directly)")
print("   ⭕  O shape    = 🍩 Donut (adds directly)")
print("   👊  C shape    = 🍔 Burger ready")
print("   👐  Both C     = 🍔 Burger → choose style")
print("   ❌  Cross      = Remove last item")
print("\n💡 HOW SUB-MENUS WORK:")
print("   1. Make gesture (Water, Soda, Ice Cream, Burger, Pizza)")
print("   2. Menu appears with options")
print("   3. Show 1-4 fingers to select option")
print("   4. Item added automatically!")
print("\n⌨️  COMMANDS:")
print("   🗑️  'c' = Clear all")
print("   🚪  'q' = Quit")
print("=" * 60)

# Create window
cv2.namedWindow("SignSpeak Pro", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SignSpeak Pro", 1200, 700)
cv2.moveWindow("SignSpeak Pro", 50, 50)

# Variables
current_gesture = None
gesture_start_time = 0
cross_detected = False
cross_start_time = 0
burger_hands = False
burger_start = 0
message = ""
msg_time = 0
frame_count = 0

print("\n🎥 CAMERA ON! Make gestures...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error, retrying...")
        time.sleep(0.1)
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    current_time = time.time()
    
    # Process hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        num_hands = len(results.multi_hand_landmarks)
        
        # ============================================
        # SUB-MENU SELECTION MODE
        # ============================================
        if waiting_for_selection:
            hand = results.multi_hand_landmarks[0]
            finger_count = get_finger_count(hand)
            
            # Draw selection menu
            menu_w = 400
            menu_h = 250
            menu_x = w//2 - menu_w//2
            menu_y = h//2 - menu_h//2
            
            cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), (255, 255, 0), 2)
            
            cv2.putText(frame, f"SELECT {pending_category}", (menu_x + 20, menu_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show options
            options = sub_menus[pending_category]
            y_offset = menu_y + 80
            for i, (option, price) in enumerate(options.items(), 1):
                color = (0, 255, 0) if i == finger_count else (150, 150, 150)
                cv2.putText(frame, f"{i}. {option} - ${price:.2f}", (menu_x + 30, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 35
            
            cv2.putText(frame, f"Show {finger_count} finger(s) to select", (menu_x + 30, menu_y + menu_h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, "Timeout in 5 seconds", (menu_x + 30, menu_y + menu_h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            # Check selection
            if (current_time - selection_start_time) > 5:
                waiting_for_selection = False
                message = "Selection cancelled"
                msg_time = current_time
            elif finger_count > 0:
                options_list = list(options.items())
                if finger_count <= len(options_list):
                    selected_name, selected_price = options_list[finger_count - 1]
                    full_name = f"{pending_category} - {selected_name}"
                    if add_item(full_name, selected_price):
                        message = f"{full_name} added!"
                        msg_time = current_time
                        waiting_for_selection = False
        
        # ============================================
        # TWO HANDS GESTURES
        # ============================================
        elif num_hands == 2:
            h1, h2 = results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]
            
            # REMOVE GESTURE
            if are_fingers_crossed(h1, h2):
                x1, y1 = int(h1.landmark[8].x * w), int(h1.landmark[8].y * h)
                x2, y2 = int(h2.landmark[8].x * w), int(h2.landmark[8].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(frame, "❌ CROSSED - Uncross to remove", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if not cross_detected:
                    cross_detected = True
                    cross_start_time = current_time
            else:
                if cross_detected and (current_time - cross_start_time) > 0.2:
                    if remove_last_item():
                        message = "Removed last item!"
                        msg_time = current_time
                cross_detected = False
            
            # BURGER GESTURE (Both hands C shape)
            if is_burger_gesture(h1, h2):
                cv2.putText(frame, "🍔 BURGER - Hold to see menu", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                for hand in [h1, h2]:
                    cx, cy = int(hand.landmark[0].x * w), int(hand.landmark[0].y * h)
                    cv2.circle(frame, (cx, cy), 60, (0, 255, 255), 2)
                
                if not burger_hands:
                    burger_hands = True
                    burger_start = current_time
                elif (current_time - burger_start) > 1.0:
                    if current_time > last_add_time + COOLDOWN:
                        pending_category = "🍔 Burger"
                        waiting_for_selection = True
                        selection_start_time = current_time
                        message = "Choose burger style!"
                        msg_time = current_time
                    burger_hands = False
            else:
                burger_hands = False
        
        # ============================================
        # ONE HAND GESTURES
        # ============================================
        elif num_hands == 1:
            hand = results.multi_hand_landmarks[0]
            gesture = get_gesture(hand)
            finger_count = get_finger_count(hand)
            
            if gesture:
                # Show detected gesture
                cv2.putText(frame, f"Detected: {gesture}", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # ===== DIRECT ADD (no sub-menu) =====
                if gesture in ["🍟 Fries", "🥗 Salad", "🍪 Cookie", "🍩 Donut"]:
                    # Hold detection for direct add
                    if gesture == current_gesture:
                        hold_time = current_time - gesture_start_time
                        # Progress bar
                        cv2.rectangle(frame, (50, 190), (50 + int(300 * hold_time / 1.0), 210), 
                                     (0, 255, 0), -1)
                        cv2.putText(frame, f"Hold {int(hold_time*100)}%", (50, 185), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        if hold_time > 1.0:
                            if current_time > last_add_time + COOLDOWN:
                                if add_item(gesture, menu[gesture]):
                                    message = f"{gesture} added!"
                                    msg_time = current_time
                            current_gesture = None
                    else:
                        current_gesture = gesture
                        gesture_start_time = current_time
                
                # ===== SUB-MENU ITEMS (Water, Soda, Ice Cream, Pizza) =====
                elif gesture in ["💧 Water", "🥤 Soda", "🍦 Ice Cream", "🍕 Pizza"]:
                    cv2.putText(frame, f"{gesture} - Hold to see options", (50, 190), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    
                    if gesture == current_gesture:
                        hold_time = current_time - gesture_start_time
                        cv2.rectangle(frame, (50, 230), (50 + int(300 * hold_time / 1.0), 250), 
                                     (0, 255, 0), -1)
                        
                        if hold_time > 1.0:
                            if current_time > last_add_time + COOLDOWN:
                                pending_category = gesture
                                waiting_for_selection = True
                                selection_start_time = current_time
                                message = f"Choose {gesture} option!"
                                msg_time = current_time
                            current_gesture = None
                    else:
                        current_gesture = gesture
                        gesture_start_time = current_time
                
                # ===== BURGER READY (single hand C shape) =====
                elif gesture == "🍔 Burger":
                    cv2.putText(frame, "🍔 BURGER READY - Use BOTH hands to order", (50, 190), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    current_gesture = None
                
                # ===== NUMBER SELECTION (for sub-menus) =====
                elif "️⃣" in gesture and waiting_for_selection:
                    # This is handled in the sub-menu section
                    pass
                
                else:
                    current_gesture = None
            else:
                current_gesture = None
    
    # ============================================
    # DISPLAY ORDER (Compact on right)
    # ============================================
    panel_w = 280
    cv2.rectangle(frame, (w-panel_w, 10), (w-10, h-10), (0, 0, 0), -1)
    cv2.rectangle(frame, (w-panel_w, 10), (w-10, h-10), (255, 255, 0), 2)
    
    cv2.putText(frame, "🛒 YOUR ORDER", (w-panel_w+10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    y = 75
    if order:
        # Count and display items
        item_counts = {}
        for item_name, price in order:
            item_counts[item_name] = item_counts.get(item_name, 0) + 1
        
        for item_name, count in list(item_counts.items())[:8]:
            # Find price
            price = next((p for n, p in order if n == item_name), 0)
            text = f"{item_name} x{count}  ${price*count:.2f}"
            cv2.putText(frame, text, (w-panel_w+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y += 28
        
        cv2.line(frame, (w-panel_w+10, y+5), (w-20, y+5), (255, 255, 0), 1)
        cv2.putText(frame, f"TOTAL: ${total:.2f}", (w-panel_w+10, y+35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Items: {len(order)}", (w-panel_w+10, y+60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "No items", (w-panel_w+10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # ============================================
    # GESTURE GUIDE (Compact on left)
    # ============================================
    cv2.rectangle(frame, (10, h-210), (210, h-15), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h-210), (210, h-15), (255, 255, 0), 1)
    cv2.putText(frame, "GESTURES", (15, h-190), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    guide = [
        "✌️ = Fries", "👍 = Ice Cream → menu",
        "✋ = Water → menu", "🖐️4 = Soda → menu",
        "🖐️5 = Salad", "👌 = Cookie",
        "⭕ = Donut", "👊 C = Burger ready",
        "👐 Both C = Burger → menu", "❌ X = Remove"
    ]
    
    gy = h-170
    for g in guide:
        cv2.putText(frame, g, (15, gy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        gy += 18
    
    # ============================================
    # MESSAGE & CONTROLS
    # ============================================
    if message and (current_time - msg_time) < 2:
        cv2.putText(frame, message, (w//2-120, h-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(frame, "'c' Clear | 'q' Quit", (10, h-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Live indicator
    frame_count += 1
    cv2.circle(frame, (w-25, 25), 6, (0, 255, 0), -1)
    cv2.putText(frame, "LIVE", (w-55, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imshow("SignSpeak Pro", frame)
    
    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        clear_order()
        message = "Order cleared!"
        msg_time = current_time

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("FINAL ORDER SUMMARY")
print("=" * 60)
if order:
    item_counts = {}
    for item_name, price in order:
        item_counts[item_name] = item_counts.get(item_name, 0) + 1
    
    for item_name, count in item_counts.items():
        price = next((p for n, p in order if n == item_name), 0)
        print(f"  {count}x {item_name} - ${price*count:.2f}")
    print("-" * 60)
    print(f"  TOTAL: ${total:.2f}")
    print(f"  Items: {len(order)}")
else:
    print("  No items ordered")
print("=" * 60)

cap.release()
cv2.destroyAllWindows()
print("\n✅ Presentation ready! Good luck! 🎉")