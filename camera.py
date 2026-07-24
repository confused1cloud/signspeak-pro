import cv2

print("Testing camera connections...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: WORKING")
        ret, frame = cap.read()
        if ret:
            print(f"  - Can read frames: YES")
            cv2.imshow(f'Camera {i}', frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        else:
            print(f"  - Can read frames: NO")
        cap.release()
    else:
        print(f"Camera {i}: NOT WORKING")
print("Test complete")