import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters (tweakable)
PINCH_THRESHOLD = 0.03        # normalized distance for pinch detection
CUBE_SIZE = 0.07              # default cube size in normalized units (relative to frame width)
DEPTH_SCALE = 1.5             # scales MediaPipe z to camera Z (higher -> cubes appear further)
FOCAL_LENGTH_RATIO = 1.0      # focal length as ratio of image width (tweak if needed)
MAX_HISTORY = 5               # smoothing for the grabbed point

class Cube:
    def __init__(self, center_norm, size_norm):
        # center_norm: (x,y,z) all in MediaPipe normalized coordinates (x,y in [0,1], z negative)
        self.center = np.array(center_norm, dtype=float)
        self.size = float(size_norm)
        self.color = tuple(np.random.randint(50,230,3).tolist())
        self.rotation = 0.0

    def update_center(self, new_center):
        self.center = np.array(new_center, dtype=float)

def normalized_to_camera_coords(norm, img_w, img_h, focal_ratio=FOCAL_LENGTH_RATIO, depth_scale=DEPTH_SCALE):
    """
    Convert mediapipe normalized landmark (x,y,z) to camera-space coords (X,Y,Z).
    We define:
      - u = x * img_w
      - v = y * img_h
      - z_norm = landmark.z (negative when towards camera typically)
    For pinhole model we need 3D (X,Y,Z) with Z>0 in front of camera.
    We'll set an approximate focal length fx = img_w * focal_ratio.
    We'll convert normalized z to a positive Z by scaling and offset.
    """
    x_n, y_n, z_n = norm
    u = x_n * img_w
    v = y_n * img_h
    # convert z_n (usually negative, small magnitude) to a positive depth value
    # offset so Z isn't zero; tune depth_scale as needed
    Z = ( -z_n ) * img_w * depth_scale + 0.5 * img_w  # the 0.5*img_w is a base distance
    # convert screen px to camera X,Y using simple inverse projection
    fx = img_w * focal_ratio
    X = (u - img_w/2.0) * (Z / fx)
    Y = (v - img_h/2.0) * (Z / fx)
    return np.array([X, Y, Z])

def project_point(XYZ, img_w, img_h, focal_ratio=FOCAL_LENGTH_RATIO):
    """
    Project 3D camera-space point (X,Y,Z) into pixel coords (u,v) using simple pinhole model.
    """
    X, Y, Z = XYZ
    if Z <= 1e-6:
        Z = 1e-6
    fx = img_w * focal_ratio
    u = (X * fx) / Z + img_w/2.0
    v = (Y * fx) / Z + img_h/2.0
    return int(u), int(v)

def cube_corners_3d(center_norm, size_norm, img_w, img_h):
    """
    Return 8 corners of a cube (in camera coords) centered at center_norm (normalized coords).
    size_norm is cube side in normalized units relative to image width.
    """
    # Convert center to camera coords
    center_cam = normalized_to_camera_coords(center_norm, img_w, img_h)
    # convert normalized size to camera-space half-size:
    # approximate side in pixels:
    side_pixels = size_norm * img_w
    # estimate half-size in camera coords at that distance: h = side_pixels * Z / fx / 2 -> but we want world size,
    # simpler: compute half-size in camera X/Y units using local depth.
    fx = img_w * FOCAL_LENGTH_RATIO
    half = (side_pixels/2.0) * (center_cam[2] / fx)
    # generate cube corners relative to center_cam
    offsets = np.array([
        [-half, -half, -half],
        [ half, -half, -half],
        [ half,  half, -half],
        [-half,  half, -half],
        [-half, -half,  half],
        [ half, -half,  half],
        [ half,  half,  half],
        [-half,  half,  half],
    ])
    corners = center_cam[np.newaxis,:] + offsets
    return corners

def draw_cube_on_image(img, corners3d, color=(0,255,0), thickness=2):
    """
    Draw wireframe cube: corners3d is 8x3 array in camera coords. Project to pixels and connect edges.
    """
    img_h, img_w = img.shape[:2]
    pts = [project_point(c, img_w, img_h) for c in corners3d]
    # edges between corner indices
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    # draw faces lightly filled for depth cue - compute average Z for face to draw from far to near
    for a,b in edges:
        cv2.line(img, pts[a], pts[b], color, thickness, lineType=cv2.LINE_AA)

def distance_norm(lm1, lm2):
    return np.linalg.norm(np.array(lm1) - np.array(lm2))

def norm_point(lm, img_w, img_h):
    return np.array([lm.x, lm.y, lm.z])

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:

        blocks = []         # list of Cube
        grabbed = None      # index of grabbed block, or None
        history = deque(maxlen=MAX_HISTORY)  # smoothing for grab pos
        last_create_time = 0
        create_cooldown = 0.4  # seconds between creates to avoid duplicates

        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            pinch = False
            pinch_center_norm = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # draw landmarks for debugging
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get thumb tip (4) and index tip (8)
                lm_thumb = hand_landmarks.landmark[4]
                lm_index = hand_landmarks.landmark[8]
                thumb_pt = norm_point(lm_thumb, img_w, img_h)
                index_pt = norm_point(lm_index, img_w, img_h)

                d = distance_norm(thumb_pt[:2], index_pt[:2])  # 2D norm
                # Use combined 3D distance for robustness
                d3 = distance_norm(thumb_pt, index_pt)
                # pinch threshold in normalized units (approx)
                if d3 < PINCH_THRESHOLD:
                    pinch = True
                    pinch_center_norm = ((thumb_pt + index_pt)/2.0)  # np array of normalized [x,y,z]
                    # smoothing
                    history.append(pinch_center_norm)
                    smooth_center = np.mean(np.array(history), axis=0)
                    pinch_center_norm = smooth_center

                # gesture: if pinch just started, create new cube (cooldown)
                current_time = time.time()
                if pinch and (current_time - last_create_time) > create_cooldown:
                    # If there is any block under the pinch (close in 3D), grab it instead of creating
                    grabbed_idx = None
                    for i, b in enumerate(blocks):
                        # compute distance between block center and pinch point in normalized coords by converting block center to camera-space similar scale
                        dd = np.linalg.norm(b.center - pinch_center_norm)
                        if dd < 0.12:  # normalized threshold; tuneable
                            grabbed_idx = i
                            break
                    if grabbed_idx is None:
                        # create new block
                        new_block = Cube(center_norm=pinch_center_norm, size_norm=CUBE_SIZE)
                        blocks.append(new_block)
                        grabbed = len(blocks)-1
                    else:
                        grabbed = grabbed_idx
                    last_create_time = current_time

                # If pinch active and we have a grabbed block, update its center to pinch point
                if pinch and grabbed is not None:
                    blocks[grabbed].update_center(pinch_center_norm)

                # If pinch released, release grabbed block
                if (not pinch) and grabbed is not None:
                    grabbed = None
                    history.clear()

            else:
                history.clear()

            # draw cubes
            for i, b in enumerate(blocks):
                corners = cube_corners_3d(b.center, b.size, img_w, img_h)
                draw_cube_on_image(frame, corners, color=b.color, thickness=2)
                # draw center point label
                center_cam = normalized_to_camera_coords(b.center, img_w, img_h)
                u,v = project_point(center_cam, img_w, img_h)
                label = f"{i}"
                cv2.circle(frame, (u,v), 4, (255,255,255), -1)
                cv2.putText(frame, label, (u+6, v-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # HUD
            cv2.putText(frame, f"Blocks: {len(blocks)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            if grabbed is not None:
                cv2.putText(frame, f"Held: {grabbed}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-6)
            prev_time = now
            cv2.putText(frame, f"{int(fps)} FPS", (10, img_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow("Hand Blocks 3D", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                # clear blocks
                blocks = []
                grabbed = None
                history.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
