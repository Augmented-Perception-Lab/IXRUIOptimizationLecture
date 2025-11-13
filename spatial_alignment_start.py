import os
import cv2
import mediapipe as mp
import numpy as np
import json
import gurobipy as gp

DIR = "imgs"
VIS_THRESH = 0.5
PAD_HEAD = 0.5
PAD_TORSO = 0.3

LOGO_SIZE = 150

def detect_pose(img):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(img)
    return res.pose_landmarks

def draw_pose(img, pose_landmarks, show=False):
    pose_img = img.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            pose_img, pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        if show:
            cv2.imshow("Pose", pose_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return pose_img
    return None

def coord_to_pixels(x, y, W, H):
    return int(round(x * W)), int(round(y * H))

def pad_bbox(bbox, pad):
    x_min, x_max, y_min, y_max = bbox
    x_center = (x_min + x_max) / 2
    x_width = x_max - x_min
    y_center = (y_min + y_max) / 2
    y_height = y_max - y_min
    x_min = x_center - 0.5 * (1 + pad) * x_width
    x_max = x_center + 0.5 * (1 + pad) * x_width
    y_min = y_center - 0.5 * (1 + pad) * y_height
    y_max = y_center + 0.5 * (1 + pad) * y_height
    return x_min, x_max, y_min, y_max

def get_head_bbox(pose_landmarks):
    mp_pose = mp.solutions.pose
    L = mp_pose.PoseLandmark
    head_ids  = [L.NOSE, L.LEFT_EYE, L.RIGHT_EYE, L.LEFT_EAR, L.RIGHT_EAR,
                 L.MOUTH_LEFT, L.MOUTH_RIGHT]
    head_pts  = [ pose_landmarks.landmark[head_id] for head_id in head_ids  ]
    head_pts = [ head_pt for head_pt in head_pts if head_pt.visibility >= VIS_THRESH ]
    head_pts = [ (head_pt.x, head_pt.y) for head_pt in head_pts ]
    # Get bounding box of head
    x_coords = [pt[0] for pt in head_pts]
    y_coords = [pt[1] for pt in head_pts]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, x_max, y_min, y_max

def get_torso_bbox(pose_landmarks):
    mp_pose = mp.solutions.pose
    L = mp_pose.PoseLandmark
    torso_ids  = [L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP]
    torso_pts  = [ pose_landmarks.landmark[torso_id] for torso_id in torso_ids  ]
    torso_pts = [ torso_pt for torso_pt in torso_pts if torso_pt.visibility >= VIS_THRESH ]
    torso_pts = [ (torso_pt.x, torso_pt.y) for torso_pt in torso_pts ]
    x_coords = [pt[0] for pt in torso_pts]
    y_coords = [pt[1] for pt in torso_pts]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, x_max, y_min, y_max

def get_head_torso_bbox(img, show=False):
    pose_landmarks = detect_pose(img)
    if pose_landmarks:
        pose_img = draw_pose(img, pose_landmarks, show=show)
        head_bbox =get_head_bbox(pose_landmarks)
        head_bbox = pad_bbox(head_bbox, PAD_HEAD)
        pose_img = draw_bbox(pose_img, head_bbox, show=show)
        torso_bbox =get_torso_bbox(pose_landmarks)
        torso_bbox = pad_bbox(torso_bbox, PAD_TORSO)
        pose_img = draw_bbox(pose_img, torso_bbox, show=show)
        return head_bbox, torso_bbox
    return None, None

def draw_bbox(img, bbox, show=False):
    bbox_img = img.copy()
    H, W = bbox_img.shape[:2]
    x_min, x_max, y_min, y_max = bbox
    x_min, y_min = coord_to_pixels(x_min, y_min, W, H)
    x_max, y_max = coord_to_pixels(x_max, y_max, W, H)
    cv2.rectangle(bbox_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    if show:
        cv2.imshow("Bbox", bbox_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bbox_img

def overlay_logo(img, logo, x, y):
    result = img.copy()
    h_img, w_img = img.shape[:2]
    h_logo, w_logo = logo.shape[:2]
    
    if logo.shape[2] == 4:
        alpha = logo[:, :, 3] / 255.0
        logo_rgb = logo[:, :, :3]
        
        y1, y2 = max(0, y), min(h_img, y + h_logo)
        x1, x2 = max(0, x), min(w_img, x + w_logo)
        
        logo_y1 = max(0, -y)
        logo_y2 = logo_y1 + (y2 - y1)
        logo_x1 = max(0, -x)
        logo_x2 = logo_x1 + (x2 - x1)
        
        bg_region = result[y1:y2, x1:x2]
        logo_region = logo_rgb[logo_y1:logo_y2, logo_x1:logo_x2]
        alpha_region = alpha[logo_y1:logo_y2, logo_x1:logo_x2]

        if bg_region.shape[0] > 0 and bg_region.shape[1] > 0:
            alpha_3d = alpha_region[:, :, np.newaxis]
            blended = (1 - alpha_3d) * bg_region + alpha_3d * logo_region
            result[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        y1, y2 = max(0, y), min(h_img, y + h_logo)
        x1, x2 = max(0, x), min(w_img, x + w_logo)
        
        logo_y1 = max(0, -y)
        logo_y2 = logo_y1 + (y2 - y1)
        logo_x1 = max(0, -x)
        logo_x2 = logo_x1 + (x2 - x1)
        
        result[y1:y2, x1:x2] = logo[logo_y1:logo_y2, logo_x1:logo_x2]
    
    return result

def load_logos(fname_logos="logos.json"):
    logos = json.load(open(fname_logos, "r"))
    for logo in logos:
        logo_img = cv2.imread(os.path.join(DIR, f"{logo}.png"), cv2.IMREAD_UNCHANGED)
        # resize to 100x100
        logo_img = cv2.resize(logo_img, (LOGO_SIZE, LOGO_SIZE))
        logos[logo]["img"] = logo_img
    
    return logos

def overlay_logos(img, logos): 
    result = img.copy()
    for logo in logos: 
        result = overlay_logo(result, logos[logo]["img"], logos[logo]["x"], logos[logo]["y"])
    return result

def draw_positions(img, x_positions, y_positions):
    img_height, img_width = img.shape[:2]
    img_positions = img.copy()
    for x_pos in x_positions:
        for y_pos in y_positions:
            # draw circle 
            img_positions = cv2.circle(img_positions, 
            (int(x_pos * img_width), int(y_pos * img_height)), 
            int(LOGO_SIZE/2), (0, 0, 255), 2)
    cv2.imshow("Image", img_positions)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_values(img, x_positions, y_positions, x_values, y_values, 
    font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.5, thickness = 2):
    img_height, img_width = img.shape[:2]
    img_values = img.copy()
    num_x_positions = len(x_positions)
    num_y_positions = len(y_positions)
    for xi in range(num_x_positions):
        for yi in range(num_y_positions):
            x_pos = x_positions[xi]
            y_pos = y_positions[yi]
            # draw circle 
            img_values = cv2.circle(img_values, 
            (int(x_pos * img_width), int(y_pos * img_height)), 
            int(LOGO_SIZE/2), (0, 0, 255), 2)
            label = f"{x_values[xi]:.2f},{y_values[yi]:.2f}"
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]
            text_x = int(x_pos * img_width - text_width / 2)
            text_y = int(y_pos * img_height - text_height / 2)
            img_values = cv2.putText(img_values, label, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
    cv2.imshow("Image", img_values)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bbox_overlap(bbox1, bbox2):
    x_min1, x_max1, y_min1, y_max1 = bbox1
    x_min2, x_max2, y_min2, y_max2 = bbox2
    if x_min1 > x_max2 or x_max1 < x_min2 or y_min1 > y_max2 or y_max1 < y_min2:
        return False
    return True

def get_placement_bbox(x, y, x_step, y_step):
    x_min = x - x_step/2
    x_max = x + x_step/2
    y_min = y - y_step/2
    y_max = y + y_step/2
    return x_min, x_max, y_min, y_max

def main():
    fname1 = "david1.png"
    fname2 = "david3.png"

    logos = load_logos()
    
    img1 = cv2.imread(os.path.join(DIR, fname1))
    img1_logos = overlay_logos(img1, logos)
    cv2.imshow("Image 1 with Logo", img1_logos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img2 = cv2.imread(os.path.join(DIR, fname2))

    img2_logos = overlay_logos(img2, logos)
    cv2.imshow("Image 2", img2_logos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    