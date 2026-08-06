import cv2
import numpy as np
from typing import Dict, List, Tuple


class ObjectDetector:
    """
    Detects suspicious objects and multiple people in interview frame.
    Monitors for phone usage, external materials, and unauthorized participants.
    """
    
    def __init__(self):
        """Initialize object and person detectors"""
        # Load cascade classifiers
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml'
        )
        
        # Warning tracking
        self.warning_count = 0
        self.max_warnings = 5
        
        # Frame counters
        self.multiple_people_frame_count = 0
        self.hand_carrying_object_frame_count = 0  # CHANGED: Single combined counter
        
        # Thresholds
        self.multiple_people_threshold = 90  # ~3 seconds at 30 FPS
        self.hand_carrying_threshold = 60  # ~2 seconds for hand holding object
        self.max_warnings = 5
        
        # Detection parameters
        self.min_people_for_warning = 2  # Trigger warning if 2+ people detected
        self.hand_area_threshold = 500  # Min pixels for hand detection
        self.material_brightness_threshold = 100  # For material detection
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a frame and detect suspicious objects/people
        
        Returns:
            dict: Detection results with keys:
                - success: bool
                - warning_count: int
                - violations: dict
                - people_count: int
                - hand_detected: bool
                - material_detected: bool
                - hand_carrying_object: bool (CRITICAL: only warns if BOTH present)
                - frame_dimensions: tuple
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        output = {
            'success': True,
            'warning_count': self.warning_count,
            'violations': {
                'multiple_people': False,
                'hand_carrying_object': False  # CHANGED: Combined detection
            },
            'people_count': 0,
            'hand_detected': False,
            'material_detected': False,
            'hand_carrying_object': False,
            'frame_dimensions': (h, w),
            'detections': {}
        }
        
        # 1. Detect multiple people
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        people_count = len(faces)
        output['people_count'] = people_count
        
        if people_count >= self.min_people_for_warning:
            output['violations']['multiple_people'] = True
        
        # 2. Detect hands (skin-colored regions)
        # Returns (bool, list of bounding boxes)
        hand_detected, hand_regions = self._detect_hands(frame)
        output['hand_detected'] = hand_detected
        
        # 3. Detect external materials near hands
        # KEY LOGIC: Only check for materials if hand is visible
        material_detected = self._detect_external_materials(frame, gray, hand_regions)
        output['material_detected'] = material_detected
        
        # 4. CRITICAL: Only trigger warning if BOTH hand AND material detected
        #    (i.e., hand carrying object - not just hand alone or object alone)
        if hand_detected and material_detected:
            output['hand_carrying_object'] = True
            output['violations']['hand_carrying_object'] = True
        
        # Update warning counts based on sustained violations
        output['warning_count'] = self._update_warnings(output['violations'])
        
        # Store detection counts
        output['detections'] = {
            'faces': len(faces),
            'hands': 1 if hand_detected else 0,
            'materials': 1 if material_detected else 0,
            'hand_carrying_object': 1 if (hand_detected and material_detected) else 0
        }
        
        return output
    
    def _detect_hands(self, frame: np.ndarray) -> Tuple[bool, List]:
        """Detect hands using skin color detection
        
        Returns:
            (bool, list): (hand_detected, hand_regions) where hand_regions are bounding boxes
            
        This is STRICT hand detection - only returns hands if clear skin regions found
        """
        try:
            # Convert to HSV for skin detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Skin color range - tuned for hand detection
            lower_skin = np.array([0, 10, 60], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Strong morphological operations to get only clear hand regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            hand_regions = []
            
            # STRICT: Only accept clear, significant hand-sized regions
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # STRICT SIZE: 2000-80000 pixels (significant hands only)
                if 2000 < area < 80000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # STRICT: Hand has specific proportions
                    if w > 0 and h > 0:
                        aspect_ratio = w / float(h)
                        # Hands are roughly 0.4 to 2.5 ratio
                        if 0.4 < aspect_ratio < 2.5:
                            # STRICT: Contour must be solid (circularity check)
                            perimeter = cv2.arcLength(contour, True)
                            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                            
                            # Hands have decent circularity (0.3-0.8)
                            if circularity > 0.25:
                                hand_regions.append((x, y, w, h))
            
            hand_detected = len(hand_regions) > 0
            return hand_detected, hand_regions
        except Exception:
            return False, []
    
    def _detect_external_materials(self, frame: np.ndarray, gray: np.ndarray,
                                   hand_regions: List = None) -> bool:
        """Detect objects ONLY if held in hands
        
        CRITICAL LOGIC:
        - If NO hands detected: Return FALSE (no warning)
        - If hands detected: Check ONLY within hand regions for held objects
        - Object must be in direct contact/near hand to count as "held"
        
        Args:
            frame: BGR frame
            gray: Grayscale frame
            hand_regions: List of hand bounding boxes
        """
        try:
            # CRITICAL: If no hands detected, return FALSE immediately
            if not hand_regions or len(hand_regions) == 0:
                return False  # NO OBJECT DETECTION WITHOUT HANDS
            
            h, w = frame.shape[:2]
            
            # For each detected hand, check if it's holding an object
            for hx, hy, hw, hh in hand_regions:
                # STRICT: Only search WITHIN the hand region and immediate surroundings
                # Not 150px around, but TIGHT: 50px margin only
                search_x1 = max(0, hx - 50)
                search_y1 = max(0, hy - 50)
                search_x2 = min(w, hx + hw + 50)
                search_y2 = min(h, hy + hh + 50)
                
                search_region = gray[search_y1:search_y2, search_x1:search_x2]
                
                if search_region.size == 0:
                    continue
                
                # STRATEGY 1: Detect high-contrast edges (objects in hand typically have edges)
                edges = cv2.Canny(search_region, 50, 150)
                
                # Count significant edges in hand region
                edge_pixels = np.sum(edges > 0)
                edge_ratio = edge_pixels / search_region.size
                
                # High edge density suggests object being held
                # STRICT threshold: >20% of region has edges
                if edge_ratio > 0.2:
                    # Verify it's not just hand edges - check for distinct bright/dark patches
                    # Objects held typically have high local contrast
                    region_std = search_region.std()
                    region_mean = search_region.mean()
                    
                    # High contrast (std > 35) AND significant edge content = held object
                    if region_std > 35:
                        return True  # Object held in this hand
                
                # STRATEGY 2: Detect solid rectangular regions (phones, papers in hand)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                edges_dilated = cv2.dilate(edges, kernel, iterations=2)
                
                contours, _ = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # STRICT: Object in hand must be significant (1500+ pixels)
                    if 1500 < area < (search_region.shape[0] * search_region.shape[1] * 0.7):
                        # Check if rectangular (phones, papers are rectangular)
                        epsilon = 0.03 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Must have 4+ vertices (rectangular)
                        if len(approx) >= 4:
                            # Verify brightness difference (phone/paper different from skin)
                            x_local, y_local, w_box, h_box = cv2.boundingRect(contour)
                            roi = search_region[y_local:y_local+h_box, x_local:x_local+w_box]
                            
                            if roi.size > 0:
                                roi_mean = roi.mean()
                                # Object must be significantly different from hand color
                                # Either very bright (paper: >150) or very dark (phone: <80)
                                if roi_mean > 150 or roi_mean < 80:
                                    return True  # Clear object in hand
            
            # No objects found in any hand region
            return False
        
        except Exception:
            return False
    
    def _update_warnings(self, violations: Dict) -> int:
        """Update warning count based on sustained violations
        
        Only triggers warning for:
        1. Multiple people detected (2+ people)
        2. Hand carrying object (hand + material detected together)
        """
        
        # Track multiple people
        if violations['multiple_people']:
            self.multiple_people_frame_count += 1
            if self.multiple_people_frame_count >= self.multiple_people_threshold:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                self.multiple_people_frame_count = 0
        else:
            self.multiple_people_frame_count = 0
        
        # Track hand carrying object (ONLY when both hand AND material present)
        if violations['hand_carrying_object']:
            self.hand_carrying_object_frame_count += 1
            if self.hand_carrying_object_frame_count >= self.hand_carrying_threshold:
                if self.warning_count < self.max_warnings:
                    self.warning_count += 1
                self.hand_carrying_object_frame_count = 0
        else:
            self.hand_carrying_object_frame_count = 0
        
        return self.warning_count
    
    def reset(self):
        """Reset detector state"""
        self.warning_count = 0
        self.multiple_people_frame_count = 0
        self.hand_carrying_object_frame_count = 0
