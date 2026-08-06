import os
import random
import cv2
import numpy as np
import json

class ImageAugmentor:
    def __init__(self, target_count=None):
        self.target_count = target_count

    def augment_brightness(self, img):
        factor = random.uniform(0.6, 1.4)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:,:,2] = hsv[:,:,2] * factor
        hsv[:,:,2][hsv[:,:,2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def augment_contrast(self, img):
        alpha = random.uniform(0.7, 1.3)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    def augment_gamma(self, img):
        gamma = random.uniform(0.6, 1.5)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def augment_gaussian_blur(self, img):
        ksize = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def augment_motion_blur(self, img):
        size = random.choice([3, 5, 7])
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(img, -1, kernel)

    def augment_gaussian_noise(self, img):
        row, col, ch = img.shape
        mean = 0
        var = random.uniform(10, 50)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def augment_jpeg_compression(self, img):
        quality = random.randint(30, 85)
        result, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(enc, 1)

    def augment_rotation(self, img):
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def augment_perspective(self, img):
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dx = random.uniform(0.02, 0.08) * w
        dy = random.uniform(0.02, 0.08) * h
        dst = np.float32([[dx, dy], [w - 1 - dx, dy], [dx, h - 1 - dy], [w - 1 - dx, h - 1 - dy]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


    def simulate_paper_cover(self, img):
        h, w = img.shape[:2]
        cover_type = random.choice(["white_paper", "blackout", "cardboard", "fabric"])
        if cover_type == "white_paper":
            color = (random.randint(225, 250), random.randint(225, 250), random.randint(225, 250))
            # Defocus blur before rectangle
            blurred = cv2.GaussianBlur(img, (99, 99), 0)
            cv2.rectangle(blurred, (0, 0), (w, h), color, -1)
            # Add slight texture noise
            noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
            out = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif cover_type == "blackout":
            out = np.random.randint(4, 15, (h, w, 3), dtype=np.uint8)
        elif cover_type == "cardboard":
            color = (random.randint(80, 130), random.randint(90, 140), random.randint(120, 170))
            out = np.full(img.shape, color, dtype=np.uint8)
            noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
            out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            # Fabric cover (greyish out-of-focus cover)
            color = (random.randint(150, 180), random.randint(150, 180), random.randint(150, 180))
            blurred = cv2.GaussianBlur(img, (77, 77), 0)
            cv2.rectangle(blurred, (0, 0), (w, h), color, -1)
            out = blurred
        return out

    def simulate_hand_cover(self, img):
        h, w = img.shape[:2]
        cover_type = random.choice(["skin", "glove_black", "glove_color", "shadow", "sleeve"])
        
        # Create a soft ellipse mask
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (random.randint(int(w*0.2), int(w*0.8)), random.randint(int(h*0.2), int(h*0.8)))
        axes = (random.randint(int(w*0.3), int(w*0.55)), random.randint(int(h*0.3), int(h*0.55)))
        angle = random.randint(0, 180)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        
        # Soft blurred out-of-focus edges
        mask = cv2.GaussianBlur(mask, (55, 55), 0)
        
        if cover_type == "skin":
            b = random.randint(100, 170)
            g = random.randint(120, 200)
            r = random.randint(180, 255)
            color = (b, g, r)
        elif cover_type == "glove_black":
            color = (random.randint(15, 35), random.randint(15, 35), random.randint(15, 35))
        elif cover_type == "glove_color":
            color = random.choice([
                (random.randint(150, 220), random.randint(80, 120), random.randint(20, 50)), # Blueish
                (random.randint(20, 50), random.randint(20, 50), random.randint(150, 220)), # Redish
            ])
        elif cover_type == "sleeve":
            color = (random.randint(50, 90), random.randint(50, 90), random.randint(50, 90))
        else: # shadow
            color = (random.randint(5, 20), random.randint(5, 20), random.randint(5, 20))
            
        color_img = np.full(img.shape, color, dtype=np.uint8)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0
        
        # Alpha blend the cover onto frame
        blended = (img * (1.0 - mask_3d) + color_img * mask_3d).astype(np.uint8)
        return blended

    def simulate_half_cover(self, img):
        h, w = img.shape[:2]
        out = img.copy()
        direction = random.choice(["top", "bottom", "left", "right"])
        color = random.choice([(10, 10, 10), (220, 220, 220), (120, 130, 140)]) # Blackout, paper or grey
        
        mask = np.zeros((h, w), dtype=np.uint8)
        if direction == "top":
            cv2.rectangle(mask, (0, 0), (w, h // 2), 255, -1)
        elif direction == "bottom":
            cv2.rectangle(mask, (0, h // 2), (w, h), 255, -1)
        elif direction == "left":
            cv2.rectangle(mask, (0, 0), (w // 2, h), 255, -1)
        else:
            cv2.rectangle(mask, (w // 2, 0), (w, h), 255, -1)
            
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        color_img = np.full(img.shape, color, dtype=np.uint8)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0
        
        blended = (img * (1.0 - mask_3d) + color_img * mask_3d).astype(np.uint8)
        return blended

    def simulate_camera_moved(self, img):
        h, w = img.shape[:2]
        # Simulate viewpoint displacement: translation, rotation, scale
        tx = random.uniform(-0.12, 0.12) * w
        ty = random.uniform(-0.12, 0.12) * h
        angle = random.uniform(-10, 10)
        scale = random.uniform(0.92, 1.08)
        
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Warp with replicates to maintain structured scene bounds
        warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return warped

    def augment_single(self, img):
        methods = [
            self.augment_brightness,
            self.augment_contrast,
            self.augment_gamma,
            self.augment_gaussian_blur,
            self.augment_motion_blur,
            self.augment_gaussian_noise,
            self.augment_jpeg_compression,
            self.augment_rotation,
            self.augment_perspective,
            self.simulate_paper_cover,
            self.simulate_hand_cover,
            self.simulate_half_cover,
            self.simulate_camera_moved
        ]
        func = random.choice(methods)
        return func(img), func.__name__

    def process_train_folder(self, train_dir, report_output_path):
        classes = ["Normal", "Tampered"]
        report = {
            "initial_counts": {},
            "final_counts": {},
            "generated_augmentations": {}
        }
        
        # Count initial files
        for cls in classes:
            cls_path = os.path.join(train_dir, cls)
            if not os.path.exists(cls_path):
                report["initial_counts"][cls] = 0
                continue
            files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
            report["initial_counts"][cls] = len(files)

        max_count = max(report["initial_counts"].values()) if report["initial_counts"] else 0
        if self.target_count:
            max_count = self.target_count

        # Apply augmentation to achieve class balance
        for cls in classes:
            cls_path = os.path.join(train_dir, cls)
            if not os.path.exists(cls_path):
                continue
            files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
            current_count = len(files)
            
            # Record final count structure
            report["final_counts"][cls] = current_count
            report["generated_augmentations"][cls] = 0
            
            if current_count < max_count and current_count > 0:
                needed = max_count - current_count
                print(f"[Augmentor] Class '{cls}' has {current_count} files, adding {needed} augmentations...")
                for i in range(needed):
                    src_file = random.choice(files)
                    img_path = os.path.join(cls_path, src_file)
                    try:
                        with open(img_path, 'rb') as f:
                            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    except Exception:
                        img = None
                    if img is None:
                        continue


                    
                    aug_img, method = self.augment_single(img)
                    
                    name_parts = os.path.splitext(src_file)
                    dest_file = f"{name_parts[0]}_aug{i}_{method}{name_parts[1]}"
                    dest_path = os.path.join(cls_path, dest_file)
                    
                    # Unicode-safe write
                    success, buf = cv2.imencode(name_parts[1], aug_img)
                    if success:
                        with open(dest_path, "wb") as f:
                            f.write(buf.tobytes())
                        report["generated_augmentations"][cls] += 1
                        
                report["final_counts"][cls] = current_count + report["generated_augmentations"][cls]
                
        os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
        with open(report_output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"[Augmentor] Augmentation report written to {report_output_path}")

        return report
