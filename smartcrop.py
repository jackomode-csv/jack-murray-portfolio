import cv2, numpy as np, os

# cover-cropped images on 9-teen.html (hero + cascade media). Originals kept for spreads.
IMAGES = [
    "9-teen-images/IMG_9284-10_1.jpg",
    "9-teen-images/IMG_9272-1_1.jpg",
    "9-teen-images/IMG_9279-5_1.jpg",
    "9-teen-images/IMG_9275-3_1.jpg",
    "9-teen-images/IMG_9290-13_1.jpg",
    "9-teen-images/IMG_9281-7_1.jpg",
]

PAD = 0.05          # padding around subject, fraction of bbox size

def subject_bbox(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # texture/edges mark the subject; the white backdrop (even where softly
    # shadowed) is smooth, so it carries almost no gradient energy.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=max(h, w) / 220.0)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # smooth, border-connected regions = backdrop (white OR soft shadow).
    # Textured subject and interior smooth patches (white pages it frames) stay.
    smooth = (mag < 18).astype(np.uint8)
    num, labels = cv2.connectedComponents(smooth)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    backdrop = np.isin(labels, list(border)) & (smooth > 0)
    m = np.where(backdrop, 0, 255).astype(np.uint8)
    ck = max(3, int(max(h, w) * 0.02) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
    ok = max(3, int(max(h, w) * 0.012) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    area = h * w
    boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > area * 0.008]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * PAD), int(bh * PAD)
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(w, x1 + px); y1 = min(h, y1 + py)
    return x0, y0, x1, y1

for p in IMAGES:
    img = cv2.imread(p)
    if img is None:
        print(f"{p}\tMISSING"); continue
    h, w = img.shape[:2]
    bb = subject_bbox(img)
    if bb is None:
        print(f"{p}\tno subject found"); continue
    x0, y0, x1, y1 = bb
    crop = img[y0:y1, x0:x1]
    base, ext = os.path.splitext(p)
    out = base + "-crop" + ext
    cv2.imwrite(out, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    kept = (x1 - x0) * (y1 - y0) / (w * h)
    print(f"{os.path.basename(out)}\t{w}x{h} -> {x1-x0}x{y1-y0}\tkept {kept*100:.0f}% of area")
