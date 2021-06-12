import cv2
# w, h = 360, 240
def display(img, w,h):
    x = int(w / 3)
    y = 2 * int(w / 3)
    u = int(h / 3)
    v = 2 * int(h / 3)
    cv2.line(img,
             (x, 0), (x, h),
             (255, 0, 230),
             3)
    cv2.line(img,
             (y, 0), (y, h),
             (255, 0, 230),
             3)
    cv2.line(img,
             (x, u), (y, u),
             (255, 0, 230),
             3)
    cv2.line(img,
             (x, v), (y, v),
             (255, 0, 230),
             3)
