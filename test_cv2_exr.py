import cv2
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.float32)
success = cv2.imwrite('test.exr', img)
print("EXR Success:", success)
if not success:
    try:
        cv2.imwrite('test.exr', img, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    except Exception as e:
        print("Error:", e)
