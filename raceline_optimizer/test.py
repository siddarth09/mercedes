import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/home/siddarth/f1ws/src/mercedes/maps/csc433_track.pgm", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")
plt.title(f"Min={img.min()} Max={img.max()}")
plt.show()
