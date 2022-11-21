import numpy as np
import cv2

def reader(f):
    cap = cv2.VideoCapture(f)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        yield frame

if __name__ == "__main__":
    import sys
    for frame in reader('test.mov'):
        print(sys.getsizeof(frame))
        break