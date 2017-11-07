# ==============Group Members==================================
# Michelle Becerra mdbecerr@usc.edu
# Amirah Anwar anwara@usc.edu
# Reetinder Kaur reetindk@usc.edu

import random
import numpy as np
import cv2

def main():
	# read a pgm file
	img = cv2.imread("gestures/M/M_up_2.pgm", cv2.IMREAD_GRAYSCALE)
	print img

if __name__ == "__main__":
    main()
