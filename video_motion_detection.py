import cv2
import numpy as np

def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  return d1
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)


cam = cv2.VideoCapture('/Users/rantav/dev/iter5/data/movies/1.mov')

cv2.namedWindow("result")
cv2.namedWindow("diff")
cv2.namedWindow("t")
cv2.namedWindow("t_minus")
cv2.namedWindow("t_plus")
cv2.namedWindow("thresh")

kernel = np.ones((5,5),np.uint8)

# Read three images first:
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)

while True:
  diff = diffImg(t_minus, t, t_plus)
  ret, thresh = cv2.threshold(diff, 20, 255, 0)
  closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
  image, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # with_contours = cv2.drawContours(t, contours, -1, (0,255,0), 3)
  xmin = ymin = 100000
  xmax = ymax = 0
  for i in range(0, len(contours)):
    print len(contours)
    cnt = contours[i]
    x, y, w, h = cv2.boundingRect(cnt)
    xmin = min(xmin, x)
    ymin = min(ymin, y)
    xmax = max(xmax, x + w)
    ymax = max(ymax, y + h)
  cv2.rectangle(t, (xmin, ymin), (xmax, ymax), (180, 200, 10), 2)
  cv2.imshow("result", t)
  # cv2.imshow('diff', diff)
  # cv2.imshow('t_minus', t_minus)
  # cv2.imshow('t_plus', t_plus)
  # cv2.imshow('thresh', thresh)

  # Read next image
  t_minus = t
  t = t_plus
  t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
  t_plus = cv2.GaussianBlur(t_plus, (5,5), 0)

  key = cv2.waitKey(10)
  if key == 27:
    cv2.destroyWindow()
    break

print "Goodbye"






# import cv

# class Target:

#     def __init__(self):
#         self.capture = cv.CaptureFromCAM(0)
#         cv.NamedWindow("Target", 1)

#     def run(self):
#         # Capture first frame to get size
#         frame = cv.QueryFrame(self.capture)
#         frame_size = cv.GetSize(frame)
#         color_image = cv.CreateImage(cv.GetSize(frame), 8, 3)
#         grey_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
#         moving_average = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_32F, 3)

#         first = True

#         while True:
#             closest_to_left = cv.GetSize(frame)[0]
#             closest_to_right = cv.GetSize(frame)[1]

#             color_image = cv.QueryFrame(self.capture)

#             # Smooth to get rid of false positives
#             cv.Smooth(color_image, color_image, cv.CV_GAUSSIAN, 3, 0)

#             if first:
#                 difference = cv.CloneImage(color_image)
#                 temp = cv.CloneImage(color_image)
#                 cv.ConvertScale(color_image, moving_average, 1.0, 0.0)
#                 first = False
#             else:
#                 cv.RunningAvg(color_image, moving_average, 0.020, None)

#             # Convert the scale of the moving average.
#             cv.ConvertScale(moving_average, temp, 1.0, 0.0)

#             # Minus the current frame from the moving average.
#             cv.AbsDiff(color_image, temp, difference)

#             # Convert the image to grayscale.
#             cv.CvtColor(difference, grey_image, cv.CV_RGB2GRAY)

#             # Convert the image to black and white.
#             cv.Threshold(grey_image, grey_image, 70, 255, cv.CV_THRESH_BINARY)

#             # Dilate and erode to get people blobs
#             cv.Dilate(grey_image, grey_image, None, 18)
#             cv.Erode(grey_image, grey_image, None, 10)

#             storage = cv.CreateMemStorage(0)
#             contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
#             points = []

#             while contour:
#                 bound_rect = cv.BoundingRect(list(contour))
#                 contour = contour.h_next()

#                 pt1 = (bound_rect[0], bound_rect[1])
#                 pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
#                 points.append(pt1)
#                 points.append(pt2)
#                 cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 1)

#             if len(points):
#                 center_point = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), points)
#                 cv.Circle(color_image, center_point, 40, cv.CV_RGB(255, 255, 255), 1)
#                 cv.Circle(color_image, center_point, 30, cv.CV_RGB(255, 100, 0), 1)
#                 cv.Circle(color_image, center_point, 20, cv.CV_RGB(255, 255, 255), 1)
#                 cv.Circle(color_image, center_point, 10, cv.CV_RGB(255, 100, 0), 1)

#             cv.ShowImage("Target", color_image)

#             # Listen for ESC key
#             c = cv.WaitKey(7) % 0x100
#             if c == 27:
#                 break

# if __name__=="__main__":
#     t = Target()
#     t.run()