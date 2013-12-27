import cv2
import numpy as np
import os
import shutil

# Moving objects that are smaller than this number are discarded
# The number is a fraction of width or height
MIN_MOVING_OBJECT_SIZE_PERC = 0.05
IMAGE_OUTPUT_INTERVAL_FRAMES = 30


def mk_clean_dir(path):
  if os.path.exists(path):
    shutil.rmtree(path)
  os.makedirs(path)

movie_id = '1'
cam = cv2.VideoCapture('/Users/rantav/dev/iter5/data/movies/%s.mov' % movie_id)
output_folder = '/Users/rantav/dev/iter5/data/output/%s' % movie_id
mk_clean_dir(output_folder)

frame_id = 0


cv2.namedWindow("result")

kernel = np.ones((5, 5), np.uint8)


def get_frame(cam):
  global frame_id
  frame_id += 1
  frame = cam.read()[1]
  frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)
  return frame, frame_bw

_, prev_frame_bw = get_frame(cam)
frame, frame_bw = get_frame(cam)

width, height = frame_bw.shape
min_width = width * MIN_MOVING_OBJECT_SIZE_PERC
min_height = height * MIN_MOVING_OBJECT_SIZE_PERC

last_image = -1
while True:
  diff = cv2.absdiff(frame_bw, prev_frame_bw)
  ret, thresh = cv2.threshold(diff, 20, 255, 0)
  closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
  image, contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
  # with_contours = cv2.drawContours(t, contours, -1, (0,255,0), 3)
  xmin = width
  ymin = height
  xmax = ymax = 0
  for i in range(0, len(contours)):
    print len(contours)
    cnt = contours[i]
    x, y, w, h = cv2.boundingRect(cnt)
    if w >= min_width and h >= min_height:
      xmin = min(xmin, x)
      ymin = min(ymin, y)
      xmax = max(xmax, x + w)
      ymax = max(ymax, y + h)
  if xmax > 0 and ymax > 0:
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (180, 200, 10), 2)
    # write an image
    if frame_id - last_image > IMAGE_OUTPUT_INTERVAL_FRAMES:
      file_name = os.path.join(output_folder, '%d.jpg' % frame_id)
      cv2.imwrite(file_name, frame)
      last_image = frame_id
  cv2.imshow("result", frame)

  # Read next image
  prev_frame_bw = frame_bw
  frame, frame_bw = get_frame(cam)

  key = cv2.waitKey(10)
  if key == 27:
    cv2.destroyWindow()
    break

print "Goodbye"
