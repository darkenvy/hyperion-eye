import cv2
import urllib.request
import sys
import numpy

stream = sys.stdin.buffer.read()

# array = numpy.frombuffer(stdin, dtype='uint8')
# img = cv2.imdecode(array, 1)
# cv2.imshow("window", img)
# cv2.waitKey()

# stream = urllib.request.urlopen('http://10.0.0.38:2222/')
bytes = ''
while True:
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        i = cv2.imdecode(numpy.fromstring(jpg, dtype=numpy.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        cv2.imshow('i', i)
        if cv2.waitKey(1) == 27:
            exit(0)   