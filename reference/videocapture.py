import cv2
import zmq
import base64
import numpy as np

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://0.0.0.0:2222')
# footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

while True:
    try:
        # frame = footage_socket.recv_string()
        
        # stdin = sys.stdin.buffer.read()
        # array = numpy.frombuffer(frame, dtype='uint8')
        # img = cv2.imdecode(array, 1)
        # cv2.imshow("window", img)
        # cv2.imshow("Stream", img)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        break