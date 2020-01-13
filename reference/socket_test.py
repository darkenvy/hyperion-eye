import socket
import json

sample = {
  "command":"image",
  "imageheight":5,
  "imagewidth":10,
  "priority":500,
}

sample['imagedata'] = 'cpGTP1FLLkdKIjxFLjYtIy0mKy0gEiYqDCQpCCQqk6Sd/frKKEVVIjxL9sdRJDI8jnU/LDAqCiEtKjU9tK+lIj9JGTdFITZGf4J/GSxCzrqd///9Hy03PkFD4t7HFDJBOUlkgYuZf4aNYGt5V0s1/fXcHCQlGSUtES09FzFR////pq2za3Z4wca7MicKs3cA+/v8Gys0'

sample = json.dumps(sample) # convert to JSON
sample = '%s\n' % sample # append \n as terminator
sample = str.encode(sample) # convert to 'binary-like object' from string

host = socket.gethostname()
port = 19444                   # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
s.send(sample)
# data = s.recv(1024)
s.close()
# print('Received', repr(data))