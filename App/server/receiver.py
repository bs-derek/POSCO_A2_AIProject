import argparse
import socket
import numpy as np
import time
import random
# import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--ip',required =False, default='192.168.0.17',help = 'device\'s IP address')
parser.add_argument('--port',required =False, default='8500', help = 'device\'s PORT number')
args = parser.parse_args()

cIP,cPORT = args.ip, int(args.port)
# cIP = '0.0.0.0'
socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('IP Address ',cIP,':',cPORT)

socket.bind((cIP,cPORT))
socket.listen(5)

x,y =500,800


try :

	c_socket, c_addr = socket.accept()
	print('Connected with ',c_addr)
	
	while True:
		

		data = c_socket.recv(4734976)
		# img = cv2.imdecode(np.fromstring(data, np.uint8), 1)
		# img = (np.fromstring(data, np.uint8), 1)
		# mat = np.frombuffer(data,dtype=np.int)
		print('getdata',str(len(data)))
		# print('get',str(img.shape[1]) +" "+ str(img.shape[0]))

		x = x + random.randint(-20,20)
		y = y + random.randint(-20,20)
		click =0
		cord = str(x) + '/' + str(y) + '/' +str(click)
		c_socket.sendall(cord)

		print('senddata',cord)
		# model data
		# connect.sendall()
except :
	print('Connecting Error')
	pass
finally :
	socket.close()
	print('End of the session')



	

