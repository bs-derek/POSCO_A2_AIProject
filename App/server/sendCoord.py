import socket
import time
import random

send_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

cIP = '192.168.0.21'
cPORT = 8500
dest = (cIP, cPORT)

x,y =500,800

for i in range(0,1000) :
	x = x+ random.randint(-20,20)
	y = y + random.randint(-20,20)
	send_sock.sendto(str(x)+'/'+str(y),dest)
	time.sleep(0.1)
	if(i%10 == 0):
		print('i is ',i)

send_sock.close()
