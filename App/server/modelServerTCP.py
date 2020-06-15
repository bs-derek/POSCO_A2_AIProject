import argparse
import socket

parser = argparse.ArgumentParser()
parser.add_argument('--ip',required =False, default='192.168.0.21',help = 'device\'s IP address')
parser.add_argument('--port',required =False, default='8000', help = 'device\'s PORT number')
args = parser.parse_args()

cIP,cPORT = args.ip, int(args.port)
cIP = '0.0.0.0'


try :
	socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	print('IP Address ',cIP,':',cPORT)

	socket.bind((cIP,cPORT))
	socket.listen(1)
	connect, addr = socket.accept()
	print('Connected with ',addr)
	
	while True :

		packet = connect.recv()
		print(data.len())
		data = packet.decode()
		print('get')
		# model data
		# connect.sendall()

		if data == 'END' :
			connect.close()
			print('Disconnected with',cIP)
			break
except :
	print('Connecting Error')
	pass
finally :
	print('End of the session')
