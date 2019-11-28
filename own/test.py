
import time

root = 'own/'
buffer = 'buffer.txt'
memory = 'memory.txt'


#  python own/test.py | ./pokemon-showdown simulate-battle > own/buffer.txt


def empty_files():
	fmemory = open(root + buffer, mode='w')
	fmemory.close()
	fbuffer = open(root + memory, mode='w')
	fbuffer.close()

def read_last_showdown_message():
	fbuffer = open(root + buffer, mode='r+')
	fbuffer.seek(0)
	output = fbuffer.read()
	fbuffer.close()
	return output

def send_message_to_showdown(message):
	# write to stdout (which is piped directly into stdin of pokemon-showdown)
	print(message, end='', flush=True)

def record_message_to_memory(message):
	fmemory = open(root + memory, mode='a')
	fmemory.write(message)
	fmemory.close()


# empty files
empty_files()

inputs = [
	'>start {"formatid":"gen7randombattle"}',
	'>player p1 {"name":"Scott"}',
	'>player p2 {"name":"Lars"}'
]


for line in inputs:

	#time.sleep(0.5)

	# send to showdown
	send_message_to_showdown(line + '\n')
	record_message_to_memory(line + '\n')

	# receive from showdown 

	time.sleep(0.5)


	message = read_last_showdown_message()
	record_message_to_memory(message + '\n')

