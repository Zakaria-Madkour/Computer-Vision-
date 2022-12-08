import socketio

sio = socketio.Client()
print('my sid is', sio.sid)
