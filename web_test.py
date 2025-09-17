import websocket
try:
    ws = websocket.WebSocket()
    ws.connect("ws://localhost:8765")
    print("WebSocket 连接成功")
    ws.close()
except Exception as e:
    print(f"WebSocket 连接失败: {e}")