import asyncio
import websockets
import json

# 存储已连接的客户端
connected_clients = set()

async def handle_connection(websocket, path):
    # 注册客户端
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                # 解析收到的棋子位置
                data = json.loads(message)
                print(f"收到棋子位置: {data}")

                # 提取 x, y, color
                x = data.get("x", 0)
                y = data.get("y", 0)
                color = data.get("color", "unknown")

                # 模拟下一步建议逻辑（可替换为实际棋盘分析）
                next_move = {
                    "next_move": f"move_to_x:{x + 1},y:{y + 1}",
                    "color": color,
                    "timestamp": asyncio.get_event_loop().time()
                }

                # 发送下一步建议给客户端
                response = json.dumps(next_move)
                await websocket.send(response)
                print(f"发送下一步建议: {response}")

            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "无效的 JSON 格式"}))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # 断开连接时移除客户端
        connected_clients.remove(websocket)


# 启动 WebSocket 服务器
async def main():
    server = await websockets.serve(handle_connection, "localhost", 8765)
    print("WebSocket 服务器运行在 ws://localhost:8765")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())