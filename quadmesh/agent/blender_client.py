"""Agent-side client for quadmesh/agent/blender_bridge.py's socket server."""
import json
import socket

HOST, PORT = "127.0.0.1", 8765


def call(op: str, **args) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(json.dumps({"op": op, "args": args}).encode())
    resp = json.loads(s.recv(65536).decode())
    s.close()
    return resp


def is_up() -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect((HOST, PORT))
        s.close()
        return True
    except Exception:
        return False
