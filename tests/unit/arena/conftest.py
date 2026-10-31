from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def llm_server():
    state = SimpleNamespace(calls=[], replies=[])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append(dict(path=self.path, body=body, authorization=self.headers.get("Authorization")))
            if state.replies:
                status, result = state.replies.pop(0)
            else:
                prompt = json.loads(body["messages"][1]["content"])
                move = prompt["legal_actions"][0]
                result = dict(choices=[dict(message=dict(content=json.dumps(dict(candidate_id=move["candidate_id"]))))],
                              usage=dict(prompt_tokens=20, completion_tokens=5))
                status = 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
