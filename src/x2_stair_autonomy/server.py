from __future__ import annotations

import argparse
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import ssl
from threading import Thread
import time
from urllib.parse import urlparse

from .auth import SessionManager, new_operator_token
from .models import Direction
from .runtime import RuntimeState


LESSONS = [
    {"id": "setup", "title": "1. Setup", "expected": "AimDK v1.0.0, Ubuntu 22.04, ROS 2 Humble on PC2."},
    {"id": "connectivity", "title": "2. Connectivity", "expected": "10.0.1.41 responds; SSH port 22 is open."},
    {"id": "sensors", "title": "3. Sensors", "expected": "Depth, LiDAR, torso IMU and joints are fresh."},
    {"id": "calibration", "title": "4. Stair calibration", "expected": "Supported straight geometry and confidence >= 0.80."},
    {"id": "recording", "title": "5. Data recording", "expected": "A synchronized rosbag and metadata manifest."},
    {"id": "simulation", "title": "6. Simulation", "expected": "Smoke environment and fault tests pass before training."},
    {"id": "policy", "title": "7. Policy evaluation", "expected": "Held-out success >= 95%; ONNX parity passes."},
    {"id": "safety", "title": "8. Safety checklist", "expected": "Vendor approval, gantry, E-stop and spotter confirmed."},
    {"id": "mission", "title": "9. Supervised mission", "expected": "Shadow mode until every hardware gate is signed."},
    {"id": "logs", "title": "10. Logs", "expected": "Every transition, fault and sensor age is recorded."},
]


def _json_bytes(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _cookie_session(raw: str | None) -> str | None:
    if not raw:
        return None
    cookie = SimpleCookie()
    cookie.load(raw)
    morsel = cookie.get("x2_session")
    return morsel.value if morsel else None


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "X2StairDashboard/0.1"

    @property
    def app(self):
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        # Never log tokens, cookies, or POST bodies.
        print(f"{self.client_address[0]} {fmt % args}")

    def _send_json(self, value, status=HTTPStatus.OK, headers=None):
        payload = _json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'")
        for name, value in headers or []:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 65536:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _authenticated(self) -> bool:
        sid = _cookie_session(self.headers.get("Cookie"))
        csrf = self.headers.get("X-CSRF-Token")
        return self.app.auth.validate(sid, csrf)

    def _static(self, path: str):
        relative = "index.html" if path in {"/", "/index.html"} else path.lstrip("/")
        if relative not in {"index.html", "app.js", "styles.css"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        target = self.app.static_dir / relative
        payload = target.read_bytes()
        content_type = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[target.suffix]
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/v1/"):
            snap = self.app.state.snapshot()
            routes = {
                "/api/v1/health": {"ok": True, "version": "0.1.0", "commands_compiled": False},
                "/api/v1/stairs/status": snap["stairs"],
                "/api/v1/mission/status": snap["mission"],
                "/api/v1/safety/status": snap["safety"],
                "/api/v1/telemetry": snap["telemetry"],
                "/api/v1/training/runs": self.app.state.training_runs,
                "/api/v1/gates/status": snap["release_gate"],
                "/api/v1/compatibility": snap["compatibility"],
                "/api/v1/lessons": LESSONS,
                "/api/v1/state": snap,
            }
            if path not in routes:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            else:
                self._send_json(routes[path])
            return
        self._static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/v1/session/login":
                result = self.app.auth.login(str(body.get("token", "")))
                if result is None:
                    self._send_json({"error": "invalid credentials"}, HTTPStatus.UNAUTHORIZED)
                    return
                sid, csrf = result
                secure = "; Secure" if self.app.secure else ""
                cookie = f"x2_session={sid}; Path=/; HttpOnly; SameSite=Strict{secure}; Max-Age=900"
                self._send_json({"authenticated": True, "csrf": csrf}, headers=[("Set-Cookie", cookie)])
                return
            if not self._authenticated():
                self._send_json({"error": "authenticated session and CSRF token required"}, HTTPStatus.FORBIDDEN)
                return
            if path == "/api/v1/checklist":
                self.app.state.set_checklist(body)
            elif path == "/api/v1/control/arm":
                self.app.state.refresh_safety()
                self.app.state.supervisor.arm()
            elif path == "/api/v1/control/start":
                direction = Direction(str(body.get("direction", "")))
                mission_id = str(body.get("mission_id", "")).strip()
                if not mission_id or len(mission_id) > 80:
                    raise ValueError("mission_id must contain 1-80 characters")
                self.app.state.supervisor.start(mission_id, direction)
            elif path == "/api/v1/control/heartbeat":
                self.app.state.supervisor.heartbeat()
            elif path == "/api/v1/control/abort":
                self.app.state.supervisor.abort("operator software stop")
            elif path == "/api/v1/control/disarm":
                self.app.state.supervisor.disarm()
            elif path == "/api/v1/control/clear-fault":
                self.app.state.refresh_safety()
                self.app.state.supervisor.clear_fault()
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(self.app.state.snapshot())
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)


class App:
    def __init__(self, root: Path, token: str, *, secure: bool = False):
        self.root = root
        self.static_dir = root / "static"
        self.secure = secure
        self.state = RuntimeState(root)
        self.auth = SessionManager(token)


def _watchdog(app: App):
    while True:
        app.state.supervisor.tick()
        time.sleep(0.05)


def main(argv=None):
    parser = argparse.ArgumentParser(description="X2 stair learning and safety dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert")
    parser.add_argument("--key")
    parser.add_argument("--token-file")
    parser.add_argument("--ros", action="store_true", help="subscribe to read-only AimDK/derived status")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    token_path = Path(args.token_file) if args.token_file else root / "config" / "operator-token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    else:
        token = new_operator_token()
        token_path.write_text(token + "\n", encoding="utf-8")
        try:
            token_path.chmod(0o600)
        except OSError:
            pass
        print(f"Created operator token at {token_path}. Read it locally; it will not be logged again.")
    if args.host != "127.0.0.1" and not (args.cert and args.key):
        raise SystemExit("Refusing non-loopback bind without --cert and --key")
    app = App(root, token, secure=bool(args.cert and args.key))
    if args.ros:
        from .ros_bridge import start_ros_bridge
        start_ros_bridge(app.state)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.app = app  # type: ignore[attr-defined]
    if args.cert and args.key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.cert, args.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    Thread(target=_watchdog, args=(app,), daemon=True).start()
    scheme = "https" if args.cert else "http"
    print(f"Dashboard: {scheme}://{args.host}:{args.port} (robot commands are not compiled)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
