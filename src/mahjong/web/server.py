"""Loopback-only, same-origin API and packaged React assets."""

from contextlib import asynccontextmanager
from pathlib import Path
import secrets

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mahjong.contracts import MahjongError
from .session import Session


STATIC = Path(__file__).parent / "static"


def create_app(
    *, models_root="outputs", weights_root="weights", device="cpu", port=8765, dev_origin=None, **timeouts
):
    session = Session(models_root, weights_root=weights_root, device=device, **timeouts)
    token = secrets.token_urlsafe(32)
    allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    if dev_origin is not None:
        from urllib.parse import urlsplit

        parsed = urlsplit(dev_origin)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("localhost", "127.0.0.1")
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            session.close()
            raise MahjongError("CONFIG_INVALID", "开发 origin 必须为本机 HTTP origin")
        allowed.add(dev_origin)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            session.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.session = session

    @app.middleware("http")
    async def guard(request, call_next):
        if request.url.hostname not in ("localhost", "127.0.0.1"):
            return JSONResponse({"error": "仅允许本机访问"}, status_code=403)
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin is not None and origin not in allowed:
                return JSONResponse({"error": "拒绝跨源请求"}, status_code=403)
            if request.method != "GET" and origin not in allowed:
                return JSONResponse({"error": "写操作需要同源 Origin"}, status_code=403)
            if request.url.path != "/api/bootstrap" and not secrets.compare_digest(
                request.headers.get("x-session-token", ""), token
            ):
                return JSONResponse({"error": "会话令牌无效，请刷新"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(MahjongError)
    async def mahjong_error(request, exc):
        return JSONResponse(dict(error=str(exc), code=exc.code), status_code=409)

    @app.get("/api/bootstrap")
    def bootstrap():
        return dict(session.bootstrap(), token=token)

    @app.get("/api/catalog")
    def catalog():
        with session.lock:
            return dict(session.catalog.snapshot(), llms=[c.public() for c in session.llm_configs.values()])

    @app.post("/api/catalog/refresh")
    def refresh():
        with session.lock:
            return dict(session.catalog.refresh(), llms=[c.public() for c in session.llm_configs.values()])

    @app.post("/api/page")
    def page(body: dict):
        return session.init_page(body.get("page_id"), body.get("generation"))

    @app.get("/api/table")
    def table(page_id: str):
        return session.snapshot(page_id)

    @app.post("/api/table")
    def create(body: dict):
        return session.create(
            body.get("page_id"),
            body.get("request_id"),
            body.get("mode"),
            body.get("opponents"),
            body.get("spectator", False),
        )

    @app.post("/api/table/action")
    def action(body: dict):
        return session.submit(
            body.get("page_id"),
            body.get("table_id"),
            body.get("request_id"),
            "action",
            window=body.get("window"),
            candidate=body.get("candidate"),
        )

    @app.post("/api/table/continue")
    def next_round(body: dict):
        return session.submit(
            body.get("page_id"),
            body.get("table_id"),
            body.get("request_id"),
            "continue",
            round_id=body.get("round_id"),
            revision=body.get("revision"),
        )

    @app.post("/api/table/close")
    def close(body: dict):
        return session.close_table(body.get("page_id"), body.get("table_id"))

    @app.post("/api/table/control")
    def control(body: dict):
        return session.submit(body.get("page_id"), body.get("table_id"),
                              body.get("request_id"), "control",
                              control=body.get("control"), value=body.get("value"))

    @app.get("/")
    def index():
        if not (STATIC / "index.html").is_file():
            return JSONResponse(
                dict(error="请先在 web/ 中执行 npm ci && npm run build"),
                status_code=503,
            )
        return FileResponse(STATIC / "index.html")

    if (STATIC / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")
    return app


def serve(options):
    if not (STATIC / "index.html").is_file():
        raise MahjongError(
            "CONFIG_INVALID", "前端尚未构建，请在 web/ 中执行 npm ci && npm run build"
        )
    import uvicorn

    app = create_app(
        models_root=options.models_root,
        weights_root=options.weights_root,
        device=options.device,
        gpu_memory_gib=options.gpu_memory_gib,
        port=options.port,
        dev_origin=options.dev_origin,
        lease_seconds=options.lease_seconds,
        load_timeout=options.load_timeout,
        decision_timeout=options.decision_timeout,
        llm_config=options.llm_config,
    )
    print(f"网页对局：http://127.0.0.1:{options.port} （{options.device}）", flush=True)
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=options.port,
            workers=1,
            log_level="warning",
            access_log=False,
        )
    finally:
        app.state.session.close()
    return 0
