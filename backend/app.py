"""FastAPI status, session controls and real-time CSI WebSocket."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import time

from anyio import CancelScope
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .session import Session, Settings


def sent(message: dict) -> dict:
    return {**message, "server_sent_ms": time.time_ns() // 1_000_000}


def check_request_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is not None and origin != f"{request.url.scheme}://{request.headers.get('host')}":
        raise HTTPException(403, "The request must come from the same origin")


def create_app(settings: Settings) -> FastAPI:
    session = Session(settings)

    @asynccontextmanager
    async def lifespan(app):
        if settings.autostart:
            await session.start()
        try:
            yield
        finally:
            await session.stop()

    app = FastAPI(title="Wi-Fi Sensing CSI", version="1", lifespan=lifespan)
    dashboard_dir = Path(__file__).resolve().parents[1] / "frontend" / "dashboard"
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    async def preview():
        return FileResponse(Path(__file__).with_name("preview.html"))

    @app.get("/api/status")
    async def status():
        return sent(session.status())

    @app.get("/api/session/report", responses={404: {"description": "No session has started"}})
    async def report():
        """Return the current or most recent session's measurements as JSON."""
        result = session.report()
        if result is None:
            raise HTTPException(404, "No session has started")
        return result

    @app.post("/api/session/start", status_code=202)
    async def start(request: Request):
        check_request_origin(request)
        if not await session.start():
            raise HTTPException(409, "A session is already active")
        return sent(session.status())

    @app.post("/api/session/stop")
    async def stop(request: Request, session_id: str | None = None):
        check_request_origin(request)
        if not await session.stop(session_id):
            raise HTTPException(409, "The session has changed")
        return sent(session.status())

    @app.websocket("/ws")
    async def stream(websocket: WebSocket):
        # Browser clients use the same origin as the preview. Clients running
        # on a frontend server can connect from that server (no Origin header).
        origin = websocket.headers.get("origin")
        expected = f"{websocket.url.scheme.replace('ws', 'http')}://{websocket.headers.get('host')}"
        if origin is not None and origin != expected:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        subscriber = session.subscribe()

        async def send_updates():
            while True:
                message = await subscriber.get()
                started = time.perf_counter()
                try:
                    await asyncio.wait_for(websocket.send_json(sent(message)), timeout=2)
                except TimeoutError:
                    session.record_send(message, None)
                    raise
                session.record_send(message, (time.perf_counter() - started) * 1000)

        async def watch_disconnect():
            while (await websocket.receive())["type"] != "websocket.disconnect":
                pass

        tasks = [asyncio.create_task(send_updates()), asyncio.create_task(watch_disconnect())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, OSError):
            pass
        except TimeoutError:
            await websocket.close(code=1013)
        finally:
            session.subscribers.discard(subscriber)
            for task in tasks:
                task.cancel()
            with CancelScope(shield=True):
                await asyncio.gather(*tasks, return_exceptions=True)

    return app
