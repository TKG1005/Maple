"""IPCBattleController

Skeleton controller for managing a Node.js Showdown bridge process and
routing NDJSON messages to Python-side player queues.

Responsibilities (skeleton):
- Spawn/terminate a Node.js process running the IPC bridge script
- Read stdout (NDJSON) and dispatch protocol messages to player queues
- Write NDJSON commands to stdin (ping, create_battle, player commands)
- Boundary ID mapping: player_0/player_1 ⇄ p1/p2

Note: This is a minimal scaffold to unblock integration. Error handling,
metrics, and full lifecycle management can be expanded in follow-up work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional


class IPCBattleController:
    """Controller that owns a Node IPC bridge process per battle.

    One controller is intended to manage exactly one battle. It launches the
    Node.js bridge script (NDJSON over stdin/stdout) and exposes methods to
    send control/protocol messages while routing incoming messages to
    player-scoped queues.
    """

    def __init__(self, node_script_path: str, battle_id: str, logger: Optional[logging.Logger] = None) -> None:
        self.node_script_path = node_script_path
        self.battle_id = battle_id
        self.logger = logger or logging.getLogger(__name__)

        self.process: Optional[asyncio.subprocess.Process] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Player message queues (Python-internal IDs: player_0 / player_1)
        self.player_queues: Dict[str, asyncio.Queue[str]] = {
            "player_0": asyncio.Queue(),
            "player_1": asyncio.Queue(),
        }

        # ID mapping at the boundary
        self._id_map_py2sd = {"player_0": "p1", "player_1": "p2"}
        self._id_map_sd2py = {v: k for k, v in self._id_map_py2sd.items()}

        # Simple state
        self._connected = False
        self._battle_created_event: Optional[asyncio.Event] = None
        self._first_request_event: Optional[asyncio.Event] = None
        # Room tag provided by Node bridge (e.g., "battle-gen9randombattle-<id>")
        self._room_tag: Optional[str] = None
        # Cached room header string (e.g., ">battle-gen9randombattle-12345\n")
        self._room_header_cache: Optional[str] = None

    # ---- Process lifecycle ----
    async def connect(self) -> None:
        """Launch the Node.js bridge process and start I/O tasks."""
        async with self._lock:
            if self.process is not None and self._connected:
                return

            node_bin = os.environ.get("NODE_BIN", "node")
            # Allow overriding path via env if not provided/exists
            script = os.environ.get("MAPLE_NODE_SCRIPT", self.node_script_path)

            self.process = await asyncio.create_subprocess_exec(
                node_bin,
                script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Start read loops
            self._stdout_task = asyncio.create_task(self._read_stdout_loop())
            self._stderr_task = asyncio.create_task(self._read_stderr_loop())
            self._connected = True
            self.logger.info("IPC process started for %s (pid=%s)", self.battle_id, getattr(self.process, "pid", "?"))

    async def disconnect(self) -> None:
        """Terminate the Node.js process and cancel I/O tasks."""
        async with self._lock:
            self._connected = False

            # Cancel readers first
            if self._stdout_task:
                self._stdout_task.cancel()
                self._stdout_task = None
            if self._stderr_task:
                self._stderr_task.cancel()
                self._stderr_task = None

            # Terminate process
            if self.process:
                try:
                    if self.process.stdin:
                        self.process.stdin.close()
                except Exception:
                    pass
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                self.process = None
            self.logger.info("IPC process terminated for %s", self.battle_id)

    async def is_alive(self) -> bool:
        """Return True if process is running and connected."""
        proc = self.process
        if proc is None:
            return False
        if proc.returncode is not None:
            return False
        return self._connected

    # ---- Control/Protocol API ----
    async def ping(self, timeout: float = 5.0) -> bool:
        """Send ping and wait for pong."""
        await self._write_json({"type": "ping"})

        # Wait for a matching pong on stdout reader side using a short latch
        fut: asyncio.Future[bool] = asyncio.get_event_loop().create_future()

        def _one_shot(msg: Dict[str, Any]) -> None:
            if not fut.done():
                fut.set_result(True)

        # Install temporary pong waiter
        waiter_id = id(fut)
        attr_name = f"_pong_waiter_{waiter_id}"
        setattr(self, attr_name, _one_shot)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            # Ensure cleanup regardless of race with route_message
            try:
                delattr(self, attr_name)
            except AttributeError:
                pass

    async def create_battle(self, format_id: str, players: list[Dict[str, Any]], seed: Optional[list[int]] = None, timeout: float = 10.0) -> None:
        """Send create_battle to Node bridge.

        Args:
            format_id: e.g., "gen9randombattle" or "gen9bssregi"
            players: list like [{"name": str, "team": Optional[str]}, {"name": str, "team": Optional[str]}]
            seed: Optional PRNG seed array
        """
        payload: Dict[str, Any] = {
            "type": "create_battle",
            "battle_id": self.battle_id,
            "format": format_id,
            "players": players,
        }
        if seed is not None:
            payload["seed"] = seed
        # Reset readiness events per creation
        self._battle_created_event = asyncio.Event()
        self._first_request_event = asyncio.Event()
        # room_tag is provided by Node; clear any previous cache
        self._room_tag = None
        self._room_header_cache = None

        await self._write_json(payload)

        # Wait until either battle_created ACK or first |init|/|request| arrives
        async def _wait_any():
            assert self._battle_created_event is not None
            assert self._first_request_event is not None
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(self._battle_created_event.wait()),
                    asyncio.create_task(self._first_request_event.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

        try:
            await asyncio.wait_for(_wait_any(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Timed out waiting for battle readiness: {self.battle_id}") from e

    async def send_protocol(self, player_py_id: str, data: str) -> None:
        """Send a protocol command for a specific player (Python-side ID)."""
        player_sd = self._id_map_py2sd.get(player_py_id)
        if player_sd is None:
            raise ValueError(f"Unknown player id: {player_py_id}")
        await self._write_json({
            "player_id": player_sd,
            "battle_id": self.battle_id,
            "data": data,
        })

    # ---- Routing ----
    async def route_message(self, message: Dict[str, Any]) -> None:
        """Route a Node→Python message to the appropriate player queue.

        Expected Node protocol shape:
          {"type":"protocol","battle_id":...,"target_player":"p1|p2","data":"..."}
        """
        mtype = message.get("type")
        if mtype != "protocol":
            # handle control side-effects (pong, errors) or ignore
            if mtype == "pong":
                # Trigger any pending pong waiter(s); cleanup is handled in ping()
                for attr in dir(self):
                    if attr.startswith("_pong_waiter_"):
                        cb = getattr(self, attr, None)
                        if callable(cb):
                            try:
                                cb(message)  # type: ignore[misc]
                            except Exception:
                                pass
                return
            if mtype == "battle_created":
                # room_tag must be provided by the Node bridge
                room_tag = message.get("room_tag")
                if not isinstance(room_tag, str) or not room_tag:
                    await self._fatal_stop("room_tag missing in battle_created; stopping controller")
                    return
                self._room_tag = room_tag
                # Invalidate cached header so it rebuilds with new room_tag
                self._room_header_cache = None
                if self._battle_created_event is not None:
                    self._battle_created_event.set()
                return
            if mtype in ("error", "exit"):
                # In a fuller impl, surface to control queue; for now, log
                self.logger.error("Control message from Node: %s", message)
                return
            return

        # Ensure room_tag is present before routing protocol lines
        msg_room_tag = message.get("room_tag")
        if self._room_tag is None and isinstance(msg_room_tag, str) and msg_room_tag:
            self._room_tag = msg_room_tag
            self._room_header_cache = None
        if self._room_tag is None:
            await self._fatal_stop("room_tag missing in protocol; stopping controller")
            return

        target_sd = message.get("target_player")
        py_player = self._id_map_sd2py.get(str(target_sd)) if target_sd is not None else None

        # Broadcast to both if unspecified (defensive)
        if py_player is None:
            data_line = str(message.get("data", ""))
            tagged = self._tag_with_room_header(data_line)
            for q in self.player_queues.values():
                await q.put(tagged)
            # Detect readiness markers in data
            if self._first_request_event is not None and ("|init|" in data_line or "|request|" in data_line):
                self._first_request_event.set()
            return

        if py_player in self.player_queues:
            data_line = str(message.get("data", ""))
            tagged = self._tag_with_room_header(data_line)
            await self.player_queues[py_player].put(tagged)
            # Detect readiness markers
            if self._first_request_event is not None and ("|init|" in data_line or "|request|" in data_line):
                self._first_request_event.set()

    def get_queue(self, player_py_id: str) -> asyncio.Queue[str]:
        """Return the message queue for a python-side player id."""
        if player_py_id not in self.player_queues:
            raise KeyError(f"Unknown player queue: {player_py_id}")
        return self.player_queues[player_py_id]

    # ---- I/O internals ----
    async def _write_json(self, payload: Dict[str, Any]) -> None:
        proc = self.process
        if proc is None or proc.stdin is None:
            raise RuntimeError("Process not started")
        data = json.dumps(payload) + "\n"
        proc.stdin.write(data.encode("utf-8"))
        await proc.stdin.drain()

    async def _read_stdout_loop(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    self.logger.error("Invalid JSON from Node: %s", text)
                    continue
                await self.route_message(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error("Stdout loop error: %s", e)

    async def _read_stderr_loop(self) -> None:
        proc = self.process
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip("\n")
                if text:
                    self.logger.error("[NODE] %s", text)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error("Stderr loop error: %s", e)

    # ---- Helpers ----
    async def _fatal_stop(self, message: str) -> None:
        """Log an error and stop the controller/process immediately."""
        self.logger.error(message)
        try:
            await self.disconnect()
        finally:
            # Raise to abort current loops
            raise RuntimeError(message)

    def _room_header(self) -> str:
        """Return cached room header prefix including trailing newline."""
        if self._room_header_cache is None:
            if not isinstance(self._room_tag, str) or not self._room_tag:
                # Should be guarded earlier; keep defensive check
                raise RuntimeError("room_tag not set; cannot build room header")
            self._room_header_cache = f">{self._room_tag}\n"
        return self._room_header_cache

    def _tag_with_room_header(self, data_line: str) -> str:
        """Prefix a single protocol line with the WS room header."""
        return f"{self._room_header()}{data_line}"
