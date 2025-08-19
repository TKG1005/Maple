"""Dual-mode player supporting both WebSocket and IPC communication.

This module provides a player implementation that can switch between
WebSocket (online) and IPC (local) communication modes while maintaining
full compatibility with the existing poke-env Player interface.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional
import time
import random

from poke_env.ps_client.server_configuration import ServerConfiguration

from .env_player import EnvPlayer
from src.sim.battle_communicator import CommunicatorFactory, BattleCommunicator
from src.env.rqid_notifier import get_global_rqid_notifier

class IPCClientWrapper:
    """IPC client wrapper to manage a Node.js process and JSON message exchange."""

    def __init__(self, node_script_path: str, logger: Optional[logging.Logger] = None, reuse_enabled: Optional[bool] = None, max_processes: Optional[int] = None) -> None:
        """
        Initialize the IPC client wrapper.

        Args:
            node_script_path: Path to the Node.js IPC server script.
            logger: Optional logger instance.
        """
        from src.env.ipc_battle_controller import IPCBattleController  # local import to avoid cycles

        self.node_script_path = node_script_path
        self.logger = logger or logging.getLogger(__name__)
        # Manage controllers via shared registry (per battle_id)
        from src.env.controller_registry import ControllerRegistry  # type: ignore
        self._registry = ControllerRegistry
        self._controllers: dict[str, "IPCBattleController"] = {}
        # Phase1: enable controller reuse (sequential battles per process)
        if reuse_enabled is None:
            self._reuse_enabled: bool = os.environ.get("MAPLE_IPC_REUSE", "1").lower() in ("1", "true", "yes")
        else:
            self._reuse_enabled = bool(reuse_enabled)
        # Pool upper bound (default 2, overridable via env/argument)
        if max_processes is None:
            env_val = os.environ.get("MAPLE_IPC_MAX_PROCESSES")
            try:
                self._max_processes = int(env_val) if env_val else 2
            except Exception:
                self._max_processes = 2
        else:
            try:
                self._max_processes = int(max(1, max_processes))
            except Exception:
                self._max_processes = 2
        self.connected: bool = False  # True if at least one controller is alive

    async def connect(self) -> None:
        """No-op: controllers are created per battle. Kept for compatibility."""
        return None

    async def disconnect(self) -> None:
        """Terminate all known controllers and cleanup."""
        for ctrl in list(self._controllers.values()):
            try:
                await ctrl.disconnect()
            except Exception:
                continue
        self._controllers.clear()
        self.connected = False

    async def send(self, message: Dict[str, Any]) -> None:
        """
        Send a JSON message to the Node.js process.

        Args:
            message: A dict representing the JSON envelope.
        """
        # Generic entrypoint for protocol/control. Prefer using explicit methods.
        mtype = message.get("type")
        battle_id = message.get("battle_id")
        if mtype == "ping":
            await self.ping()
            return
        if mtype == "create_battle":
            await self.create_battle(
                battle_id,
                message.get("format"),
                message.get("players", []),
                message.get("seed"),
            )
            return
        # protocol to player
        if battle_id is None:
            raise ValueError("battle_id is required for protocol messages")
        player_id = message.get("player_id")
        data = message.get("data")
        if player_id is None or data is None:
            raise ValueError("player_id and data are required for protocol messages")
        ctrl = await self._ensure_controller(battle_id)
        await ctrl.send_protocol(self._py_id_from_sd(player_id), data)

    # receive_message was intentionally removed. Use per-player recv() instead.

    async def is_alive(self) -> bool:
        """
        Check if the Node.js subprocess is alive.

        Returns:
            True if process is running and connected.
        """
        # True if any controller is alive
        for ctrl in self._controllers.values():
            if await ctrl.is_alive():
                return True
        return False

    async def _read_stdout_loop(self) -> None:
        """Internal task: read JSON lines from stdout."""
        return None

    async def _read_stderr_loop(self) -> None:
        """Internal task: read stderr lines for logging."""
        return None

    # ---- Convenience API matching controller responsibilities ----
    async def ping(self) -> bool:
        """Ping using an existing controller, or a short-lived probe."""
        # Prefer the first alive controller
        for ctrl in self._controllers.values():
            if await ctrl.is_alive():
                return await ctrl.ping()

        # Fallback: create a short-lived controller to probe Node
        from src.env.controller_registry import ControllerRegistry  # type: ignore
        probe_id = f"probe-{os.getpid()}"
        ctrl = ControllerRegistry.get_or_create(self.node_script_path, probe_id, logger=self.logger)
        try:
            await ctrl.connect()
            ok = await ctrl.ping()
            await ctrl.disconnect()
            return ok
        except Exception:
            try:
                await ctrl.disconnect()
            except Exception:
                pass
            return False

    async def create_battle(self, battle_id: str, format_id: str, players: list[dict], seed: Optional[list[int]] = None) -> None:
        """Create a battle by delegating to the per-battle controller."""
        ctrl = await self._ensure_controller(battle_id)
        await ctrl.create_battle(format_id, players, seed, battle_id=battle_id)

    async def create_battle_by_room(self, room_tag: str, format_id: str, players: list[dict], seed: Optional[list[int]] = None) -> None:
        """Create a battle referenced by `room_tag` instead of a numeric battle_id.

        This helper registers/obtains a controller keyed by `room_tag` and
        delegates the create_battle call. This allows Phase3-style room_tag
        based creation while keeping backwards-compatible APIs available.
        """
        # Use shared controller when reuse is enabled
        ctrl = await self._ensure_controller(room_tag)
        await ctrl.create_battle(format_id, players, seed, battle_id=room_tag)

    async def listen(self) -> None:
        """Compatibility stub: controller streams are push-based; nothing to do here."""
        return None

    async def recv(self, battle_id: str, player_py_id: str, timeout: Optional[float] = None) -> str:
        """プレイヤー専用キューから1行のShowdown生テキストを受信する。

        Args:
            battle_id: 対象バトルID
            player_py_id: "player_0" または "player_1"
            timeout: 秒指定のタイムアウト（Noneで無期限）

        Returns:
            受信した1行のShowdown生テキスト
        """
        ctrl = await self._ensure_controller(battle_id)
        try:
            queue = ctrl.get_queue(player_py_id)
        except KeyError as e:
            raise ValueError(f"Unknown player_py_id: {player_py_id}") from e

        try:
            if timeout is None:
                raw = await queue.get()
            else:
                raw = await asyncio.wait_for(queue.get(), timeout=timeout)

            # Raw line logging removed to reduce overhead

            return raw
        except Exception:
            # Let callers handle timeouts and other exceptions.
            raise

    async def recv_by_room(self, room_tag: str, player_py_id: str, timeout: Optional[float] = None) -> str:
        """Receive a protocol line by room_tag (migration helper).

        This temporarily maps the room_tag to a battle_id and delegates to
        `recv`. During the migration period callers can use room_tag.
        """
        battle_id = self._get_battle_id_from_room(room_tag)
        return await self.recv(battle_id, player_py_id, timeout=timeout)

    def _get_battle_id_from_room(self, room_tag: str) -> str:
        """Derive a best-effort battle_id from a room_tag.

        Example: "battle-gen9randombattle-1234-5678" -> "battle-1234-5678"
        If pattern doesn't match, return the original room_tag.
        """
        parts = room_tag.split("-")
        if len(parts) >= 4 and parts[0] == "battle":
            return f"battle-{parts[-2]}-{parts[-1]}"
        return room_tag

    async def _ensure_controller(self, battle_id: str):
        """Get or create a controller, optionally reusing a shared instance.

        When reuse is enabled, return a shared controller identified by a pool key
        derived from the node script path so both players land on the same process.
        Otherwise, return a per-battle controller keyed by battle_id.
        """
        from src.env.controller_registry import ControllerRegistry  # type: ignore

        if self._reuse_enabled:
            # Stable pool key per node script path (both players share the same)
            base = os.path.basename(self.node_script_path) or "node-ipc-bridge.js"
            pool_key = f"__ipc_pool__:{base}"
            # Use pool with upper bound from config/env
            ctrl = ControllerRegistry.get_shared_for_battle(
                self.node_script_path,
                pool_key,
                battle_id,
                max_processes=self._max_processes,
                logger=self.logger,
            )
            self._controllers[pool_key] = ctrl
        else:
            ctrl = ControllerRegistry.get_or_create(self.node_script_path, battle_id, logger=self.logger)
            self._controllers[battle_id] = ctrl

        if not await ctrl.is_alive():
            await ctrl.connect()
        self.connected = True
        return ctrl

    # ---- Internals ----
    async def _ensure_controller(self, battle_id: str):
        # In full migration, controllers are keyed by room_tag. The identifier
        # passed here may already be a room_tag. We store controllers locally
        # keyed by the same identifier for reuse.
        from src.env.controller_registry import ControllerRegistry  # type: ignore
        ctrl = self._controllers.get(battle_id)
        if ctrl is None:
            # Treat `battle_id` as room_tag in the registry
            ctrl = ControllerRegistry.get_or_create(self.node_script_path, battle_id, logger=self.logger)
            self._controllers[battle_id] = ctrl
        if not await ctrl.is_alive():
            await ctrl.connect()
        self.connected = True
        return ctrl

    @staticmethod
    def _py_id_from_sd(player_sd: str) -> str:
        if player_sd == "p1":
            return "player_0"
        if player_sd == "p2":
            return "player_1"
        raise ValueError(f"Unknown Showdown player id: {player_sd}")


class DualModeEnvPlayer(EnvPlayer):
    """Environment player supporting both online and local battle modes.
    
    This player extends EnvPlayer to support dual communication modes:
    - Online mode: Traditional WebSocket communication with Pokemon Showdown servers
    - Local mode: High-speed IPC communication with embedded Node.js processes
    
    The mode is determined at initialization and affects all battle communication.
    """
    
    def __init__(
        self,
        env: Any,
        player_id: str,
        mode: str = "local",  # "local" or "online"
        server_configuration: Optional[ServerConfiguration] = None,
        ipc_script_path: str = "scripts/node-ipc-bridge.js",
        full_ipc: bool = False,  # Phase 4: Enable full IPC mode without WebSocket fallback
        reuse_processes: Optional[bool] = None,
        max_processes: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize dual-mode player.
        
        Args:
            env: The PokemonEnv instance
            player_id: Unique identifier for this player
            mode: Communication mode ("local" for IPC, "online" for WebSocket)
            server_configuration: WebSocket server config (required for online mode)
            ipc_script_path: Path to Node.js IPC server script (for local mode)
            **kwargs: Additional arguments passed to parent EnvPlayer
        """
        self.mode = mode
        # Allow environment override; fall back to provided/default path
        self.ipc_script_path = os.environ.get("MAPLE_NODE_SCRIPT", ipc_script_path)
        self._communicator: Optional[BattleCommunicator] = None
        self._original_ps_client = None
        self.player_id = player_id  # Set player_id early for logging
        self._env = env  # Store environment reference for battle queue access
        
        self._logger = logging.getLogger(__name__)
        # Background receive pump tasks per battle_id (IPC mode)
        self._ipc_pump_tasks: dict[str, asyncio.Task] = {}
        # In-flight handler tasks per battle_id (for parallel dispatch & cleanup)
        self._ipc_inflight_tasks: dict[str, set[asyncio.Task]] = {}
        # Per-battle execution semaphores to cap parallelism
        self._exec_semaphores: dict[str, asyncio.Semaphore] = {}
        # Default max parallel handler tasks per battle
        self._ipc_max_concurrency: int = 2
        # Phase1: mapping between battle_id and room_tag (provided by Node.js)
        # battle_id -> room_tag and reverse
        self._battle_to_room: dict[str, str] = {}
        self._room_to_battle: dict[str, str] = {}
        # Lock protecting mapping updates
        self._mapping_lock: asyncio.Lock = asyncio.Lock()
        # 招待を受け取るためのキュー（player_1）
        self._ipc_invitations: asyncio.Queue[str] = asyncio.Queue()
        # Phase2: room-based battle and pump registries
        self._battles_by_room: dict[str, "Battle"] = {}
        self._pump_tasks_by_room: dict[str, asyncio.Task] = {}
        
        # Initialize parent class based on mode and full_ipc setting
        if mode == "online":
            if server_configuration is None:
                raise ValueError("server_configuration required for online mode")
            super().__init__(
                env=env,
                player_id=player_id,
                server_configuration=server_configuration,
                **kwargs
            )
            self._logger.debug(f"Online mode player initialized for {player_id}")
        else:
            # Local mode initialization - Always run in full IPC mode (WebSocket disabled)
            if server_configuration is None:
                # Create a valid server configuration for compatibility
                server_configuration = ServerConfiguration("localhost", 8000)
            # Disable WebSocket completely in local mode
            self._logger.debug(f"Initializing full IPC local mode for {player_id} (WebSocket disabled)")
            kwargs['start_listening'] = False
            try:
                super().__init__(
                    env=env,
                    player_id=player_id,
                    server_configuration=server_configuration,
                    **kwargs
                )
                self._logger.debug(f"Full IPC mode player initialized for {player_id} (WebSocket disabled)")
            except Exception as e:
                self._logger.error(f"❌ Full IPC initialization failed for {player_id}: {e}")
                raise RuntimeError(f"Local mode requires working IPC infrastructure: {e}") from e
        
        # For local mode, initialize IPC communicator
        if mode == "local":
            # Initialize communicator (create IPCClientWrapper)
            self._initialize_communicator(reuse_processes, max_processes)
            # Log IPC readiness
            self._logger.debug(f"IPC capability initialized for {self.player_id} (full IPC mode)")
    
    def _initialize_communicator(self, reuse_processes: Optional[bool] = None, max_processes: Optional[int] = None) -> None:
        """Initialize IPCClientWrapper for local mode."""
        if self.mode != "local":
            return
        self._logger.debug(f"Local IPC mode configured for player {self.player_id}")
        # Create the IPC client wrapper; actual process start is deferred
        self.ipc_client_wrapper = IPCClientWrapper(
            node_script_path=self.ipc_script_path,
            logger=self._logger,
            reuse_enabled=reuse_processes,
            max_processes=max_processes,
        )

    def _sd_id(self) -> str:
        """Return Showdown player id for this player (p1/p2)."""
        return "p1" if self.player_id == "player_0" else "p2"
    
    
    
    
    async def close_connection(self) -> None:
        """Close the communicator connection."""
        if self._communicator:
            await self._communicator.disconnect()
        
        # Call parent close if in online mode
        if self.mode == "online" and hasattr(super(), 'close_connection'):
            await super().close_connection()
    
    async def battle_against(self, opponent, n_battles=1):
        """Override battle_against for IPC mode.
        
        In IPC mode, we use asymmetric battle creation:
        - player_0 creates the battle
        - player_1 waits for invitation
        """
        if self.mode != "local":
            # Use parent class implementation for WebSocket mode
            return await super().battle_against(opponent, n_battles)
        
        self._logger.debug(f"IPC battle_against called for {self.player_id} vs {opponent.username}")
        
        for i in range(n_battles):
            # Only player_0 generates battle ID to avoid duplicates
            if self.player_id == "player_0":
                # Generate unique battle ID
                # Phase3: Generate room_tag and create battle by room
                timestamp = int(time.time() * 1000)
                random_id = random.randint(1000, 9999)
                format_id = self._format if hasattr(self, '_format') else "gen9randombattle"
                room_tag = f"battle-{format_id}-{timestamp}-{random_id}"
                self._logger.debug(f"[player_0] Creating battle (room_tag): {room_tag}")
                # Create battle via IPC using room_tag
                await self._create_ipc_battle_by_room(room_tag, opponent)
                # For downstream logic use battle_id variable (set to room_tag)
                battle_id = room_tag
            else:
                # player_1 waits for invitation
                self._logger.debug(f"[player_1] Waiting for battle invitation...")
                battle_id = await self._wait_for_battle_invitation()
                self._logger.debug(f"[player_1] Received invitation for battle: {battle_id}")
            
            # Both players now wait for battle to be ready
            self._logger.debug(f"[{self.player_id}] Waiting for battle {battle_id} to be ready...")
            await self._wait_for_battle_ready(battle_id)
            
            # Wait for battle to complete
            await self._wait_for_battle_completion(battle_id)
    
    async def _create_ipc_battle(self, battle_id: str, opponent):
        """Create a battle through IPC."""
        try:
            # Get battle format
            battle_format = self._format if hasattr(self, '_format') else "gen9randombattle"
            
            # Prepare player configurations
            # Convert team objects to strings if needed
            p1_team = None
            if hasattr(self, '_team') and self._team is not None:
                # Get the packed team string
                if hasattr(self._team, 'yield_team'):
                    p1_team = self._team.yield_team()
                else:
                    p1_team = str(self._team)
            
            p2_team = None
            if hasattr(opponent, '_team') and opponent._team is not None:
                # Get the packed team string
                if hasattr(opponent._team, 'yield_team'):
                    p2_team = opponent._team.yield_team()
                else:
                    p2_team = str(opponent._team)
            
            players = [
                {
                    "id": "p1",
                    "name": self.username,
                    "team": p1_team
                },
                {
                    "id": "p2", 
                    "name": opponent.username,
                    "team": p2_team
                }
            ]
            
            # Ensure both wrappers share the same controller
            creator_ctrl = await self.ipc_client_wrapper._ensure_controller(battle_id)
            # Pre-connect opponent wrapper to the same controller
            if hasattr(opponent, 'ipc_client_wrapper'):
                await opponent.ipc_client_wrapper._ensure_controller(battle_id)

            # Connect and ping
            await creator_ctrl.connect()
            await self.ipc_client_wrapper.ping()

            # Create battle via IPC and待機（ACKまたは初回|request|）
            await self.ipc_client_wrapper.create_battle(battle_id, battle_format, players)

            # Phase1: obtain room_tag from the controller created for this battle
            try:
                creator_ctrl = await self.ipc_client_wrapper._ensure_controller(battle_id)
                room_tag = getattr(creator_ctrl, "_room_tag", None)
                if isinstance(room_tag, str) and room_tag:
                    # Save mapping for both creator and opponent (if applicable)
                    try:
                        async with self._mapping_lock:
                            self._battle_to_room[battle_id] = room_tag
                            self._room_to_battle[room_tag] = battle_id
                            if hasattr(opponent, '_battle_to_room') and hasattr(opponent, '_room_to_battle'):
                                opponent._battle_to_room[battle_id] = room_tag
                                opponent._room_to_battle[room_tag] = battle_id
                        self._logger.debug(f"Battle {battle_id} mapped to room {room_tag}")
                    except Exception:
                        self._logger.exception("Failed to store mapping for %s", battle_id)
                else:
                    # Defensive: log a warning if room_tag not yet available
                    self._logger.warning(f"room_tag not available immediately for {battle_id}")
            except Exception as e:
                self._logger.error(f"Failed to retrieve room_tag for {battle_id}: {e}")
            
            # Do not register None placeholders here; the real Battle object
            # will be created and stored by `_handle_battle_start` to avoid
            # races where PSClient observes a None value.
            
            # Notify both players about the battle (invitation for player_1)
            await self._handle_battle_start(battle_id)
            await opponent._handle_battle_start(battle_id)
            # 通知: 招待キューへ投入（player_1 はここを待機）
            if hasattr(opponent, '_ipc_invitations'):
                await opponent._ipc_invitations.put(battle_id)
            
            self._logger.debug(f"IPC battle created: {battle_id}")

        except Exception as e:
            self._logger.error(f"❌ Failed to create IPC battle: {e}")
            raise

    async def _wait_for_battle_invitation(self) -> str:
        """player_1 が招待（battle_id）を待つ。"""
        battle_id = await self._ipc_invitations.get()
        # 自身のラッパーを同じControllerに接続
        # battle_id may be a room_tag in Phase3; ensure controller for that id
        await self.ipc_client_wrapper._ensure_controller(battle_id)
        
        # Do not register a placeholder here; `_handle_battle_start` will
        # create and register the Battle object when ready.
        
        return battle_id

    async def _create_ipc_battle_by_room(self, room_tag: str, opponent):
        """Create a battle through IPC using a room_tag as the identifier.

        This mirrors `_create_ipc_battle` but uses `room_tag` as the battle key
        so Python-side registries and controllers are keyed by the room.
        """
        try:
            # Get battle format
            battle_format = self._format if hasattr(self, '_format') else "gen9randombattle"

            # Prepare player configurations
            p1_team = None
            if hasattr(self, '_team') and self._team is not None:
                if hasattr(self._team, 'yield_team'):
                    p1_team = self._team.yield_team()
                else:
                    p1_team = str(self._team)

            p2_team = None
            if hasattr(opponent, '_team') and opponent._team is not None:
                if hasattr(opponent._team, 'yield_team'):
                    p2_team = opponent._team.yield_team()
                else:
                    p2_team = str(opponent._team)

            players = [
                {
                    "id": "p1",
                    "name": self.username,
                    "team": p1_team
                },
                {
                    "id": "p2", 
                    "name": opponent.username,
                    "team": p2_team
                }
            ]

            # Use new API to create by room
            await self.ipc_client_wrapper.create_battle_by_room(room_tag, battle_format, players)

            # Registering a placeholder (None) here causes a race where
            # poke_env may see a None battle object before the real
            # Battle is constructed in `_handle_battle_start`. Avoid
            # inserting a None placeholder; the real Battle will be
            # created and stored by `_handle_battle_start`.

            # Try to persist mapping (Phase1 compatibility)
            try:
                creator_ctrl = await self.ipc_client_wrapper._ensure_controller(room_tag)
                rt = getattr(creator_ctrl, "_room_tag", None)
                if isinstance(rt, str) and rt:
                    try:
                        async with self._mapping_lock:
                            self._battle_to_room[room_tag] = rt
                            self._room_to_battle[rt] = room_tag
                            if hasattr(opponent, '_battle_to_room') and hasattr(opponent, '_room_to_battle'):
                                opponent._battle_to_room[room_tag] = rt
                                opponent._room_to_battle[rt] = room_tag
                    except Exception:
                        self._logger.exception("Failed to store mapping for room %s", room_tag)
            except Exception:
                pass

            # Notify both players
            await self._handle_battle_start(room_tag)
            await opponent._handle_battle_start(room_tag)
            if hasattr(opponent, '_ipc_invitations'):
                await opponent._ipc_invitations.put(room_tag)

            self._logger.debug(f"IPC battle created by room: {room_tag}")

        except Exception as e:
            self._logger.error(f"❌ Failed to create IPC battle by room: {e}")
            raise
    
    async def _handle_battle_start(self, battle_id: str):
        """Handle battle start notification."""
        try:
            # Create battle object
            from poke_env.environment.abstract_battle import AbstractBattle
            from poke_env.environment.battle import Battle
            
            # Get the generation from the format
            gen = 9  # Default to gen 9
            if hasattr(self, '_format'):
                # Extract generation number from format string (e.g., "gen9bssregi" -> 9)
                import re
                match = re.search(r'gen(\d+)', self._format)
                if match:
                    gen = int(match.group(1))
            
            # Initialize battle with the given ID and generation
            # Use room_tag as the Battle.battle_tag if available (separation step A)
            try:
                rt = self.get_room_tag(battle_id)
            except Exception:
                rt = None
            battle_tag_for_obj = rt if isinstance(rt, str) and rt else battle_id
            battle = Battle(battle_tag_for_obj, self.username, self._logger, gen=gen)
            # Debug: log before registering the battle to help diagnose races
            try:
                self._logger.debug(
                    "DualModeEnvPlayer._handle_battle_start: creating battle_id=%s battle_tag=%s keys_before=%s",
                    battle_id,
                    battle_tag_for_obj,
                    list(self._battles.keys()),
                )
            except Exception:
                pass
            # store battle under local battle_id to keep controller mapping stable
            self._battles[battle_id] = battle
            try:
                self._logger.debug(
                    "DualModeEnvPlayer._handle_battle_start: stored battle_id=%s value=%r id=%s keys_after=%s",
                    battle_id,
                    repr(battle),
                    hex(id(battle)),
                    list(self._battles.keys()),
                )
            except Exception:
                pass
            # Phase2: register by room_tag if available
            try:
                room_tag = self.get_room_tag(battle_id)
                if isinstance(room_tag, str) and room_tag:
                    async def _reg_battles_by_room():
                        self._battles_by_room[room_tag] = battle
                    try:
                        async with self._mapping_lock:
                            await _reg_battles_by_room()
                    except Exception:
                        self._logger.exception("Failed to register battle by room %s", room_tag)
            except Exception:
                pass
            
            # Queue battle for environment processing
            if hasattr(self, '_env') and hasattr(self._env, '_battle_queues'):
                await self._env._battle_queues[self.player_id].put(battle)
                self._logger.debug(f"📥 Battle {battle_id} queued for {self.player_id}")

            # Start background receive pump to forward IPC messages to PSClient
            if self.mode == "local":
                await self._start_ipc_pump(battle_id)
            
        except Exception as e:
            self._logger.error(f"❌ Failed to handle battle start: {e}")
            raise
    
    async def _wait_for_battle_ready(self, battle_id: str) -> None:
        """Wait until first request is processed via event-driven rqid update."""
        timeout = 10.0  # seconds
        start_time = time.time()
        # prefer room_tag key if available
        room_key = self.get_room_tag(battle_id) or battle_id
        battle = self._battles.get(room_key) or self._battles.get(battle_id)
        # If the first request already arrived, return immediately
        if battle is not None and getattr(battle, "last_request", None):
            self._logger.debug(f"[{self.player_id}] Battle {battle_id} is ready (pre-check)!")
            try:
                dt_ms = int((time.time() - start_time) * 1000)
                self._logger.info(
                    "[METRIC] tag=ready_wait_event player=%s battle=%s ready_wait_latency_ms=%d",
                    self.player_id,
                    battle_id,
                    dt_ms,
                )
            except Exception:
                pass
            return

        # Determine baseline rqid (None if no last_request yet)
        baseline = None
        if battle is not None:
            lr = getattr(battle, "last_request", None)
            if isinstance(lr, dict):
                baseline = lr.get("rqid")

        # Register and wait for rqid update
        notifier = get_global_rqid_notifier()
        try:
            notifier.register_battle(self.player_id, initial_rqid=baseline)
            await notifier.wait_for_rqid_change(self.player_id, baseline_rqid=baseline, timeout=timeout)
            # Success
            self._logger.debug(f"[{self.player_id}] Battle {battle_id} is ready (event)!")
            try:
                dt_ms = int((time.time() - start_time) * 1000)
                self._logger.info(
                    "[METRIC] tag=ready_wait_event player=%s battle=%s ready_wait_latency_ms=%d",
                    self.player_id,
                    battle_id,
                    dt_ms,
                )
            except Exception:
                pass
            return
        except Exception as e:
            # Fail-fast per policy (no fallback)
            try:
                dt_ms = int((time.time() - start_time) * 1000)
                self._logger.exception(
                    "[READY WAIT ERROR] player=%s battle=%s room_key=%s baseline=%s elapsed_ms=%d err=%s",
                    self.player_id,
                    battle_id,
                    room_key,
                    baseline,
                    dt_ms,
                    e,
                )
                self._logger.info(
                    "[METRIC] tag=ready_wait_event_timeout player=%s battle=%s elapsed_ms=%d",
                    self.player_id,
                    battle_id,
                    dt_ms,
                )
            except Exception:
                pass
            raise SystemExit(1)
    
    async def _wait_for_battle_completion(self, battle_id: str):
        """Wait for a battle to complete."""
        # prefer room_tag key if available
        room_key = self.get_room_tag(battle_id) or battle_id
        while room_key in self._battles or battle_id in self._battles:
            battle = self._battles.get(room_key) or self._battles.get(battle_id)
            if battle and battle.finished:
                break
            await asyncio.sleep(0.1)
        
        self._logger.debug(f"Battle {battle_id} completed")
        # Stop and cleanup receive pump
        await self._stop_ipc_pump(battle_id)

    # ---- IPC receive pump ----
    async def _start_ipc_pump(self, battle_id: str) -> None:
        # Avoid duplicate pumps
        task = self._ipc_pump_tasks.get(battle_id)
        if task is not None and not task.done():
            return
        self._ipc_pump_tasks[battle_id] = asyncio.create_task(self._ipc_receive_pump(battle_id))
        # Also register pump by room_tag (Phase2)
        try:
            room_tag = self.get_room_tag(battle_id)
            if isinstance(room_tag, str) and room_tag:
                async with self._mapping_lock:
                    self._pump_tasks_by_room[room_tag] = self._ipc_pump_tasks[battle_id]
        except Exception:
            pass

    async def _stop_ipc_pump(self, battle_id: str) -> None:

        task = self._ipc_pump_tasks.pop(battle_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cancel any in-flight handler tasks for this battle
        inflight = self._ipc_inflight_tasks.pop(battle_id, set())
        for t in list(inflight):
            if not t.done():
                t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        
        # Drop semaphore for this battle
        self._exec_semaphores.pop(battle_id, None)

        # Also cleanup room-based pump registry
        try:
            room_tag = self.get_room_tag(battle_id)
            if isinstance(room_tag, str) and room_tag:
                async with self._mapping_lock:
                    other = self._pump_tasks_by_room.pop(room_tag, None)
                if other is not None and other is not task:
                    if not other.done():
                        other.cancel()
                        try:
                            await other
                        except asyncio.CancelledError:
                            pass
        except Exception:
            pass

    # ---- Phase1: room_tag <-> battle_id mapping accessors ----
    def get_room_tag(self, battle_id: str) -> Optional[str]:
        """Return the room_tag associated with a battle_id, or None."""
        return self._battle_to_room.get(battle_id)

    def get_battle_id(self, room_tag: str) -> Optional[str]:
        """Return the battle_id associated with a room_tag, or None."""
        return self._room_to_battle.get(room_tag)

    async def _ipc_receive_pump(self, battle_id: str) -> None:
        """Continuously forward IPC messages, dispatching handlers in parallel.

        This mirrors PSClient.listen() behavior where each incoming websocket
        message is handled in its own task. We cap per-battle concurrency via a
        semaphore to avoid unbounded parallelism.
        """
        try:
            while True:
                # Use room_key to receive: if we have room mapping, prefer it
                room_key = self.get_room_tag(battle_id) or battle_id
                raw = await self.ipc_client_wrapper.recv(room_key, self.player_id)
                
                if not raw:
                    continue

                # Dispatch handling in a background task (parallelize like WS)
                if battle_id not in self._ipc_inflight_tasks:
                    self._ipc_inflight_tasks[battle_id] = set()
                task = asyncio.create_task(self._dispatch_handle_message(battle_id, raw))
                self._ipc_inflight_tasks[battle_id].add(task)
                # Ensure cleanup on task completion
                def _done_cb(t: asyncio.Task, bid: str = battle_id) -> None:
                    try:
                        self._ipc_inflight_tasks.get(bid, set()).discard(t)
                    except Exception:
                        pass
                task.add_done_callback(_done_cb)

                # Exit if battle finished
                battle = self._battles.get(room_key) or self._battles.get(battle_id)
                if battle and getattr(battle, "finished", False):
                    
                    # Do not notify env from pump to avoid duplicate finish signals.
                    break
        except asyncio.CancelledError:
            return
        except Exception as e:
            self._logger.error(f"❌ IPC receive pump error for {battle_id}: {e}")
            return

    async def _dispatch_handle_message(self, battle_id: str, raw: str) -> None:
        """Handle a single raw message with per-battle concurrency limits."""
        # Acquire per-battle semaphore
        sem = self._exec_semaphores.get(battle_id)
        if sem is None:
            sem = asyncio.Semaphore(self._ipc_max_concurrency)
            self._exec_semaphores[battle_id] = sem
        async with sem:
            try:
                await self.ps_client._handle_message(raw)
                # After handling, drive WS-like post-processing in local mode.
                if getattr(self, "mode", None) == "local":
                    try:
                        await self._schedule_finish_callbacks_if_needed(battle_id)
                    except Exception:
                        self._logger.exception("post-handle finish scheduling failed")
                # Publish rQID update (rqid が更新＝発火)。失敗時はログ出力後に停止。
                try:
                    room_key = self.get_room_tag(battle_id) or battle_id
                    battle = self._battles.get(room_key) or self._battles.get(battle_id)
                    rqid = None
                    if battle is not None:
                        lr = getattr(battle, "last_request", None)
                        if isinstance(lr, dict):
                            rqid = lr.get("rqid")
                    notifier = get_global_rqid_notifier()
                    await notifier.publish_rqid_update(self.player_id, rqid)
                except Exception as e:
                    # 仕様: フォールバックせずに停止
                    try:
                        self._logger.exception(
                            "[RQID PUBLISH ERROR] battle_id=%s room_key=%s player=%s rqid=%s err=%s",
                            battle_id,
                            room_key if 'room_key' in locals() else battle_id,
                            self.player_id,
                            rqid if 'rqid' in locals() else None,
                            e,
                        )
                    except Exception:
                        pass
                    raise SystemExit(1)
                # Do not notify env from dispatch; rely solely on Player._battle_finished_callback.
            except Exception:
                # Surface errors but do not crash the pump
                pass

    async def _schedule_finish_callbacks_if_needed(self, battle_id: str) -> None:
        """Drive WS-like finish callback scheduling in local mode.

        - If the current battle is finished but the EnvPlayer callback diagnostics
          show no invocation, trigger the callback path on the correct loop.
        - Idempotent at the environment level due to _finished_enqueued.
        """
        if getattr(self, "mode", None) != "local":
            return
        try:
            room_key = self.get_room_tag(battle_id) or battle_id
            battle = self._battles.get(room_key) or self._battles.get(battle_id)
            if not battle or not getattr(battle, "finished", False):
                return
            # If diagnostics show callback already invoked for this tag, skip.
            diag_tag = getattr(self, "_last_finish_cb_battle_tag", None)
            if isinstance(diag_tag, str) and diag_tag == getattr(battle, "battle_tag", None):
                return
            # Trigger EnvPlayer callback path on POKE_LOOP to match WS semantics.
            try:
                from poke_env.concurrency import POKE_LOOP  # type: ignore
                POKE_LOOP.call_soon_threadsafe(self._battle_finished_callback, battle)
            except Exception:
                # If POKE_LOOP is not accessible, call directly (we're already on loop in local mode)
                try:
                    self._battle_finished_callback(battle)
                except Exception:
                    self._logger.exception("failed to invoke finished callback")
            # Optionally wait briefly and log propagation status
            
        except Exception:
            # Non-fatal: only affects timely propagation
            self._logger.exception("finish scheduling failed")


def create_dual_mode_players(
    env: Any,
    mode: str = "local",
    server_configuration: Optional[ServerConfiguration] = None,
    **kwargs: Any
) -> tuple[DualModeEnvPlayer, DualModeEnvPlayer]:
    """Create a pair of dual-mode players for battle.
    
    Args:
        env: The PokemonEnv instance
        mode: Communication mode ("local" or "online")
        server_configuration: Required for online mode
        **kwargs: Additional arguments passed to players
        
    Returns:
        Tuple of two DualModeEnvPlayer instances
    """
    player_0 = DualModeEnvPlayer(
        env=env,
        player_id="player_0",
        mode=mode,
        server_configuration=server_configuration,
        **kwargs
    )
    
    player_1 = DualModeEnvPlayer(
        env=env,
        player_id="player_1", 
        mode=mode,
        server_configuration=server_configuration,
        **kwargs
    )
    
    return player_0, player_1


def get_mode_from_config(config: dict) -> str:
    """Extract communication mode from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Communication mode ("local" or "online")
    """
    return config.get("battle_mode", "local")


def validate_mode_configuration(mode: str, config: dict) -> None:
    """Validate that the configuration is appropriate for the specified mode.
    
    Args:
        mode: Communication mode
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid for the mode
    """
    if mode == "online":
        if "pokemon_showdown" not in config:
            raise ValueError("Online mode requires 'pokemon_showdown' configuration")
        if "servers" not in config["pokemon_showdown"]:
            raise ValueError("Online mode requires server configuration")
    elif mode == "local":
        local_config = config.get("local_mode", {})
        if "max_processes" in local_config and local_config["max_processes"] < 1:
            raise ValueError("local_mode.max_processes must be at least 1")
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'local' or 'online'")
