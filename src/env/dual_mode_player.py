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

class IPCClientWrapper:
    """IPC client wrapper to manage a Node.js process and JSON message exchange."""

    def __init__(self, node_script_path: str, logger: Optional[logging.Logger] = None) -> None:
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
        await ctrl.create_battle(format_id, players, seed)

    async def create_battle_by_room(self, room_tag: str, format_id: str, players: list[dict], seed: Optional[list[int]] = None) -> None:
        """Create a battle referenced by `room_tag` instead of a numeric battle_id.

        This helper registers/obtains a controller keyed by `room_tag` and
        delegates the create_battle call. This allows Phase3-style room_tag
        based creation while keeping backwards-compatible APIs available.
        """
        from src.env.controller_registry import ControllerRegistry  # type: ignore

        # Create or fetch controller using room_tag as the primary identifier
        ctrl = ControllerRegistry.get_or_create(self.node_script_path, room_tag, room_tag, logger=self.logger)
        # Ensure controller process is running
        if not await ctrl.is_alive():
            await ctrl.connect()
        await ctrl.create_battle(format_id, players, seed)

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

        if timeout is None:
            return await queue.get()
        else:
            return await asyncio.wait_for(queue.get(), timeout=timeout)

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
            self._logger.info(f"🌐 Online mode player initialized for {player_id}")
        else:
            # Local mode initialization - Phase 4 adds full_ipc support
            if server_configuration is None:
                # Create a valid server configuration for compatibility
                server_configuration = ServerConfiguration("localhost", 8000)
            
            if full_ipc:
                # Phase 4: Full IPC mode - completely disable WebSocket
                self._logger.info(f"🚀 Phase 4: Initializing full IPC mode for {player_id}")
                kwargs['start_listening'] = False  # Disable WebSocket completely
                
                try:
                    super().__init__(
                        env=env,
                        player_id=player_id,
                        server_configuration=server_configuration,
                        **kwargs
                    )
                    self._logger.info(f"✅ Full IPC mode player initialized for {player_id} (WebSocket disabled)")
                except Exception as e:
                    self._logger.error(f"❌ Full IPC initialization failed for {player_id}: {e}")
                    raise RuntimeError(f"Full IPC mode requires working IPC infrastructure: {e}") from e
            else:
                # Phase 3: Local mode with WebSocket fallback
                super().__init__(
                    env=env,
                    player_id=player_id,
                    server_configuration=server_configuration,
                    **kwargs
                )
                self._logger.info(f"✅ Local mode player initialized for {player_id} with IPC capability (WebSocket fallback for Phase 3)")
        
        # For local mode, initialize IPC communicator and (if requested) establish full IPC
        if mode == "local":
            # Initialize communicator (create IPCClientWrapper)
            self._initialize_communicator()
            if full_ipc:
                # Phase 4: Full IPC mode - must establish working connection
                self._logger.info(f"🔌 Phase 4: Establishing mandatory IPC connection for {self.player_id}")
                self._establish_full_ipc_connection()
            else:
                # Phase 3: IPC capability initialized
                self._logger.info(f"🚀 IPC capability initialized for {self.player_id} (WebSocket fallback)")
    
    def _initialize_communicator(self) -> None:
        """Initialize IPCClientWrapper for local mode."""
        if self.mode != "local":
            return
        self._logger.info(f"Local IPC mode configured for player {self.player_id}")
        # Create the IPC client wrapper; actual process start is deferred
        self.ipc_client_wrapper = IPCClientWrapper(
            node_script_path=self.ipc_script_path,
            logger=self._logger,
        )
    
    
    
    
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
        
        self._logger.info(f"🎮 IPC battle_against called for {self.player_id} vs {opponent.username}")
        
        for i in range(n_battles):
            # Only player_0 generates battle ID to avoid duplicates
            if self.player_id == "player_0":
                # Generate unique battle ID
                # Phase3: Generate room_tag and create battle by room
                timestamp = int(time.time() * 1000)
                random_id = random.randint(1000, 9999)
                format_id = self._format if hasattr(self, '_format') else "gen9randombattle"
                room_tag = f"battle-{format_id}-{timestamp}-{random_id}"
                self._logger.info(f"🎲 [player_0] Creating battle (room_tag): {room_tag}")
                # Create battle via IPC using room_tag
                await self._create_ipc_battle_by_room(room_tag, opponent)
            else:
                # player_1 waits for invitation
                self._logger.info(f"⏳ [player_1] Waiting for battle invitation...")
                battle_id = await self._wait_for_battle_invitation()
                self._logger.info(f"📨 [player_1] Received invitation for battle: {battle_id}")
            
            # Both players now wait for battle to be ready
            self._logger.info(f"🔄 [{self.player_id}] Waiting for battle {battle_id} to be ready...")
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
                        self._logger.info(f"Battle {battle_id} mapped to room {room_tag}")
                    except Exception:
                        self._logger.exception("Failed to store mapping for %s", battle_id)
                else:
                    # Defensive: log a warning if room_tag not yet available
                    self._logger.warning(f"room_tag not available immediately for {battle_id}")
            except Exception as e:
                self._logger.error(f"Failed to retrieve room_tag for {battle_id}: {e}")
            
            # Register battle with both players
            self._battles[battle_id] = None  # Will be populated when battle starts
            opponent._battles[battle_id] = None
            
            # Notify both players about the battle (invitation for player_1)
            await self._handle_battle_start(battle_id)
            await opponent._handle_battle_start(battle_id)
            # 通知: 招待キューへ投入（player_1 はここを待機）
            if hasattr(opponent, '_ipc_invitations'):
                await opponent._ipc_invitations.put(battle_id)
            
            self._logger.info(f"✅ IPC battle created: {battle_id}")

        except Exception as e:
            self._logger.error(f"❌ Failed to create IPC battle: {e}")
            raise

    async def _wait_for_battle_invitation(self) -> str:
        """player_1 が招待（battle_id）を待つ。"""
        battle_id = await self._ipc_invitations.get()
        # 自身のラッパーを同じControllerに接続
        # battle_id may be a room_tag in Phase3; ensure controller for that id
        await self.ipc_client_wrapper._ensure_controller(battle_id)
        
        # Register battle placeholder
        self._battles[battle_id] = None
        
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

            # Register battle using room_tag as key
            self._battles[room_tag] = None
            opponent._battles[room_tag] = None

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

            self._logger.info(f"✅ IPC battle created by room: {room_tag}")

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
            # store battle under local battle_id to keep controller mapping stable
            self._battles[battle_id] = battle
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
        """Wait until first request is processed (without consuming IPC queues)."""
        try:
            timeout = 10.0  # seconds
            start_time = time.time()
            while time.time() - start_time < timeout:
                # prefer room_tag key if available
                room_key = self.get_room_tag(battle_id) or battle_id
                battle = self._battles.get(room_key) or self._battles.get(battle_id)
                # When first |request| arrives, poke-env sets last_request
                if battle is not None and getattr(battle, "last_request", None):
                    self._logger.info(f"✅ [{self.player_id}] Battle {battle_id} is ready!")
                    return
                await asyncio.sleep(0.05)
            raise TimeoutError(f"Battle {battle_id} did not start within {timeout} seconds")
        except Exception as e:
            self._logger.error(f"❌ [{self.player_id}] Error waiting for battle ready: {e}")
            raise
    
    async def _wait_for_battle_completion(self, battle_id: str):
        """Wait for a battle to complete."""
        # prefer room_tag key if available
        room_key = self.get_room_tag(battle_id) or battle_id
        while room_key in self._battles or battle_id in self._battles:
            battle = self._battles.get(room_key) or self._battles.get(battle_id)
            if battle and battle.finished:
                break
            await asyncio.sleep(0.1)
        
        self._logger.info(f"🏁 Battle {battle_id} completed")
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
        """Continuously forward IPC messages to PSClient handler for this player."""
        try:
            while True:
                # Use room_key to receive: if we have room mapping, prefer it
                room_key = self.get_room_tag(battle_id) or battle_id
                raw = await self.ipc_client_wrapper.recv(room_key, self.player_id)
                if not raw:
                    continue
                await self.ps_client._handle_message(raw)  # reuse existing PSClient path

                # Exit if battle finished
                battle = self._battles.get(room_key) or self._battles.get(battle_id)
                if battle and getattr(battle, "finished", False):
                    break
        except asyncio.CancelledError:
            return
        except Exception as e:
            self._logger.error(f"❌ IPC receive pump error for {battle_id}: {e}")
            return


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
