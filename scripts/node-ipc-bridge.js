/**
 * Minimal IPC bridge for Pokemon Showdown using NDJSON over stdin/stdout.
 * One battle per Node.js process is recommended, but this supports multiple IDs.
 *
 * Message spec (Node ⇄ Python):
 *  - From Python (control): {"type":"ping"} → {"type":"pong"}
 *  - From Python (create): {"type":"create_battle","battle_id","format","players":[{name,team},{name,team}],"seed"?}
 *  - From Python (protocol to player): {"player_id":"p1|p2","battle_id","data":"..."}
 *  - From Node (protocol to Python): {"type":"protocol","battle_id","target_player":"p1|p2","data":"..."}
 */

/* eslint-disable no-console */
const readline = require('readline');

// Resolve Showdown dist modules
const path = require('path');
const showdownDist = path.resolve(__dirname, '../pokemon-showdown/dist/sim');
const { BattleStream, getPlayerStreams } = require(path.join(showdownDist, 'battle-stream.js'));

// Helpers
function deriveRoomTag(format, battleId) {
  // If battleId is already a canonical room tag for this format, return it
  const asStr = String(battleId || '').trim();
  if (asStr.startsWith(`battle-${format}-`)) return asStr;

  // Otherwise, derive the id part. If battleId already starts with `battle-`
  // but for a different format, strip the leading `battle-` prefix to avoid
  // constructing tags like `battle-<format>-battle-<other>-...`.
  const idPart = asStr.startsWith('battle-') ? asStr.slice(7) : asStr;
  return `battle-${format}-${idPart}`;
}

function writeJSON(obj) {
  try {
    process.stdout.write(JSON.stringify(obj) + '\n');
  } catch (e) {
    // last-resort logging
    console.error('[IPC_NODE] Failed to stringify JSON:', e?.message);
  }
}

function sanitizePlayerCommand(raw) {
  if (typeof raw !== 'string') return '';
  // If payload contains a battle tag like "battle-xxx|/choose ...", extract after last '|'
  const idx = raw.lastIndexOf('|');
  const cmd = idx >= 0 ? raw.slice(idx + 1) : raw;
  // Do not forcibly append a trailing newline; send as-is
  return cmd;
}

class BattleSession {
  constructor(battleId, format, players, seed, onEnd) {
    this.battleId = battleId;
    this.format = format;
    this.players = players;
    this.seed = seed;
    this.streams = null;
    this.closed = false;
    this.roomTag = deriveRoomTag(this.format, this.battleId);
    // Counter used to inject rqid for request/init messages when missing.
    // Only used when the JSON payload does not already contain an `rqid`.
    this.rqidCounter = 0;
    this.onEnd = typeof onEnd === 'function' ? onEnd : null;
  }

  async init() {
    const spec = { formatid: this.format };
    if (this.seed) spec.seed = this.seed;
    const p1spec = { name: this.players?.[0]?.name || 'p1', team: this.players?.[0]?.team };
    const p2spec = { name: this.players?.[1]?.name || 'p2', team: this.players?.[1]?.team };

    const battleStream = new BattleStream();
    const streams = getPlayerStreams(battleStream);
    this.streams = streams;

    // Attach per-player output handlers (player perspective)
    this._attachOut(streams.p1, 'p1');
    this._attachOut(streams.p2, 'p2');

    // Initialize battle through omniscient stream
    const initCommands = `>start ${JSON.stringify(spec)}\n>player p1 ${JSON.stringify(p1spec)}\n>player p2 ${JSON.stringify(p2spec)}\n`;
    // Fire and forget
    void streams.omniscient.write(initCommands);
  }

  async _attachOut(readable, target) {
    const battleId = this.battleId;
    const roomTag = this.roomTag;
    (async () => {
      try {
        for await (const chunk of readable) {
          // chunk may contain multiple lines
          const lines = String(chunk).split('\n');
          for (const line of lines) {
            if (!line.trim()) continue;

            // If this is a |request| or |init| line with a JSON payload, ensure it
            // contains an `rqid` field. Parse and inject only when missing.
            try {
              let handled = false;
              if (line.startsWith('|request|') || line.startsWith('|init|')) {
                const prefix = line.startsWith('|request|') ? '|request|' : '|init|';
                const payloadText = line.slice(prefix.length).trim();
                if (payloadText) {
                  try {
                    const payload = JSON.parse(payloadText);
                    if (!Object.prototype.hasOwnProperty.call(payload, 'rqid')) {
                      this.rqidCounter += 1;
                      payload.rqid = this.rqidCounter;
                    }
                    const newLine = prefix + JSON.stringify(payload);
                    writeJSON({ type: 'protocol', battle_id: battleId, room_tag: roomTag, target_player: target, data: newLine });
                    handled = true;
                  } catch (e) {
                    // JSON parse failed, fall through to send original line
                  }
                }
              }

              if (handled) continue;
            } catch (e) {
              // Defensive: any unexpected error should not break the read loop
            }

            writeJSON({ type: 'protocol', battle_id: battleId, room_tag: roomTag, target_player: target, data: line });

            // Detect battle end signals and trigger session cleanup once
            try {
              if (!this.closed && (line.startsWith('|win|') || line.startsWith('|tie|'))) {
                this.closed = true;
                if (this.onEnd) {
                  try { this.onEnd(battleId); } catch {}
                }
                // Inform Python side explicitly (optional)
                writeJSON({ type: 'battle_closed', battle_id: battleId, room_tag: roomTag });
              }
            } catch {}
          }
        }
      } catch (e) {
        console.error(`[IPC_NODE] Read loop error for ${battleId}/${target}:`, e?.message);
      }
    })();
  }

  sendToPlayer(playerId, rawData) {
    if (!this.streams) throw new Error('BattleSession not initialized');
    if (playerId !== 'p1' && playerId !== 'p2') throw new Error(`Invalid playerId: ${playerId}`);
    const cmd = sanitizePlayerCommand(rawData);
    return this.streams[playerId].write(cmd);
  }

  async close() {
    this.closed = true;
    // Streams will end automatically when battle finishes
  }
}

class IPCBridge {
  constructor() {
    this.sessions = new Map(); // battle_id -> BattleSession
    this.rl = readline.createInterface({ input: process.stdin });
  }

  start() {
    this.rl.on('line', (line) => this._onLine(line));
    process.on('SIGINT', () => this._shutdown(130));
    process.on('SIGTERM', () => this._shutdown(143));
    console.error('[IPC_NODE] Bridge started');
  }

  _onLine(line) {
    let msg;
    try {
      msg = JSON.parse(line.trim());
    } catch (e) {
      return writeJSON({ type: 'error', code: 'JSON_PARSE_ERROR', message: e?.message });
    }
    this._handleMessage(msg);
  }

  _handleMessage(msg) {
    const { type } = msg || {};
    if (type === 'ping') return writeJSON({ type: 'pong' });
    if (type === 'create_battle') return this._handleCreateBattle(msg);

    // Default: treat as protocol to player
    const { player_id, battle_id, data } = msg;
    if (!battle_id || !player_id) {
      return writeJSON({ type: 'error', code: 'INVALID_PROTOCOL', message: 'Missing battle_id/player_id' });
    }
    const session = this.sessions.get(battle_id);
    if (!session) return writeJSON({ type: 'error', code: 'BATTLE_NOT_FOUND', message: `Unknown battle_id: ${battle_id}` });
    try {
      void session.sendToPlayer(player_id, data);
    } catch (e) {
      writeJSON({ type: 'error', code: 'SEND_FAILED', message: e?.message, battle_id });
    }
  }

  async _handleCreateBattle(msg) {
    const { battle_id, format, players, seed } = msg;
    if (!battle_id || !format || !Array.isArray(players) || players.length !== 2) {
      return writeJSON({ type: 'error', code: 'INVALID_CREATE_BATTLE', message: 'battle_id/format/players[2] required' });
    }
    if (this.sessions.has(battle_id)) {
      return writeJSON({ type: 'error', code: 'BATTLE_EXISTS', message: `Battle already exists: ${battle_id}` });
    }
    try {
      const session = new BattleSession(battle_id, format, players, seed, (id) => this._onSessionEnd(id));
      this.sessions.set(battle_id, session);
      await session.init();
      // Acknowledge creation; first protocol lines will arrive asynchronously
      writeJSON({ type: 'battle_created', battle_id, room_tag: session.roomTag, success: true });
    } catch (e) {
      writeJSON({ type: 'error', code: 'CREATE_FAILED', message: e?.message, battle_id });
    }
  }

  _onSessionEnd(battleId) {
    const sess = this.sessions.get(battleId);
    if (sess) {
      try { void sess.close(); } catch {}
    }
    this.sessions.delete(battleId);
  }

  _shutdown(code) {
    for (const session of this.sessions.values()) {
      void session.close();
    }
    writeJSON({ type: 'exit', code });
    process.exit(0);
  }
}

// Start bridge
new IPCBridge().start();
