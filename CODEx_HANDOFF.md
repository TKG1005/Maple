*** Begin CODEx_HANDOFF.md (テンプレート) ***
Title: Codex Handoff — rqid patch (IPC)
Branch: feature/node-ipc-server-development

Summary:
- 目的：子プロセス（IPC）経路で送られる sim の `|request|{...}` メッセージに `rq
id` を追加する。
- 対象ファイル: `pokemon-showdown/lib/process-manager.ts`
- 現状: 親の `RoomBattle.receive` は依然として受信時に `rqid` を上書きするため、
 websocket 挙動は変わらないことを確認済み。
- 状態: パッチ本体を作成した（未適用, または適用に失敗した）。ファイル編集がこの
環境でブロックされているため、手動/再現可能な自動手順を用意。

Patch file (if present):
- /tmp/rqid.patch もしくは repo の tools/rqid.patch

What to run next (recommended):
1. Ensure on branch: 
   git checkout feature/node-ipc-server-development
2. Optional: stash uncommitted changes:
   git stash -u
3. Check patch:
   git apply --check /tmp/rqid.patch
4. Apply patch:
   git apply /tmp/rqid.patch
5. Verify changes:
   git --no-pager diff -- pokemon-showdown/lib/process-manager.ts | sed -n '1,16
0p'
   rg 'rqidCounter|\\|request\\|' pokemon-showdown/lib/process-manager.ts -n
6. Build / type-check:
   npm run build || npx tsc --noEmit
7. Run quick check:
   Start server (dev command) and verify child→parent messages add `rqid`.
8. If OK, commit:
   git add pokemon-showdown/lib/process-manager.ts
   git commit -m "Add rqid injection to RawProcessManager.pipeStream (IPC)"
   git push origin feature/node-ipc-server-development

Notes:
- If patch apply fails because file changed, open `pokemon-showdown/lib/process-
manager.ts`, search for function `async pipeStream(...)` and replace the body pe
r `/tmp/rqid.patch`.
- If this environment prevents file writes, copy the file contents and apply pat
ch on a developer machine with write access.

Verification checklist:
- The child process to parent messages containing `|request|{...}` now include a
n `rqid` field in that JSON portion.
- Parent `RoomBattle` behavior unchanged: it will still set its own `this.rqid` 
when sideupdate/request arrives.

*** End CODEx_HANDOFF.md ***