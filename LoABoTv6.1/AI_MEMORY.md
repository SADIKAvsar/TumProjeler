# AI_MEMORY.md
## LoABot v6.0 - Unified Architecture + Agent Memory

This file is the single source of truth for both:
- architecture intent (from `ARCHITECTURE_v6.md`)
- operational memory and critical fixes (from old `AI_MEMORY.md`)

If a new agent takes over, read this file first.

---

## 1) Project Goal

LoABot v6.0 is a fully local, autonomous MMORPG bot system.
Main goals:
- low latency decisions
- no cloud dependency for core decision flow
- robust long-run operation (stuck recovery, phase recovery, restart safety)
- continuous learning without catastrophic forgetting

---

## 2) AI and Control Stack

### Tier-1: Deterministic Engine
- Rule-based flow control
- OpenCV template and anchor checks
- Sequence execution (`click`, `boss_secimi`, `key`)
- Safety fallbacks and restart logic

### Tier-2: Reflex Model (PyTorch)
- Model: `TemporalAgenticNet`
- Input: 9-channel stacked frames `[T-2, T-1, T]`
- Backbone: ResNet18 / EfficientNet-B0
- Output: normalized coordinate regression `(x, y)`
- Current policy: pure regression output, no old `is_stagnant` early-return path

### Tier-3: Strategic Brain (Local LLM)
- Local Ollama + Qwen multimodal
- JSON-oriented strategic outputs
- Cloud LLM APIs are not part of production design

---

## 3) Core Runtime (v6)

### Core modules
- `core/game_state.py`: thread-safe centralized state (`RLock`, `snapshot`, `update`, `set_phase`)
- `core/event_bus.py`: pub/sub event router
- `core/event_types.py`: canonical event constants (`EVT.*`)
- `compat/bot_bridge.py`: backward compatibility bridge (`self.bot.*` <-> `GameState` + `EventBus`)
- `core/bootstrap.py`: startup wiring

### Bootstrap contract
In `loabot_main.py`:
```python
from core.bootstrap import bootstrap_v6
self.game_state, self.event_bus, self.bridge = bootstrap_v6(self)
```

If core import fails, controlled fallback to v5.9-compatible mode is allowed.

### GameState canonical fields (important)
- lifecycle: `running`, `paused`
- phase machine: `phase`, `stage`, `phase_reason`, `phase_extra`, `phase_ts`
- combat: `attacking_target`, `active_event`, `in_combat`
- location: `current_region`, `current_location`
- vision: `last_vision_ts`, `detected_objects`, `enemy_visible`
- boss: `next_boss_name`, `next_boss_spawn_ts`, `last_boss_killed`, `last_boss_kill_ts`
- metrics: `recording_active`, `inference_ms`, `fps`

---

## 4) Event-Driven Contracts

Critical events in daily flow:
- `PHASE_CHANGED`
- `COMBAT_STARTED` / `COMBAT_FINISHED`
- `BOSS_KILLED`
- `LOCATION_CHANGED`
- `MISSION_STARTED` / `MISSION_ENDED`
- `STUCK_DETECTED`

Design rule:
- New modules should publish/subscribe via EventBus.
- Legacy modules can still run through BotBridge interceptors.

---

## 5) Boss Automation Algorithm (Production Flow)

### High-level loop
1. Collect ready targets (`_collect_ready_targets`)
2. Select target + protocol (`_select_target_with_protocol`)
3. Start recording
4. Execute precise flow (`execute_precise_boss_flow`)
5. Post-attack logic + chain handling
6. Always reset combat state (`_reset_combat_state`) in all exits

### Protocol selection
- `FULL_MENU_SEQUENCE` when region mismatch
- `SHORT_LIST_SEQUENCE` when same layer
- `DIRECT_BOSS_SELECTION` for direct anchor path

### Timer integrity rule (critical)
`recalculate_times(target, attack_start)` must run on both success and fail.
Reason: boss respawn period is fixed; otherwise countdown can freeze at `00:00:00`.

### GUI boss kill counter sync
After kill/fail, update AI memory metrics:
- `memory.update_boss_performance(..., success=True/False)`
This keeps GUI `Boss: X kill` in sync.

---

## 6) Navigation Stability Logic (Pulse + Evasion + Re-anchor)

Implemented in `boss_manager.py`.

### Navigation Pulse Check
Method: `_check_navigation_pulse(...)`
- capture gray frame A
- wait 1 second
- capture gray frame B
- apply masked `absdiff` motion ratio

### If motion is low
1. Evasion combo: `Space -> q -> Space`
2. If low-motion persists for configured duration, run re-anchor:
   - `boss_list_ac -> boss_secimi -> z`

### Area lock guard (critical fix)
If area is already found (`area_found` or `target["_area_check_ok"]`), pulse/evasion must stop.
Character standing still in boss area is expected behavior, not stuck behavior.

---

## 7) Strategic Wait Priority (Same Layer 90s Guard)

In `combat_manager.py/check_strategic_wait`:

Priority order:
1. Hard guard: if next same-layer boss ready window is within 90s, do NOT return to `EXP_FARM`.
2. Same-layer threshold rule: keep waiting on map when near ready window.
3. AI strategic decision may suggest wait/return.
4. Rule fallback if AI decision is absent.

Return logs are explicitly tagged with source:
- `source=rule_same_layer_fast90`
- `source=ai`
- `source=rule`
- `source=no_upcoming_boss`

---

## 8) Reliability Fixes Already Applied

- State lock prevention with centralized `_reset_combat_state`
- Popup lock crash fix (`RLock` compatible action lock check)
- Navigation pulse + evasion + re-anchor
- Area lock guard to avoid false evasion near boss
- Boss timer recalc on both success/fail
- GUI boss kill counter sync via memory updates

---

## 9) Training Pipeline Memory

`train_agentic.py` supports fine-tuning continuation:
- `--weights <checkpoint.pt>` loads pretrained `model_state_dict`
- Used to prevent catastrophic forgetting when training on newer sessions

Current expected training log root:
- `D:\\LoABot_Training_Data\\runtime_data\\training_logs`

---

## 10) Operations Quick Checklist

Before long run:
1. Confirm v6 bootstrap active in logs
2. Confirm same-layer 90s guard logs appear when expected
3. Confirm pulse events stop after area is found
4. Confirm boss kill counter increments in GUI
5. Confirm no `00:00:00` countdown freeze after fail cycles

Before training:
1. Sanitize broken sessions
2. Use `--weights` for continuation training
3. Validate dataset split and session counts

---

## 11) Non-Negotiable Rules

1. Keep system local-first and deterministic-first.
2. Preserve Tier hierarchy: Tier-1 safety, Tier-2 reflex, Tier-3 strategy.
3. All critical transitions must be observable in logs/events.
4. New agents must read this file before code edits.
5. Avoid architecture drift between docs and code.

---

## 12) Status Summary

LoABot v6.0 is running with a hybrid architecture:
- rule-based stability layer
- reflex coordinate model
- local strategic reasoning
- core event/state backbone

Current focus is production hardening: stable navigation, consistent phase resets,
robust chain handling, and safe incremental model improvement.

---

## 13) v6.1 Refactoring Summary

### Module Consolidation (5 files removed)
- `event_types.py` → merged into `event_bus.py` (EVT class now lives there)
- `bootstrap.py` → merged into `bot_bridge.py` (bootstrap_v6 function now there)
- `popup_manager.py` + `location_manager.py` → merged into `environment_manager.py`
- `video_recorder_integration.py` → deleted (was only patch instructions)

### Import Changes
All modules that used `from core.event_types import EVT` now use `from event_bus import EVT`.
All modules that used `from core.bootstrap import bootstrap_v6` now use `from bot_bridge import bootstrap_v6`.
All modules that used `from location_manager import LocationManager` or `from popup_manager import PopupManager`
now use `from environment_manager import LocationManager, PopupManager`.

### Navigation Evasion → PvP Transfer
- `boss_manager._run_navigation_evasion_combo()` removed
- `boss_manager._last_nav_evasion_ts` removed
- Same combo `Space → q → Space` now in `PvPManager._run_evasion_combo()`
- Triggered only when PvP hp damage is detected (not on navigation stall)
- Config: `pvp.evasion_cooldown_sn` (default 3.0)

### Video Recording → Manual Only
- `boss_manager.automation_thread` no longer calls `auto_start_recording` / `auto_stop_recording`
- `event_manager._run_scheduled_event` no longer calls recording methods
- Recording is controlled exclusively via GUI button or manual command
- `auto_start_recording` / `auto_stop_recording` methods remain for programmatic use

### GameState & Bridge Simplification
- `_ATTR_MAP` reduced (removed `_global_mission_reason`, `_global_mission_extra`, `_global_phase_ts`)
- Sync thread interval increased from 0.5s to 2.0s (interceptors handle real-time sync)
- Unused dunder methods (`__getitem__`, `__setitem__`, `__contains__`) removed from GameState
