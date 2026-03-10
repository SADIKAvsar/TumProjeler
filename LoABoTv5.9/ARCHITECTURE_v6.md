# ARCHITECTURE_v6.md
## LoABot v6.0 Architecture Reference

This file is kept as an entry point.
The canonical architecture + operational memory document is now:

- `AI_MEMORY.md`

Reason:
- prevent drift between architecture notes and production fixes
- keep one maintained source for agent handover
- keep algorithm and runtime contracts easier to follow

---

## Quick Architecture Snapshot

LoABot v6.0 uses:
- `GameState` for centralized thread-safe runtime state
- `EventBus` for module decoupling (publish/subscribe)
- `BotBridge` for backward compatibility with legacy `self.bot.*` calls
- 3-tier AI/control stack:
  - Tier-1 deterministic rules and OpenCV sequences
  - Tier-2 PyTorch temporal coordinate regression
  - Tier-3 local strategic reasoning (Ollama/Qwen)

For full contracts, flow, and production rules see `AI_MEMORY.md`.
