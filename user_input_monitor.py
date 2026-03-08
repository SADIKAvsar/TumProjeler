"""
user_input_monitor.py — Merkezi Global Input Manager (v2.0)
===========================================================
Sistemdeki TEK pynput hook kaynağı.  Fare ve klavye OS-seviyesi
dinleyicilerini başlatır; gelen ham olayları kayıtlı subscriber'lara
(observer pattern) dağıtır.

Neden tek merkez?
  - Birden fazla pynput.Listener aynı OS hook slot'unu kullanır →
    double-fire ve input gecikmesine yol açar.
  - Bu modül hook'ları bir kez açar, diğer modüller (VideoRecorder,
    TrainingLogger, vb.) subscribe() ile olayları alır.

Subscriber Callback İmzası:
    def on_input(event: InputEvent) -> None: ...

InputEvent alanları:
    source     : "user" | "bot"
    event_type : "mouse_click" | "key_press"
    data       : {"x": int, "y": int, "button": str} veya {"key": str}
    ts_mono    : float (time.monotonic)
    phase      : str (aktif oyun fazı)
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional

try:
    import pyautogui
except Exception:
    pyautogui = None

if TYPE_CHECKING:
    pass

from utils import BOT_IS_CLICKING_EVENT as _BOT_CLICK_EVT

# Bot'un press_key() çağrısında set edilecek global event
BOT_IS_PRESSING_KEY_EVENT = threading.Event()

# Agentic eğitim için izlenen tuşlar
_TRACKED_KEYS = frozenset({"a", "q", "v", "z"})

# Debounce süresi (ms)
_DEBOUNCE_MS: float = 100.0


# ══════════════════════════════════════════════════════════════════════
#  INPUT EVENT
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class InputEvent:
    """Observer'lara dağıtılan standart girdi olayı."""
    source: str                         # "user" | "bot"
    event_type: str                     # "mouse_click" | "key_press"
    data: Dict[str, object]             # {"x": …, "y": …} veya {"key": …}
    ts_mono: float                      # time.monotonic
    phase: str = "UNKNOWN_PHASE"


# Subscriber callback tipi
InputCallback = Callable[[InputEvent], None]


# ══════════════════════════════════════════════════════════════════════
#  USER INPUT MONITOR
# ══════════════════════════════════════════════════════════════════════

class UserInputMonitor:
    """
    Tek merkezi pynput hook yöneticisi + observer hub.

    Kullanım:
        monitor = UserInputMonitor(bot)
        monitor.subscribe("video_rec", my_callback)
        ...
        monitor.unsubscribe("video_rec")
        monitor.shutdown()
    """

    def __init__(self, bot):
        self.bot = bot
        self._active = True
        self._lock = threading.Lock()

        # Debounce
        self._last_key_ts: float = 0.0
        self._last_click_ts: float = 0.0

        # Subscriber registry — {name: callback}
        self._subscribers: Dict[str, InputCallback] = {}
        self._sub_lock = threading.Lock()

        # pynput listeners
        self._keyboard_listener = None
        self._mouse_listener = None

        self._start_keyboard_listener()
        self._start_mouse_listener()
        bot.log("UserInputMonitor v2: Merkezi input hub aktif.")

    # ── Subscriber API ─────────────────────────────────────────────

    def subscribe(self, name: str, callback: InputCallback) -> None:
        """Bir modülü input olaylarına abone et."""
        with self._sub_lock:
            self._subscribers[name] = callback
        self.bot.log(f"UserInputMonitor: '{name}' subscribe oldu.", level="DEBUG")

    def unsubscribe(self, name: str) -> None:
        """Modül aboneliğini kaldır."""
        with self._sub_lock:
            self._subscribers.pop(name, None)
        self.bot.log(f"UserInputMonitor: '{name}' unsubscribe oldu.", level="DEBUG")

    # ── Bezier Click (insan benzeri fare hareketi) ─────────────────

    def bezier_click(self, x: int, y: int, duration: float = 0.22) -> bool:
        """Fareyi cubic Bezier eğriyle hedefe taşır ve tıklar."""
        if pyautogui is None:
            return False
        try:
            sx, sy = pyautogui.position()
            tx, ty = int(x), int(y)
            dx, dy = float(tx - sx), float(ty - sy)
            dist = max(1.0, math.hypot(dx, dy))

            steps = int(max(12, min(70, dist / 14.0)))
            jitter = max(20.0, min(170.0, dist * 0.35))
            c1x = float(sx) + (dx * 0.33) + random.uniform(-jitter, jitter)
            c1y = float(sy) + (dy * 0.33) + random.uniform(-jitter, jitter)
            c2x = float(sx) + (dx * 0.66) + random.uniform(-jitter, jitter)
            c2y = float(sy) + (dy * 0.66) + random.uniform(-jitter, jitter)

            def _bp(t):
                o = 1.0 - t
                bx = o**3*sx + 3*o**2*t*c1x + 3*o*t**2*c2x + t**3*tx
                by = o**3*sy + 3*o**2*t*c1y + 3*o*t**2*c2y + t**3*ty
                return bx, by

            total = max(0.08, float(duration))
            for i in range(1, steps + 1):
                bx, by = _bp(i / float(steps))
                pyautogui.moveTo(int(round(bx)), int(round(by)), duration=0)
                time.sleep(max(0.001, (total / steps) * random.uniform(0.85, 1.15)))

            pyautogui.click(tx, ty, duration=0)
            return True
        except Exception as exc:
            self.bot.log(f"UserInputMonitor: Bezier click hatasi: {exc}", level="WARNING")
            return False

    # ── Shutdown ───────────────────────────────────────────────────

    def shutdown(self):
        """Bot kapanırken — hook'ları ve subscriber'ları temizler."""
        self._active = False
        for listener in (self._keyboard_listener, self._mouse_listener):
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
        with self._sub_lock:
            self._subscribers.clear()
        self.bot.log("UserInputMonitor: Kapatıldı.")

    # ── Broadcast ─────────────────────────────────────────────────


    def _should_log_user_input(self) -> bool:
        """
        Kullanıcı inputlarının log dosyasına yazılıp yazılmayacağını belirler.

        Kurallar:
          - Aktif bir aksiyon döngüsündeyse (boss saldırısı, event, navigasyon)
            → True  (her input eğitim verisi olarak değerlidir)
          - EXP_FARM'da boşta bekliyorsa (IDLE + hedef yok + event yok)
            → False (log şişmesini önle)

        NOT: Bu sadece bot.log() ve log_training_action() çağrılarını kontrol eder.
        _broadcast() her zaman çalışır — subscriber'lar (VideoRecorder vb.)
        olayları almaya devam eder.
        """
        # 1. Aktif saldırı hedefi varsa → logla
        if getattr(self.bot, "attacking_target_aciklama", None):
            return True

        # 2. Aktif event varsa → logla
        if getattr(self.bot, "active_event", None):
            return True

        # 3. _global_phase aktif bir fazda mı?
        phase = str(getattr(self.bot, "_global_phase", "")).strip().upper()
        if phase in {"NAV_PHASE", "COMBAT_PHASE", "LOOT_PHASE", "EVENT_PHASE"}:
            return True

        # 4. Video kayıt aktifse → logla (eğitim verisi toplanıyor)
        vid = getattr(self.bot, "video_recorder", None)
        if vid is not None and getattr(vid, "is_recording", False):
            return True

        # Hiçbir koşul sağlanmadı → EXP_FARM boşta → sustur
        return False

    def _broadcast(self, event: InputEvent) -> None:
        """Olayı tüm subscriber'lara dağıt (hata izolasyonu ile)."""
        with self._sub_lock:
            callbacks = list(self._subscribers.values())
        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:
                self.bot.log(
                    f"UserInputMonitor: Subscriber callback hatasi: {exc}",
                    level="WARNING",
                )

    # ── Klavye Hook ───────────────────────────────────────────────

    def _start_keyboard_listener(self):
        try:
            from pynput import keyboard as _kb

            def on_press(key):
                if not self._active:
                    return
                is_bot = BOT_IS_PRESSING_KEY_EVENT.is_set()

                key_char = self._extract_key_char(key)
                if key_char is None:
                    return

                # Debounce
                now = time.monotonic()
                with self._lock:
                    if (now - self._last_key_ts) * 1000 < _DEBOUNCE_MS:
                        return
                    self._last_key_ts = now

                source = "bot" if is_bot else "user"
                phase = getattr(self.bot, "_global_phase", "UNKNOWN_PHASE")

                event = InputEvent(
                    source=source,
                    event_type="key_press",
                    data={"key": key_char},
                    ts_mono=now,
                    phase=phase,
                )

                # Kullanıcı olaylarını TrainingLogger'a bildir
                # ★ FIX: EXP_FARM boşta iken loglamayı sustur (log şişmesi önlenir)
                if source == "user" and self._should_log_user_input():
                    action_label = (
                        f"key_{key_char}" if key_char in _TRACKED_KEYS
                        else f"user_key_{key_char}"
                    )
                    self.bot.log(
                        f"[InputHub] Kullanici tusu: {key_char} -> {action_label}",
                        level="DEBUG",
                    )
                    self.bot.log_training_action(
                        "user_key_press",
                        {"key": key_char, "action_label": action_label,
                         "phase": phase, "source": "manual_user"},
                    )

                # Tüm subscriber'lara (VideoRecorder vb.) dağıt
                self._broadcast(event)

            listener = _kb.Listener(on_press=on_press)
            listener.daemon = True
            listener.start()
            self._keyboard_listener = listener

        except ImportError:
            self.bot.log(
                "pynput yuklu degil — 'pip install pynput' ile kurun. "
                "Kullanici klavye etkilesimleri kaydedilmeyecek.",
                level="WARNING",
            )
        except Exception as exc:
            self.bot.log(
                f"UserInputMonitor: Klavye listener baslatma hatasi: {exc}",
                level="WARNING",
            )

    # ── Fare Hook ─────────────────────────────────────────────────

    def _start_mouse_listener(self):
        try:
            from pynput import mouse as _mouse

            def on_click(x, y, button, pressed):
                if not pressed or not self._active:
                    return

                is_bot = _BOT_CLICK_EVT.is_set()

                # Debounce
                now = time.monotonic()
                with self._lock:
                    if (now - self._last_click_ts) * 1000 < _DEBOUNCE_MS:
                        return
                    self._last_click_ts = now

                btn_name = "left"
                try:
                    btn_name = button.name
                except Exception:
                    pass

                source = "bot" if is_bot else "user"
                phase = getattr(self.bot, "_global_phase", "UNKNOWN_PHASE")

                event = InputEvent(
                    source=source,
                    event_type="mouse_click",
                    data={"x": int(x), "y": int(y), "button": btn_name},
                    ts_mono=now,
                    phase=phase,
                )

                # Kullanıcı tıklamalarını TrainingLogger'a bildir
                # ★ FIX: EXP_FARM boşta iken loglamayı sustur (log şişmesi önlenir)
                if source == "user" and self._should_log_user_input():
                    self.bot.log(f"[InputHub] Kullanici tik: ({x},{y})", level="DEBUG")
                    self.bot.log_training_action(
                        "user_mouse_click",
                        {"x": int(x), "y": int(y), "phase": phase,
                         "source": "manual_user"},
                    )

                # Tüm subscriber'lara dağıt
                self._broadcast(event)

            listener = _mouse.Listener(on_click=on_click)
            listener.daemon = True
            listener.start()
            self._mouse_listener = listener

        except ImportError:
            pass
        except Exception as exc:
            self.bot.log(
                f"UserInputMonitor: Fare listener baslatma hatasi: {exc}",
                level="WARNING",
            )

    # ── Yardımcılar ──────────────────────────────────────────────

    @staticmethod
    def _extract_key_char(key) -> Optional[str]:
        """pynput Key nesnesinden tek karakter çıkarır."""
        try:
            from pynput.keyboard import Key
            if isinstance(key, Key):
                name = key.name if hasattr(key, "name") else None
                if name and name.startswith("f") and name[1:].isdigit():
                    return name
                return None
            char = getattr(key, "char", None)
            if char and char.isprintable() and len(char) == 1:
                return char.lower()
        except Exception:
            pass
        return None
