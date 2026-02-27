"""
user_input_monitor.py â€” KullanÄ±cÄ± Klavye + Fare MÃ¼dahale Dinleyicisi
=====================================================================
Bot Ã§alÄ±ÅŸÄ±rken kullanÄ±cÄ±nÄ±n yaptÄ±ÄŸÄ± GERÃ‡EK fiziksel klavye/fare
eylemlerini yakalar, bot'un kendi pyautogui eylemlerinden ayÄ±rt eder
ve SequentialRecorder'a "manual_user" etiketiyle bildirir.

Neden gerekli?
  - SequentialRecorder'da sadece fare dinleyicisi var, klavye YOK.
  - KullanÄ±cÄ± A/Q/Z/V'ye bastÄ±ÄŸÄ±nda (dÃ¼zeltme / mÃ¼dahale) bu bilgi
    ÅŸu an hiÃ§ kaydedilmiyor.
  - Agentic eÄŸitim iÃ§in "bot ne yaptÄ±" vs "kullanÄ±cÄ± ne dÃ¼zeltti"
    ayrÄ±mÄ± kritik â€” bu olmadan model Ã¶dÃ¼l sinyali Ã¶ÄŸrenemez.

Entegrasyon:
    # loabot_main.py â†’ __init__ iÃ§inde:
    from user_input_monitor import UserInputMonitor
    self.user_monitor = UserInputMonitor(self)

    # SequentialRecorder._start_mouse_listener() artÄ±k buraya taÅŸÄ±ndÄ±.
    # seq_recorder.py'daki _start_mouse_listener() kaldÄ±rÄ±lmalÄ± veya
    # bu modÃ¼le delege edilmeli.

KayÄ±tlanan eylemler:
    Klavye: a, q, v, z  (ACTION_COMMAND_MAP ile Ã¶rtÃ¼ÅŸen tuÅŸlar)
            + herhangi ek tuÅŸ  (bot aksiyonu deÄŸilse "user_key_X")
    Fare  : sol tÄ±k (bot tÄ±klamasÄ± filtrele)

Metadata.json'a eklenen alanlar:
    "trigger_type" : "manual_user"
    "user_key"     : "a"               # sadece klavye olayÄ±nda
    "click_point"  : {"x": 1234, "y": 567}  # sadece fare olayÄ±nda
"""

from __future__ import annotations

import math
import random
import threading
import time
from typing import TYPE_CHECKING, Optional

try:
    import pyautogui
except Exception:
    pyautogui = None

if TYPE_CHECKING:
    pass  # DÃ¶ngÃ¼sel import engellemek iÃ§in

# Bot'un kendi pyautogui tuÅŸ basmalarÄ±nÄ± iÅŸaretleyen global event
# utils.py'daki BOT_IS_CLICKING_EVENT'e benzer ÅŸekilde
from utils import BOT_IS_CLICKING_EVENT as _BOT_CLICK_EVT

# Bot'un press_key() Ã§aÄŸrÄ±sÄ±nda set edilecek yeni event
# (automator.py'a eklenmeli â€” aÅŸaÄŸÄ±da aÃ§Ä±klanÄ±yor)
import threading as _threading
BOT_IS_PRESSING_KEY_EVENT = _threading.Event()

# ACTION_COMMAND_MAP'teki tuÅŸlar â€” sadece bunlar iÃ§in sekans mÃ¼hÃ¼rle
_TRACKED_KEYS = frozenset({"a", "q", "v", "z"})

# KullanÄ±cÄ± tÄ±klamasÄ± iÃ§in tekrar gÃ¶nderme kÄ±sÄ±tlamasÄ± (ms)
_DEBOUNCE_MS: float = 150.0


class UserInputMonitor:
    """
    KullanÄ±cÄ±nÄ±n fiziksel klavye ve fare girdilerini dinler.

    Ã–zellikler:
      - pynput ile sistem seviyesinde hook (pyautogui'dan baÄŸÄ±msÄ±z)
      - BOT_IS_CLICKING_EVENT: bot fare tÄ±klamasÄ± â†’ filtrele
      - BOT_IS_PRESSING_KEY_EVENT: bot tuÅŸ basmasÄ± â†’ filtrele
      - Ã‡ift tetikleme engeli (debounce)
      - KayÄ±t aktif deÄŸilse sessizce geÃ§
    """

    def __init__(self, bot):
        self.bot = bot
        self._last_key_ts: float = 0.0
        self._last_click_ts: float = 0.0
        self._lock = threading.Lock()
        self._keyboard_listener = None
        self._mouse_listener = None
        self._active = True

        self._start_keyboard_listener()
        self._start_mouse_listener()
        bot.log("UserInputMonitor: Klavye + fare dinleyicisi aktif.")

    # â”€â”€ DÄ±ÅŸa AÃ§Ä±k API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def shutdown(self):
        """Bot kapanÄ±rken Ã§aÄŸrÄ±lÄ±r."""
        self._active = False
        for listener in (self._keyboard_listener, self._mouse_listener):
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass

    # â”€â”€ Klavye Dinleyicisi â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def bezier_click(self, x: int, y: int, duration: float = 0.22) -> bool:
        """
        Fareyi insan benzeri cubic Bezier egriyle hedefe tasir ve tiklar.
        """
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

            def bezier_point(t: float) -> tuple[float, float]:
                omt = 1.0 - t
                bx = (omt ** 3) * sx + 3 * (omt ** 2) * t * c1x + 3 * omt * (t ** 2) * c2x + (t ** 3) * tx
                by = (omt ** 3) * sy + 3 * (omt ** 2) * t * c1y + 3 * omt * (t ** 2) * c2y + (t ** 3) * ty
                return bx, by

            total_duration = max(0.08, float(duration))
            for i in range(1, steps + 1):
                t = i / float(steps)
                bx, by = bezier_point(t)
                pyautogui.moveTo(int(round(bx)), int(round(by)), duration=0)
                time.sleep(max(0.001, (total_duration / steps) * random.uniform(0.85, 1.15)))

            pyautogui.click(tx, ty, duration=0)
            return True
        except Exception as exc:
            self.bot.log(f"UserInputMonitor: Bezier click hatasi: {exc}", level="WARNING")
            return False


    def _start_keyboard_listener(self):
        try:
            from pynput import keyboard as _kb

            def on_press(key):
                if not self._active:
                    return
                # Bot kendi tuÅŸ basmasÄ±nÄ± iÅŸaretlemiÅŸse filtrele
                if BOT_IS_PRESSING_KEY_EVENT.is_set():
                    return
                # KayÄ±t aktif deÄŸilse geÃ§
                if not self._is_recording():
                    return

                key_char = self._extract_key_char(key)
                if key_char is None:
                    return

                # Debounce
                now = time.monotonic()
                with self._lock:
                    if (now - self._last_key_ts) * 1000 < _DEBOUNCE_MS:
                        return
                    self._last_key_ts = now

                action_label = (
                    f"key_{key_char}" if key_char in _TRACKED_KEYS
                    else f"user_key_{key_char}"
                )

                self.bot.log(
                    f"[UserMonitor] Kullanici tusu: {key_char} â†’ {action_label}",
                    level="DEBUG",
                )

                # SequentialRecorder'a bildir
                seq = getattr(self.bot, "seq_recorder", None)
                if seq is not None and seq.is_recording:
                    seq.seal_action(
                        action_label=action_label,
                        extra={
                            "trigger_type": "manual_user",
                            "user_key": key_char,
                            "source": "keyboard",
                        },
                    )

                # TrainingLogger'a da bildir
                self.bot.log_training_action(
                    "user_key_press",
                    {
                        "key": key_char,
                        "action_label": action_label,
                        "phase": getattr(self.bot, "_global_phase", "UNKNOWN"),
                        "source": "manual_user",
                    },
                )

            listener = _kb.Listener(on_press=on_press)
            listener.daemon = True
            listener.start()
            self._keyboard_listener = listener
            self.bot.log("UserInputMonitor: Klavye hook aktif.")

        except ImportError:
            self.bot.log(
                "pynput yuklu degil â€” 'pip install pynput' ile kurun. "
                "Kullanici klavye etkilesimleri kaydedilmeyecek.",
                level="WARNING",
            )
        except Exception as exc:
            self.bot.log(f"UserInputMonitor: Klavye listener baslatma hatasi: {exc}", level="WARNING")

    # â”€â”€ Fare Dinleyicisi â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _start_mouse_listener(self):
        """
        SequentialRecorder'daki _start_mouse_listener yerine burada merkezi yÃ¶netim.
        SequentialRecorder'da _start_mouse_listener() metodunu devre dÄ±ÅŸÄ± bÄ±rakÄ±n
        ve bu sÄ±nÄ±fÄ± kullanÄ±n.
        """
        try:
            from pynput import mouse as _mouse

            def on_click(x, y, button, pressed):
                if not pressed or button != _mouse.Button.left:
                    return
                if not self._active:
                    return
                if _BOT_CLICK_EVT.is_set():
                    return  # Bot'un kendi tÄ±klamasÄ±
                if not self._is_recording():
                    return

                now = time.monotonic()
                with self._lock:
                    if (now - self._last_click_ts) * 1000 < _DEBOUNCE_MS:
                        return
                    self._last_click_ts = now

                click_point = {"x": int(x), "y": int(y)}
                self.bot.log(
                    f"[UserMonitor] Kullanici tik: ({x},{y})",
                    level="DEBUG",
                )

                seq = getattr(self.bot, "seq_recorder", None)
                if seq is not None and seq.is_recording:
                    seq.seal_action(
                        action_label="manual_mouse_click",
                        extra={
                            "trigger_type": "manual_user",
                            "click_point": click_point,
                            "source": "mouse",
                        },
                    )

                self.bot.log_training_action(
                    "user_mouse_click",
                    {
                        "x": int(x),
                        "y": int(y),
                        "phase": getattr(self.bot, "_global_phase", "UNKNOWN"),
                        "source": "manual_user",
                    },
                )

            listener = _mouse.Listener(on_click=on_click)
            listener.daemon = True
            listener.start()
            self._mouse_listener = listener
            self.bot.log("UserInputMonitor: Fare hook aktif.")

        except ImportError:
            pass  # pynput yoksa klavye listener da baÅŸarÄ±sÄ±z olmuÅŸtur, uyarÄ± verildi
        except Exception as exc:
            self.bot.log(f"UserInputMonitor: Fare listener baslatma hatasi: {exc}", level="WARNING")

    # â”€â”€ YardÄ±mcÄ±lar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _is_recording(self) -> bool:
        """SeqRecorder veya AutoLog aktif mi?"""
        seq = getattr(self.bot, "seq_recorder", None)
        if seq is not None and seq.is_recording:
            return True
        return bool(getattr(self.bot, "_auto_log_active", False))

    @staticmethod
    def _extract_key_char(key) -> Optional[str]:
        """pynput Key nesnesinden tek karakter Ã§Ä±karÄ±r."""
        try:
            from pynput.keyboard import Key
            # Ã–zel tuÅŸlar (Shift, Ctrl, ...) â€” gÃ¶rmezden gel
            if isinstance(key, Key):
                return None
            char = getattr(key, "char", None)
            if char and char.isprintable() and len(char) == 1:
                return char.lower()
        except Exception:
            pass
        return None

