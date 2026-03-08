import os
import time

import pyautogui

# Fallback ekran yakalama için (vision.capture_full_screen yoksa)
try:
    import mss as _mss_fallback
    import numpy as _np_fallback
    import cv2 as _cv2_fallback
    _HAS_FALLBACK_CAPTURE = True
except ImportError:
    _HAS_FALLBACK_CAPTURE = False

from utils import BOT_IS_CLICKING_EVENT
from user_input_monitor import BOT_IS_PRESSING_KEY_EVENT


class Automator:
    def __init__(self, bot):
        self.bot = bot

    def _capture_action_decision(self, action_name: str, payload: dict, stage: str) -> str:
        if not hasattr(self.bot, "capture_local_decision_frame"):
            return ""
        try:
            return self.bot.capture_local_decision_frame(
                action_name=action_name,
                payload=dict(payload or {}),
                phase="ACTION_PHASE",
                stage=stage,
            )
        except Exception:
            return ""

    @staticmethod
    def _normalize_action_name(action) -> str:
        name = str(action or "").strip().lower()
        aliases = {
            "clcik": "click",
            "katman_secimi": "katman_secimi",
            "katman_seçimi": "katman_secimi",
            "katman_seã§imi": "katman_secimi",
            "boss_secimi": "boss_secimi",
            "boss_seçimi": "boss_secimi",
            "boss_seã§imi": "boss_secimi",
        }
        return aliases.get(name, name)

    @staticmethod
    def _normalize_label_token(value: str) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        text = (
            text.replace("ç", "c")
            .replace("ğ", "g")
            .replace("ı", "i")
            .replace("ö", "o")
            .replace("ş", "s")
            .replace("ü", "u")
            .replace("ã§", "c")
        )
        return "".join(ch for ch in text if ch.isalnum())

    def _scale_coord_to_screen(self, coord: dict) -> dict:
        if not isinstance(coord, dict):
            return {}
        try:
            x = float(coord.get("x"))
            y = float(coord.get("y"))
        except Exception:
            return {}

        base_w = max(1.0, float(self.bot.settings.get("SCREEN_WIDTH", 2560)))
        base_h = max(1.0, float(self.bot.settings.get("SCREEN_HEIGHT", 1440)))
        try:
            screen_w, screen_h = pyautogui.size()
        except Exception:
            screen_w, screen_h = int(base_w), int(base_h)

        sx = int(round((x / base_w) * float(screen_w)))
        sy = int(round((y / base_h) * float(screen_h)))
        return {"x": sx, "y": sy}

    def _build_yolo_candidates(self, label: str, fallback_coord: dict = None, context_target: dict = None) -> set:
        candidates = set()

        def add_candidate(raw):
            if raw is None:
                return
            if isinstance(raw, dict):
                for key in ("image_file", "aciklama", "boss_id", "label", "name"):
                    add_candidate(raw.get(key))
                return

            text = str(raw).strip()
            if not text:
                return

            candidates.add(self._normalize_label_token(text))

            if text.lower().endswith(".png"):
                stem = os.path.splitext(os.path.basename(text))[0]
                candidates.add(self._normalize_label_token(stem))

        add_candidate(label)

        if isinstance(fallback_coord, dict):
            add_candidate(fallback_coord.get("label"))
            add_candidate((fallback_coord.get("anchor") or {}).get("image_file"))

        if isinstance(context_target, dict):
            for key in ("aciklama", "boss_id", "katman_id", "koordinat_ref"):
                add_candidate(context_target.get(key))
            for key in ("anchor", "gorsel_dogrulama", "area_check", "spawn_check"):
                add_candidate(context_target.get(key))

        return {c for c in candidates if c}

    def _pick_best_yolo_match(self, preds, candidates: set, offset_x: int = 0, offset_y: int = 0) -> tuple:
        best = None
        best_conf = 0.0

        for result in preds or []:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {}) or {}
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                class_name = str(names.get(cls_id, cls_id))
                normalized_cls = self._normalize_label_token(class_name)
                if not normalized_cls:
                    continue

                matched = any(
                    normalized_cls == cand or normalized_cls in cand or cand in normalized_cls
                    for cand in candidates
                )
                if not matched:
                    continue

                xyxy = box.xyxy[0].tolist() if box.xyxy is not None else None
                if not xyxy or len(xyxy) != 4:
                    continue

                x1, y1, x2, y2 = [float(v) for v in xyxy]
                cx = int(round((x1 + x2) * 0.5)) + int(offset_x)
                cy = int(round((y1 + y2) * 0.5)) + int(offset_y)
                b_conf = float(box.conf.item()) if box.conf is not None else 0.0
                if best is None or b_conf > best_conf:
                    best = {"x": cx, "y": cy}
                    best_conf = b_conf

        return best, float(best_conf)

    def _detect_coord_with_yolo(self, label: str, fallback_coord: dict = None, context_target: dict = None) -> tuple:
        try:
            # vision objesini güvenli bir şekilde al
            vision = getattr(self.bot, "vision", getattr(self.bot, "vision_manager", None))
            if vision is None or not hasattr(vision, "detect_with_yolo"):
                return None, 0.0

            candidates = self._build_yolo_candidates(label, fallback_coord=fallback_coord, context_target=context_target)
            if not candidates:
                return None, 0.0

            # GÜVENLİ EKRAN YAKALAMA (Fallback mekanizmalı)
            frame = None
            if hasattr(vision, "capture_full_screen"):
                frame = vision.capture_full_screen()
            else:
                # vision_manager içinde capture_full_screen yoksa, mss ile kendimiz yakalıyoruz (Çökmeyi engeller!)
                if not _HAS_FALLBACK_CAPTURE:
                    return None, 0.0
                with _mss_fallback.mss() as sct:
                    # Ana monitörü yakala
                    monitor = sct.monitors[1]
                    shot = sct.grab(monitor)
                    # YOLO BGR formatını sever, dönüştürüyoruz
                    frame = _cv2_fallback.cvtColor(_np_fallback.array(shot), _cv2_fallback.COLOR_BGRA2BGR)

            if frame is None:
                return None, 0.0

            conf = float(self.bot.settings.get("YOLO_CLICK_CONF", 0.25))
            roi_first = bool(self.bot.settings.get("YOLO_CLICK_ROI_FIRST", True))
            full_frame_fallback = bool(self.bot.settings.get("YOLO_CLICK_FULL_FRAME_FALLBACK", False))

            # Hız odaklı: once fallback koordinat etrafinda ROI ile dene.
            if roi_first and isinstance(fallback_coord, dict):
                scaled = self._scale_coord_to_screen(fallback_coord)
                cx = int(scaled.get("x", 0))
                cy = int(scaled.get("y", 0))
                if cx > 0 and cy > 0:
                    h, w = frame.shape[:2]
                    half_w = max(80, int(self.bot.settings.get("YOLO_CLICK_ROI_HALF_W", 360)))
                    half_h = max(80, int(self.bot.settings.get("YOLO_CLICK_ROI_HALF_H", 240)))
                    x1 = max(0, cx - half_w)
                    y1 = max(0, cy - half_h)
                    x2 = min(w, cx + half_w)
                    y2 = min(h, cy + half_h)

                    if (x2 - x1) > 1 and (y2 - y1) > 1:
                        roi = frame[y1:y2, x1:x2]
                        preds = vision.detect_with_yolo(roi, conf=conf)
                        best, best_conf = self._pick_best_yolo_match(preds, candidates, offset_x=x1, offset_y=y1)
                        if best is not None:
                            return best, float(best_conf)

            # Opsiyonel: ROI'de bulunamazsa tum ekrani tara (yavas olabilir).
            if full_frame_fallback:
                preds = vision.detect_with_yolo(frame, conf=conf)
                best, best_conf = self._pick_best_yolo_match(preds, candidates)
                if best is not None:
                    return best, float(best_conf)

            return None, 0.0

        except Exception as exc:
            self.bot.log(f"[FALLBACK DEVREDE] YOLO tespiti sirasinda kritik hata ({label}): {exc}", level="WARNING")
            return None, 0.0

    def _human_like_click(self, x: int, y: int, duration: float = 0.2) -> bool:
        if not bool(self.bot.settings.get("HUMAN_LIKE_CLICK_ENABLED", True)):
            pyautogui.click(int(x), int(y), duration=0)
            return True

        user_monitor = getattr(self.bot, "user_monitor", None)
        if user_monitor is not None:
            for method_name in ("bezier_click", "human_like_bezier_click", "move_and_click_bezier"):
                fn = getattr(user_monitor, method_name, None)
                if callable(fn):
                    try:
                        return bool(fn(int(x), int(y), float(duration)))
                    except TypeError:
                        try:
                            return bool(fn(x=int(x), y=int(y), duration=float(duration)))
                        except Exception:
                            pass
                    except Exception:
                        pass

        pyautogui.click(int(x), int(y), duration=float(duration))
        return True

    def click(self, label, coord=None, coord_override=None, duration=0.2, seal_label: str = "mouse_click"):
        """Klik islemi. coord_override eski kodlarla uyumluluk icindir.
        seal_label: SequentialRecorder'a gonderilecek aksiyon etiketi.
        """
        target = coord_override if coord_override is not None else coord
        decision_payload = {
            "label": str(label or ""),
            "x": int((target or {}).get("x", 0)) if isinstance(target, dict) else 0,
            "y": int((target or {}).get("y", 0)) if isinstance(target, dict) else 0,
            "duration": float(duration),
        }
        decision_id = self._capture_action_decision("click", decision_payload, stage="click_decision")

        if not target:
            self.bot.log_training_action(
                "click",
                {
                    "label": str(label or ""),
                    "success": False,
                    "reason": "missing_target",
                    "decision_id": decision_id,
                },
            )
            return False

        try:
            BOT_IS_CLICKING_EVENT.set()
            self._human_like_click(int(target["x"]), int(target["y"]), duration=float(duration))
            self.bot.log_training_action(
                "click",
                {
                    "label": str(label or ""),
                    "x": int(target.get("x", 0)),
                    "y": int(target.get("y", 0)),
                    "duration": float(duration),
                    "success": True,
                    "decision_id": decision_id,
                },
            )
            # ClickKnowledgeBase: basarili AI tiklamalarini ogren
            # (sadece AI kaynakli klikler; kural tabanli sekans klikleri haric)
            if seal_label == "mouse_click":
                self._record_click_to_kb(target)
            # Konum proxy'si: tiklanan koordinati son bilinen konum olarak guncelle
            if isinstance(target, dict):
                self._update_seq_position(target.get("x", 0), target.get("y", 0))
            # 10 FPS sekans muhurlme
            self._seal_seq_action(seal_label)
            # ── Video: bot tıklamasını logla ──
            vid = getattr(self.bot, "video_recorder", None)
            if vid is not None and vid.is_recording:
                vid.log_action(
                    event_type="bot_click",
                    data={"x": int(target.get("x", 0)), "y": int(target.get("y", 0))},
                    source="bot",
                    action_label=str(label or seal_label),
                )
            return True
        except Exception as exc:
            self.bot.log_training_action(
                "click",
                {
                    "label": str(label or ""),
                    "x": int(target.get("x", 0)),
                    "y": int(target.get("y", 0)),
                    "duration": float(duration),
                    "success": False,
                    "error": str(exc),
                    "decision_id": decision_id,
                },
            )
            return False
        finally:
            time.sleep(0.1)
            BOT_IS_CLICKING_EVENT.clear()

    def perform_agentic_click(
        self,
        label: str,
        fallback_coord: dict,
        context_target: dict = None,
        duration: float = 0.2,
        seal_label: str = "mouse_click",
    ) -> bool:
        perf_enabled = bool(self.bot.settings.get("SEQ_PERF_LOG_ENABLED", True))
        detect_t0 = time.perf_counter()
        yolo_coord, yolo_conf = self._detect_coord_with_yolo(
            label=label,
            fallback_coord=fallback_coord,
            context_target=context_target,
        )
        detect_ms = (time.perf_counter() - detect_t0) * 1000.0
        source = "yolo"
        target = yolo_coord

        if target is None:
            source = "fallback"
            target = self._scale_coord_to_screen(fallback_coord)

        if not target:
            self.bot.log(f"Sekans tiklama hatasi: hedef yok ({label}).", level="WARNING")
            self.bot.log_training_action(
                "agentic_click",
                {
                    "label": str(label or ""),
                    "success": False,
                    "reason": "missing_target",
                    "source": source,
                },
            )
            return False

        decision_payload = {
            "label": str(label or ""),
            "x": int(target.get("x", 0)),
            "y": int(target.get("y", 0)),
            "source": source,
            "yolo_conf": float(yolo_conf),
            "duration": float(duration),
        }
        decision_id = self._capture_action_decision("sequence_click", decision_payload, stage="sequence_click")

        seq_extra = {
            "trigger_type": "auto_rule_engine",
            "source": source,
            "label": str(label or ""),
            "x": int(target.get("x", 0)),
            "y": int(target.get("y", 0)),
            "yolo_conf": float(yolo_conf),
        }

        try:
            total_t0 = time.perf_counter()

            # BOUNDARY START: Fare hareketinden HEMEN ONCE - aksiyon niyeti (intention)
            self._boundary_start(seal_label, extra={**seq_extra, "click_point": {"x": target["x"], "y": target["y"]}})

            self._seal_seq_action(f"{seal_label}_T-1", extra={**seq_extra, "triplet": "T-1"})
            BOT_IS_CLICKING_EVENT.set()
            click_t0 = time.perf_counter()
            self._human_like_click(int(target["x"]), int(target["y"]), duration=float(duration))
            click_ms = (time.perf_counter() - click_t0) * 1000.0
            self._seal_seq_action(f"{seal_label}_T", extra={**seq_extra, "triplet": "T"})

            # 10 FPS kayit dongusunde bir sonraki kareyi yakalamasi icin kisa bekleme.
            time.sleep(0.11)
            self._seal_seq_action(f"{seal_label}_T+1", extra={**seq_extra, "triplet": "T+1"})

            # BOUNDARY END: T+1 mührü sonrası - aksiyon tamamlandı, karakter stabil
            self._boundary_end(seal_label, result_status="success", extra=seq_extra)

            self._update_seq_position(target.get("x", 0), target.get("y", 0))
            self.bot.log_training_action(
                "agentic_click",
                {
                    "label": str(label or ""),
                    "x": int(target.get("x", 0)),
                    "y": int(target.get("y", 0)),
                    "source": source,
                    "yolo_conf": float(yolo_conf),
                    "success": True,
                    "decision_id": decision_id,
                },
            )
            # ── Video: agentic tıklamayı logla ──
            vid = getattr(self.bot, "video_recorder", None)
            if vid is not None and vid.is_recording:
                vid.log_action(
                    event_type="bot_click",
                    data={
                        "x": int(target.get("x", 0)),
                        "y": int(target.get("y", 0)),
                        "source": source,
                        "yolo_conf": float(yolo_conf),
                    },
                    source="bot",
                    action_label=str(label or seal_label),
                )
            if perf_enabled:
                total_ms = (time.perf_counter() - total_t0) * 1000.0
                self.bot.log(
                    f"[SEQ_PERF] label={label} src={source} detect_ms={detect_ms:.1f} "
                    f"click_ms={click_ms:.1f} total_ms={total_ms:.1f}",
                    level="DEBUG",
                )
            return True
        except Exception as exc:
            # BOUNDARY END (fail): Exception durumunda da termination kaydedilir
            self._boundary_end(seal_label, result_status="fail", extra={"error": str(exc)})
            self.bot.log_training_action(
                "agentic_click",
                {
                    "label": str(label or ""),
                    "x": int(target.get("x", 0)),
                    "y": int(target.get("y", 0)),
                    "source": source,
                    "yolo_conf": float(yolo_conf),
                    "success": False,
                    "error": str(exc),
                    "decision_id": decision_id,
                },
            )
            return False
        finally:
            BOT_IS_CLICKING_EVENT.clear()

    def execute_sequence(self, sequence: list, coord_map: dict, context_target: dict = None) -> bool:
        if not sequence:
            self.bot.log("UYARI: Bos sekans calistirilmaya calisildi.")
            return False
        
        if not isinstance(sequence, list):
            self.bot.log(f"KRİTİK HATA: Sekans listesi bozuk veya dict olarak geldi! Tip: {type(sequence)}")
            return False

        self.bot.log(f"-> Sekans islemi baslatiliyor... ({len(sequence)} Adim)")

        for idx, step in enumerate(sequence):
            try:
                if not self.bot.running.is_set():
                    self.bot.log("Sekans iptal: Bot durduruldu.")
                    return False

                if not isinstance(step, dict):
                    self.bot.log(f"Sekans hatasi: Adim dict formatinda degil ({idx}).")
                    return False

                action = self._normalize_action_name(step.get("action"))
                label = step.get("label")
                wait_ms = int(step.get("wait_ms", 100)) if str(step.get("wait_ms", "")).strip() else 100
                seq_click_duration = float(self.bot.settings.get("SEQ_HUMAN_CLICK_DURATION", 0.12))

                self.bot.log(f"[*] Adim {idx+1}/{len(sequence)} isleniyor: {action} -> {label}")

                success = False

                if action == "click":
                    coord = coord_map.get(label) if isinstance(coord_map, dict) else None
                    if not isinstance(coord, dict):
                        self.bot.log(f"Sekans hatasi: '{label}' koordinati YAML'da bulunamadi.")
                        return False
                    success = self.perform_agentic_click(
                        label=str(label or ""),
                        fallback_coord=coord,
                        context_target=context_target,
                        duration=seq_click_duration,
                        seal_label=f"seq_{label}" if label else "seq_click",
                    )

                elif action == "press_key":
                    key = step.get("key")
                    if not key:
                        self.bot.log("Sekans hatasi: press_key adiminda 'key' eksik.")
                        return False
                    success = self.press_key(key, label=str(label or key))

                elif action == "katman_secimi":
                    if not context_target:
                        self.bot.log("Sekans hatasi: katman_secimi icin context_target eksik.")
                        return False
                    katman_label = context_target.get("katman_id")
                    coord = coord_map.get(katman_label) if isinstance(coord_map, dict) else None
                    if not katman_label or not isinstance(coord, dict):
                        self.bot.log(f"Sekans hatasi: katman koordinati bulunamadi ({katman_label}).")
                        return False
                    success = self.perform_agentic_click(
                        label=str(katman_label),
                        fallback_coord=coord,
                        context_target=context_target,
                        duration=seq_click_duration,
                        seal_label=f"seq_katman_{katman_label}" if katman_label else "seq_katman_secimi",
                    )

                elif action == "boss_secimi":
                    if not context_target:
                        self.bot.log("Sekans hatasi: boss_secimi icin context_target eksik.")
                        return False

                    coord = None
                    click_label = context_target.get("aciklama", "boss_coord")
                    boss_coord_label = context_target.get("koordinat_ref")
                    
                    if isinstance(boss_coord_label, str):
                        coord = coord_map.get(boss_coord_label) if isinstance(coord_map, dict) else None
                        if not isinstance(coord, dict):
                            coord = context_target.get("koordinat")
                        click_label = boss_coord_label
                    elif isinstance(boss_coord_label, dict):
                        coord = boss_coord_label
                    else:
                        coord = context_target.get("koordinat")

                    if not isinstance(coord, dict):
                        self.bot.log(f"Sekans hatasi: boss koordinati bulunamadi ({boss_coord_label}).")
                        return False

                    success = self.perform_agentic_click(
                        label=str(click_label),
                        fallback_coord=coord,
                        context_target=context_target,
                        duration=seq_click_duration,
                        seal_label=f"seq_boss_{click_label}" if click_label else "seq_boss_secimi",
                    )

                else:
                    self.bot.log(f"Sekans hatasi: Bilinmeyen YAML aksiyonu '{action}'.")
                    return False

                if not success:
                    self.bot.log(f"[X] Sekans adimi basarisiz oldu: '{action}' -> '{label or ''}'.")
                    return False

                if wait_ms > 0 and not self.bot._interruptible_wait(wait_ms / 1000.0):
                    return False
                
            except Exception as e:
                self.bot.log(f"KRİTİK HATA: Sekans dongusunde cokus! Adim: {idx} - Hata: {e}")
                return False

        self.bot.log("-> Sekans basariyla tamamlandi!")
        return True

    def _run_sequence(self, sequence: list, coord_map: dict, context_target: dict = None) -> bool:
        return self.execute_sequence(sequence=sequence, coord_map=coord_map, context_target=context_target)

    def press_key(self, key, label=None):
        decision_payload = {"key": str(key), "label": str(label or "")}
        decision_id = self._capture_action_decision("press_key", decision_payload, stage="key_decision")
        BOT_IS_PRESSING_KEY_EVENT.set()
        try:
            pyautogui.press(key)
        finally:
            BOT_IS_PRESSING_KEY_EVENT.clear()
        self.bot.log_training_action(
            "press_key",
            {"key": str(key), "label": str(label or ""), "success": True, "decision_id": decision_id},
        )
        # 10 FPS sekans muhurlme
        self._seal_seq_action(f"key_{key}")
        # ── Video: bot tuş basmasını logla ──
        vid = getattr(self.bot, "video_recorder", None)
        if vid is not None and vid.is_recording:
            vid.log_action(
                event_type="bot_key",
                data={"key": str(key)},
                source="bot",
                action_label=str(label or f"bot_key_{key}"),
            )
        return True

    def _record_click_to_kb(self, coord: dict) -> None:
        """Basarili AI tiklamasini ClickKnowledgeBase'e bildir (ogrenme)."""
        try:
            tb = getattr(self.bot, "tactical_brain", None)
            if tb is None:
                return
            kb = getattr(tb, "_click_kb", None)
            if kb is None:
                return
            phase = getattr(self.bot, "_global_phase", "UNKNOWN")
            kb.record_click(phase=phase, coord=coord, action="mouse_click", source="bot_observed")
        except Exception:
            pass

    def _seal_seq_action(self, action_label: str, extra: dict = None):
        """No-op: SequentialRecorder kaldırıldı. Çağrı noktaları geriye uyumluluk için korundu."""
        pass

    def _boundary_start(self, action_label: str, extra: dict = None) -> None:
        """No-op: SequentialRecorder kaldırıldı."""
        pass

    def _boundary_end(
        self,
        action_label: str,
        result_status: str = "unknown",
        extra: dict = None,
    ) -> None:
        """No-op: SequentialRecorder kaldırıldı."""
        pass

    def _update_seq_position(self, x: float, y: float):
        """No-op: SequentialRecorder kaldırıldı."""
        pass

    def login_to_game(self):
        """Restart sonrasi tam giris sekansi:
        1. server_s575.png bul ve tikla
        2. Harita yuklenme suresi kadar bekle (GAME_UI_READY_DELAY_SN)
        3. Etkinlik okunu kapat (etkinlik_oku_close)
        4. Z tusu ile otomatik savas modunu ac
        """
        self.bot.log("Automator: S575 sunucusu araniyor...")
        region_full = self.bot.ui_regions.get("region_full_screen", {"x": 0, "y": 0, "w": 2560, "h": 1440})

        for attempt in range(5):
            match = self.bot.vision.find("server_s575.png", region_full, 0.8)
            if match:
                self.click("server_selection", {"x": match[0], "y": match[1]})

                # Harita yuklenmesi icin bekle
                map_wait = float(self.bot.settings.get("GAME_UI_READY_DELAY_SN", 20.0))
                self.bot.log(f"Sunucuya baglandi, harita yukleniyor ({int(map_wait)} sn)...")
                time.sleep(map_wait)

                # Etkinlik okunu kapat - anchor image ile once varlik kontrol et
                etkinlik_coord = self.bot.coordinates.get("etkinlik_oku_close")
                if etkinlik_coord:
                    region_menu = self.bot.ui_regions.get(
                        "region_ust_menu",
                        self.bot.ui_regions.get("region_full_screen", {"x": 0, "y": 0, "w": 2560, "h": 1440}),
                    )
                    anchor = etkinlik_coord.get("anchor", {})
                    anchor_img = anchor.get("image_file", "etkinlik_oku_close.png")
                    anchor_conf = float(anchor.get("confidence", 0.8))
                    # Gorunur olana kadar max 5sn bekle
                    arrow_visible = False
                    for _ in range(5):
                        if self.bot.vision.find(anchor_img, region_menu, anchor_conf):
                            arrow_visible = True
                            break
                        time.sleep(1.0)
                    if arrow_visible:
                        self.bot.log("Etkinlik oku kapatiliyor...")
                        self.click("etkinlik_oku_close", {"x": etkinlik_coord["x"], "y": etkinlik_coord["y"]})
                        time.sleep(1.0)
                    else:
                        self.bot.log("Etkinlik oku gorulmedi, atlaniyor.")

                # Otomatik savas modunu ac
                self.press_key("z", label="login_auto_mode")
                self.bot.log("Login sekans tamamlandi.")
                return True

            self.bot.log(f"Sunucu bekleniyor... ({attempt + 1}/5)")
            time.sleep(3)

        self.bot.log("HATA: server_s575.png bulunamadi.")
        return False

    def return_to_exp_farm(self, force_restart_if_failed=True):
        """Kullanici talebi: ek exp_farm_sequence olmadan EXP_FARM kabul et."""
        if self.bot.location_manager.get_region_name() == "EXP_FARM":
            return True

        self.bot.log("Automator: EXP_FARM'a donus baslatildi.")

        if not self.safe_exit_map():
            if force_restart_if_failed:
                self.bot.log("Automator: Haritadan cikis basarisiz, restart denenecek.")
                return self.bot.restart_game()
            return False

        self.bot.location_manager.set_current_location_by_name("EXP_FARM")
        self.bot.stop_global_mission(reason="exp_farm_return")
        return True

    def safe_exit_map(self):
        exit_reg = self.bot.ui_regions.get("exit_map_area", {"x": 2000, "y": 0, "w": 560, "h": 300})
        match = self.bot.vision.find("exit_map.png", exit_reg, 0.75)

        if match:
            self.click("exit_map_btn", {"x": match[0], "y": match[1]})
            self.bot._interruptible_wait(3)
            return True

        boss_list_btn = self.bot.coordinates.get("boss_list_ac")
        if boss_list_btn:
            self.click("boss_list_force", boss_list_btn)
            self.bot._interruptible_wait(1)
            self.press_key("v", label="force_close_menu")
            return True

        return False
