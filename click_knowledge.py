"""
click_knowledge.py — Tıklama Koordinat Bilgi Tabanı
====================================================
Alamet modeli "mouse_click" kararı verdiğinde nereye tıklanacağını bilir.

Öğrenme döngüsü:
  1. config/click_map.yaml'dan başlar  (YAML statik tanımlar)
  2. Her başarılı bot tıklaması ile büyür  (dinamik gözlemler)
  3. models/active_model/click_knowledge.json'a kaydeder
  4. train_agentic.py --with-click-head ile best.pt'ye öğretilir
     (screenshot + phase → norm_x, norm_y)

Kullanım:
    kb = ClickKnowledgeBase(screen_w=2560, screen_h=1440)
    coord = kb.get_coord("EVENT_PHASE")       # → {"x": 1276, "y": 108}
    kb.record_click("EVENT_PHASE", coord)     # gözlemi kaydet
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


_DEFAULT_MAP_PATH  = Path(__file__).parent / "config" / "click_map.yaml"
_DEFAULT_DB_PATH   = Path(__file__).parent / "models" / "active_model" / "click_knowledge.json"
_COORD_TOLERANCE   = 8    # piksel; bu kadar yakın tıklamalar aynı hedef sayılır


class ClickKnowledgeBase:
    """
    Faz + aksiyon → tıklama koordinatı sözlüğü.

    Öncelik sırası:
      1. Dinamik (gözlemlenmiş, count yüksek)
      2. Statik  (click_map.yaml'dan)
      3. None    (bilinmiyor, dışarı bildir)
    """

    def __init__(
        self,
        map_path:  Optional[Path] = None,
        db_path:   Optional[Path] = None,
        screen_w:  int = 2560,
        screen_h:  int = 1440,
        logger=None,
    ):
        self._map_path  = Path(map_path)  if map_path  else _DEFAULT_MAP_PATH
        self._db_path   = Path(db_path)   if db_path   else _DEFAULT_DB_PATH
        self._screen_w  = screen_w
        self._screen_h  = screen_h
        self._log       = logger          # bot.log veya None

        # {faz|aksiyon: {"coord": {x,y}, "label": str, "source": str}}
        self._static: Dict[str, Dict]       = {}
        # {faz|aksiyon: [{"coord":{x,y},"count":int,"last_seen":float,"source":str}, ...]}
        self._dynamic: Dict[str, List[Dict]] = {}

        self._load_static()
        self._load_dynamic()

    # ── Dışa Açık API ─────────────────────────────────────────────────────

    def get_coord(self, phase: str, action: str = "mouse_click") -> Optional[Dict]:
        """
        Faz + aksiyon için en iyi koordinatı döndürür.
        Önce dinamik (gözlemlenmiş en çok kullanılan), sonra statik.
        """
        key = self._key(phase, action)

        if key in self._dynamic and self._dynamic[key]:
            best = max(self._dynamic[key], key=lambda e: e["count"])
            return dict(best["coord"])

        if key in self._static:
            return dict(self._static[key]["coord"])

        return None

    def record_click(
        self,
        phase:   str,
        coord:   Dict,
        action:  str = "mouse_click",
        source:  str = "bot_observed",
    ) -> None:
        """
        Başarılı bir tıklamayı kaydet.
        Aynı bölgede (tolerans dahilinde) daha önce kayıt varsa sayacı artır.
        """
        key = self._key(phase, action)
        x, y = int(coord.get("x", 0)), int(coord.get("y", 0))

        entries = self._dynamic.setdefault(key, [])
        for entry in entries:
            ex, ey = entry["coord"]["x"], entry["coord"]["y"]
            if abs(ex - x) <= _COORD_TOLERANCE and abs(ey - y) <= _COORD_TOLERANCE:
                entry["count"]    += 1
                entry["last_seen"] = time.time()
                break
        else:
            entries.append({
                "coord":     {"x": x, "y": y},
                "count":     1,
                "last_seen": time.time(),
                "source":    source,
            })

        self._save_dynamic()
        self._info(f"Tiklama kaydedildi: {phase}|{action} → ({x},{y})")

    def export_training_samples(self) -> List[Dict]:
        """
        train_agentic.py --with-click-head için öğrenme örnekleri.

        Döndürür:
            [{"phase": str, "action": str, "x_norm": float, "y_norm": float,
              "count": int, "source": str}, ...]
        """
        samples = []
        # Dinamik (gözlemlenmiş)
        for key, entries in self._dynamic.items():
            phase, action = key.split("|", 1)
            for e in entries:
                samples.append({
                    "phase":  phase,
                    "action": action,
                    "x_norm": round(e["coord"]["x"] / self._screen_w, 5),
                    "y_norm": round(e["coord"]["y"] / self._screen_h, 5),
                    "count":  e["count"],
                    "source": e["source"],
                })
        # Statik (YAML) — düşük öncelik
        for key, entry in self._static.items():
            phase, action = key.split("|", 1)
            # Dinamikte yoksa ekle
            if key not in self._dynamic:
                samples.append({
                    "phase":  phase,
                    "action": action,
                    "x_norm": round(entry["coord"]["x"] / self._screen_w, 5),
                    "y_norm": round(entry["coord"]["y"] / self._screen_h, 5),
                    "count":  0,
                    "source": entry.get("source", "yaml"),
                })
        return samples

    def summary(self) -> Dict:
        return {
            "static_entries":  len(self._static),
            "dynamic_entries": sum(len(v) for v in self._dynamic.values()),
            "phases_known":    list({k.split("|")[0] for k in {**self._static, **self._dynamic}}),
        }

    # ── Dahili: Yükleme / Kaydetme ────────────────────────────────────────

    def _load_static(self) -> None:
        if not self._map_path.exists():
            self._info(f"click_map.yaml bulunamadi: {self._map_path}", "WARNING")
            return
        try:
            with open(self._map_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for key, entry in data.items():
                if "coord" in entry:
                    self._static[key] = entry
            self._info(f"Statik harita yuklendi: {len(self._static)} giris")
        except Exception as exc:
            self._info(f"click_map.yaml okunamadi: {exc}", "WARNING")

    def _load_dynamic(self) -> None:
        if not self._db_path.exists():
            return
        try:
            with open(self._db_path, encoding="utf-8") as f:
                self._dynamic = json.load(f)
            total = sum(len(v) for v in self._dynamic.values())
            self._info(f"Dinamik bilgi yuklendi: {total} gozlem")
        except Exception as exc:
            self._info(f"click_knowledge.json okunamadi: {exc}", "WARNING")

    def _save_dynamic(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._db_path, "w", encoding="utf-8") as f:
                json.dump(self._dynamic, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            self._info(f"click_knowledge.json kaydedilemedi: {exc}", "WARNING")

    # ── Dahili: Yardımcı ──────────────────────────────────────────────────

    @staticmethod
    def _key(phase: str, action: str) -> str:
        return f"{phase}|{action}"

    def _info(self, msg: str, level: str = "INFO") -> None:
        if self._log is not None:
            try:
                self._log(f"[ClickKB] {msg}", level=level)
            except Exception:
                pass
