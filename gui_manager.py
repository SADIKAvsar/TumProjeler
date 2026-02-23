# modules/gui_manager.py
import tkinter as tk
from tkinter import messagebox, simpledialog
from datetime import datetime
import math
import time
import traceback
from queue import Queue

class GUIManager:
    def __init__(self, bot, update_interval_ms: int = 1000):
        self.bot = bot
        self.root = getattr(bot, "root", None)
        self.gui_update_interval_ms = int(update_interval_ms)
        self.gui_queue = getattr(bot, "gui_queue", Queue())
        self.boss_targets_dict = getattr(bot, "bosslar", {})
        
        self.gui_widgets = {}
        self.title_label_line1 = None
        self.title_label_line2 = None
        self.log_text_widget = None
        self._created = False
        self.status_message_active = False
        self.last_status_message_time = 0

    def _log(self, text: str):
        self.bot.log(f"[GUI] {text}")

    def _parse_time_to_seconds(self, s: str) -> int | None:
        s = s.strip()
        try:
            parts = list(map(int, s.split(':')))
            if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
            elif len(parts) == 2: return parts[0]*60 + parts[1]
            elif len(parts) == 1: return parts[0]
            return None
        except: return None

    def _create_round_button(self, parent, symbol, color, command, col):
        canvas = tk.Canvas(parent, width=65, height=65, bg="black", highlightthickness=0, cursor="hand2")
        canvas.grid(row=0, column=col, padx=12, pady=5)
        
        # v5.8 Renk Paleti ve Şekiller
        circle = canvas.create_oval(5, 5, 60, 60, fill=color, outline=color, activefill="#333333")
        text_item = canvas.create_text(32, 32, text=symbol, fill="white", font=("Consolas", 20, "bold"))

        # KRİTİK: Tıklamayı her iki objeye de bağla (Binding)
        canvas.tag_bind(circle, "<Button-1>", lambda e: command())
        canvas.tag_bind(text_item, "<Button-1>", lambda e: command())
    def _edit_timer_label(self, event, boss_aciklama: str):
        """Zamanlayıcıya tıklandığında manuel süre giriş penceresini açar."""
        input_time = simpledialog.askstring("Zamanı Düzenle", 
                                          f"'{boss_aciklama}' için kalan süreyi girin:\n(HH:MM:SS, MM:SS veya Saniye)", 
                                          parent=self.root)
        if input_time is not None:
            kalan_saniye = self._parse_time_to_seconds(input_time)
            if kalan_saniye is not None:
                self.bot._set_spawn_time(boss_aciklama, kalan_saniye, source="manual")

    def _process_gui_queue(self):
        """Thread'ler arası mesaj iletişimini sağlar."""
        while not self.gui_queue.empty():
            try:
                task, data = self.gui_queue.get_nowait()
                if task == "log" and self.log_text_widget:
                    self.log_text_widget.config(state=tk.NORMAL)
                    self.log_text_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {data}\n")
                    self.log_text_widget.see(tk.END)
                    self.log_text_widget.config(state=tk.DISABLED)
                elif task == "set_spawn":
                    aciklama, spawn_time, source = data
                    self._update_spawn_time_in_gui(aciklama, spawn_time, source)
                # YENİ EKLENEN KISIM: Watchdog'dan gelen UI durum güncellemeleri
                elif task == "status_line1":
                    if self.title_label_line1:
                        self.title_label_line1.config(text=data[0], fg=data[1])
                elif task == "status_line2":
                    if self.title_label_line2:
                        self.title_label_line2.config(text=data[0], fg=data[1])
            except: break

    def _update_spawn_time_in_gui(self, aciklama, spawn_time, source):
        widgets = self.gui_widgets.get(aciklama)
        if widgets and "entry" in widgets:
            spawn_str = datetime.fromtimestamp(spawn_time).strftime("%H:%M:%S") if spawn_time > 0 else "--:--:--"
            widgets["entry"].config(state=tk.NORMAL)
            widgets["entry"].delete(0, tk.END)
            widgets["entry"].insert(0, spawn_str)
            # Kaynağa göre renk (v5.8 standardı)
            color = "cyan" if source == "automation" else "yellow"
            widgets["entry"].config(fg=color, state="readonly")

    def _update_gui(self):
        try:
            self._process_gui_queue()
            remaining = []
            for aciklama, hedef in self.bot.bosslar.items():
                widgets = self.gui_widgets.get(aciklama)
                if not widgets: continue
                spawn_t = hedef.get("spawn_time")
                if spawn_t and isinstance(spawn_t, (int, float)):
                    kalan = max(0, spawn_t - time.time())
                    if kalan > 0:
                        remaining.append((kalan, aciklama))
                    kalan_int = max(0, int(kalan))
                    saat, rem = divmod(kalan_int, 3600)
                    m, s = divmod(rem, 60)
                    try:
                        widgets["timer_label"].config(text=f"{saat:02}:{m:02}:{s:02}")
                    except Exception:
                        pass
                else:
                    try:
                        widgets["timer_label"].config(text="--:--:--")
                    except Exception:
                        pass

            remaining.sort(key=lambda x: x[0])
            self._update_status_labels(remaining)
            self._update_ai_panel()
            self._update_rec_btn()

        except Exception as e:
            traceback.print_exc()

        finally:
            try:
                if self.root and self.root.winfo_exists():
                    self.root.after(self.gui_update_interval_ms, self._update_gui)
            except Exception:
                pass

    def _update_status_labels(self, remaining):
        if not self.title_label_line1 or not self.title_label_line2:
            return

        try:
            is_paused = self.bot.paused
            is_running = self.bot.running.is_set()
            attacking = self.bot.attacking_target_aciklama
            region = self.bot.location_manager.get_region_name()

            # --- SATIR 1: Her zaman KONUM göster ---
            if is_paused:
                self.title_label_line1.config(text="Duraklatıldı", fg="goldenrod")
            elif not is_running:
                self.title_label_line1.config(text="Beklemede", fg="gray")
            else:
                color1 = "lime" if region == "EXP_FARM" else "cyan"
                self.title_label_line1.config(text=f"KONUM: {region}", fg=color1)

            # --- SATIR 2: Boss durumu (saldırıda kırmızı) ---
            if is_paused or not is_running:
                self.title_label_line2.config(text="Boss takibi duraklatıldı", fg="goldenrod")
            elif attacking:
                self.title_label_line2.config(text=f"{attacking} saldırılıyor...", fg="red")
            elif remaining:
                kalan, aciklama = remaining[0]
                kalan_int = max(0, int(kalan))
                saat, rem = divmod(kalan_int, 3600)
                m, s = divmod(rem, 60)
                time_str = f"{saat:02}:{m:02}:{s:02}"
                self.title_label_line2.config(text=f"SIRADAKİ: {aciklama} ({time_str})", fg="cyan")
            else:
                self.title_label_line2.config(text="BOSS HEDEFİ YOK", fg="gray")

        except Exception:
            pass

    def _toggle_recording(self):
        """Kayit alma islemi acip kapatir."""
        try:
            if hasattr(self.bot, "toggle_recording"):
                self.bot.toggle_recording()
            self._update_rec_btn()
        except Exception:
            pass

    def _update_rec_btn(self):
        """REC butonunun gorunumunu kayit moduna gore gunceller.

        Manuel kayit  → kirmizi  "KAYIT DURDUR"
        Oto kayit     → yesil    "AUTO [BOSS|ETKINLIK]"
        Pasif         → gri      "KAYIT BASLAT"
        """
        if not hasattr(self, "rec_btn"):
            return
        try:
            manual = getattr(self.bot, "_manual_recording", False)
            auto_active = getattr(self.bot, "_auto_log_active", False)

            if manual:
                self.rec_btn.config(
                    text="⏹  KAYIT DURDUR",
                    bg="#3a0000",
                    fg="#FF4444",
                )
            elif auto_active:
                trigger = getattr(self.bot, "_auto_log_trigger", "")
                _labels = {"boss_attack": "BOSS", "event_entry": "ETKINLIK"}
                short = _labels.get(trigger, "AUTO")
                self.rec_btn.config(
                    text=f"⏺  AUTO [{short}]",
                    bg="#002a00",
                    fg="#44FF88",
                )
            else:
                self.rec_btn.config(
                    text="⏺  KAYIT: KAPALI",
                    bg="#1a1a1a",
                    fg="#555555",
                )
        except Exception:
            pass

    def _toggle_ai(self):
        """AI'yi acip kapatir."""
        try:
            ai_engine = getattr(self.bot, 'brain', None)
            ai_engine = getattr(ai_engine, 'ai_engine', None) if ai_engine else None
            if ai_engine:
                ai_engine.toggle_ai()
        except Exception:
            pass

    def _update_ai_panel(self):
        """AI istatistik panelini guncelle."""
        try:
            ai_engine = getattr(self.bot, 'brain', None)
            ai_engine = getattr(ai_engine, 'ai_engine', None) if ai_engine else None

            if not ai_engine or not hasattr(self, 'ai_status_label'):
                return

            if ai_engine.enabled:
                stats = ai_engine.get_statistics()
                mode = "YEREL-GOZLEM" if ai_engine.observer_mode else "YEREL-OTONOM"
                self.ai_status_label.config(
                    text=f"AI: {mode} | Eval: {stats.get('ai_calls_made', 0)} | FB: {stats.get('fallback_count', 0)}",
                    fg="#00FF88"
                )
                self.ai_stats_label.config(
                    text=f"Karar: {stats.get('decisions_made', 0)} | "
                         f"Cache: {stats.get('cache_hit_rate', 0)}% | "
                         f"Maliyet: Yerel",
                    fg="#AAAAAA"
                )
                # Performance line
                history = stats.get('history', {})
                ai_ratio = history.get('local_ratio', 0)
                total_kills = stats.get('total_bosses_killed', 0)
                self.ai_perf_label.config(
                    text=f"Boss: {total_kills} kill | Local: {ai_ratio}% | Log: {history.get('total_logged', 0)}",
                    fg="#888888"
                )
                # Toggle button
                self.ai_toggle_btn.config(text="ON", fg="#00FF88", bg="#1a1a1a")
            else:
                self.ai_status_label.config(text="AI: Devre Disi", fg="red")
                self.ai_stats_label.config(text="Toggle ile aktif edin", fg="#666666")
                self.ai_perf_label.config(text="Kural tabanli mod aktif", fg="#555555")
                self.ai_toggle_btn.config(text="OFF", fg="red", bg="#1a1a1a")
        except Exception:
            pass

    def create_gui(self):
        self.root.title("LoA Bot v5.9 (Local-First)")
        self.root.configure(bg="black")
        self.root.wm_attributes("-topmost", True)
        # Pencere yüksekliği: boss sayısına göre dinamik
        n_boss = len(getattr(self.bot, "bosslar", {}))
        win_h = 560 + n_boss * 24   # her boss satırı ~24px
        self.root.geometry(f"320x{win_h}")

        # Durum Panel (2 Satır)
        self.title_label_line1 = tk.Label(self.root, text="Başlatılıyor...", bg="black", fg="cyan", font=("Consolas", 11, "bold"))
        self.title_label_line1.pack(pady=(10,0))
        self.title_label_line2 = tk.Label(self.root, text="Bekleniyor...", bg="black", fg="gray", font=("Consolas", 11, "bold"))
        self.title_label_line2.pack(pady=(0,5))

        # Boss Grid Alanı
        boss_frame = tk.LabelFrame(self.root, text=" Boss Zamanlayıcıları ", bg="black", fg="white", font=("Consolas", 10), bd=2)
        boss_frame.pack(fill=tk.X, padx=10, pady=5)

        boss_frame.columnconfigure(0, weight=1)
        boss_frame.columnconfigure(1, weight=1)
        boss_frame.columnconfigure(2, weight=1)

        for i, (aciklama, hedef) in enumerate(self.bot.bosslar.items()):
            tk.Label(boss_frame, text=f"{aciklama}:", bg="black", fg="white", font=("Consolas", 10), anchor="e").grid(row=i, column=0, sticky="e", padx=(15,8))

            ent = tk.Entry(boss_frame, width=9, font=("Consolas", 10, "bold"), readonlybackground="#202020", relief=tk.FLAT, justify='center')
            ent.grid(row=i, column=1, padx=8, pady=2)
            ent.config(state="readonly", fg="cyan")

            tmr = tk.Label(boss_frame, text="--:--:--", bg="black", fg="white", font=("Consolas", 10), cursor="hand2", anchor="w")
            tmr.grid(row=i, column=2, padx=(8,15))
            # Zamanlayıcıya tıklandığında düzenleme açılır
            tmr.bind("<Button-1>", lambda e, a=aciklama: self._edit_timer_label(e, a))

            self.gui_widgets[aciklama] = {"entry": ent, "timer_label": tmr}

        # v5.8'in Meşhur Yuvarlak Butonları
        btn_frame = tk.Frame(self.root, bg="black")
        btn_frame.pack(pady=10)
        self._create_round_button(btn_frame, "▶", "darkgreen", self.bot.start, 0)
        self._create_round_button(btn_frame, "⏸", "goldenrod", self.bot.stop, 1)
        self._create_round_button(btn_frame, "✖", "darkred", self.root.quit, 2)

        # Kayıt Kontrol Butonu
        rec_frame = tk.Frame(self.root, bg="black")
        rec_frame.pack(pady=(0, 6))
        self.rec_btn = tk.Button(
            rec_frame,
            text="⏺  KAYIT BAŞLAT",
            bg="#1a1a1a",
            fg="#555555",
            font=("Consolas", 9, "bold"),
            width=22,
            bd=1,
            relief=tk.RIDGE,
            cursor="hand2",
            command=self._toggle_recording,
        )
        self.rec_btn.pack()
        self._update_rec_btn()

        # AI İstatistik Paneli
        self.ai_frame = tk.LabelFrame(self.root, text=" AI Agent ", bg="black", fg="#00BFFF", font=("Consolas", 10, "bold"), bd=2)
        self.ai_frame.pack(fill=tk.X, padx=10, pady=5)

        # AI Status + Toggle Button (aynı satır)
        ai_top_frame = tk.Frame(self.ai_frame, bg="black")
        ai_top_frame.pack(fill=tk.X, padx=5, pady=1)

        self.ai_status_label = tk.Label(ai_top_frame, text="AI: Baslatiliyor...", bg="black", fg="gray", font=("Consolas", 9), anchor="w")
        self.ai_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.ai_toggle_btn = tk.Button(ai_top_frame, text="ON", bg="#1a1a1a", fg="#00FF88",
                                        font=("Consolas", 8, "bold"), width=4, bd=1, relief=tk.RIDGE,
                                        cursor="hand2", command=self._toggle_ai)
        self.ai_toggle_btn.pack(side=tk.RIGHT, padx=(5, 0))

        self.ai_stats_label = tk.Label(self.ai_frame, text="Karar: 0 | Cache: 0% | $0.00", bg="black", fg="#888888", font=("Consolas", 8), anchor="w")
        self.ai_stats_label.pack(fill=tk.X, padx=5, pady=0)

        self.ai_perf_label = tk.Label(self.ai_frame, text="Boss: 0 av | AI: 0%", bg="black", fg="#666666", font=("Consolas", 8), anchor="w")
        self.ai_perf_label.pack(fill=tk.X, padx=5, pady=(0, 3))

        # Terminal Log Alanı — sabit yükseklik, expand=False (boss timers'ı itmez)
        self.log_text_widget = tk.Text(self.root, height=6, bg="#101010", fg="lime", font=("Consolas", 9), state=tk.DISABLED, relief=tk.SOLID, bd=1)
        self.log_text_widget.pack(fill=tk.X, expand=False, padx=10, pady=(5, 10))

        self._update_gui()
        self._created = True
