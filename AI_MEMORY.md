# AI_MEMORY.md
## LoABot v5.9 - Ajan Hafiza Dosyasi

**Proje:** LoABot v5.9  
**Amac:** Ajanlar arasi geciste (Codex, Claude vb.) mimari surekliligi korumak ve "ajan amnezisi" riskini sifira indirmek.  
**Kapsam:** Bu dosya, projenin kalici mimari sozlesmesidir. Yeni bir ajan gorevi devraldiginda ilk referans bu dosyadir.

---

## 1) Proje Ozeti ve Vizyon

LoABot v5.9, yerel agda calisan, tam otonom MMORPG otomasyon botudur.  
Sistem hedefi:

- dusuk gecikme,
- internetten bagimsiz karar alma,
- uzun sureli stabil calisma,
- veriyi kaybetmeden kademeli ogrenme.

---

## 2) Temel Mimari (3-Tier System)

## Tier-1 (Sol Beyin) - Deterministik Motor
- Statik kurallar
- OpenCV tabanli template/anchor kontrolu
- UI sekans yurutumu (`click`, `boss_secimi`, `key`)
- Emniyet fallback mekanizmalari

## Tier-2 (Refleks Merkezi) - Temporal PyTorch
- Model: `TemporalAgenticNet`
- Girdi: 9 kanal (`[T-2, T-1, T]`)
- Backbone: ResNet18 / EfficientNet-B0
- Cikti: normalize koordinat regresyonu (`x, y`)

Not: Tier-2 artik sadece saf koordinat tahmini yapar.  
`pytorch_inference.py` icindeki eski `is_stagnant` tabanli erken cikis kaldirilmistir.

## Tier-3 (Sag Beyin / Stratejik Zihin)
- Yerel Ollama ustunde Qwen 3.5 9B Multimodal
- JSON tabanli stratejik kararlar
- Bulut API / Google GenAI KULLANILMAZ

---

## 3) Son Kritik Ozellikler ve Duzeltmeler

## 3.1 Navigation Pulse Check (1 saniyelik nabiz kontrolu)
`boss_manager.py` icine `_check_navigation_pulse()` eklendi.

Calisma:
1. Ekran griye cevrilir.
2. 1 saniye sonra ikinci gri kare alinir.
3. Merkez bolge maskelenmis fark analizi (absdiff) yapilir.
4. Hareket dusukse aksiyon tetiklenir.

## 3.2 Evasion Combo (Matrix kacisi)
Nabiz "hareket yok" dediginde ilk refleks:

`Space -> Q -> Space`

Amac: pusu/stagger durumunda karakteri yuruyuse geri sokmak.

## 3.3 Re-Anchor Fallback
Evasion sonrasi hala hareket yoksa (varsayilan 3sn):

`boss_list_ac -> boss_secimi -> z`

Bu kombinasyon rota kilitlenmesini kirar ve hedefe yeniden anchor atar.

## 3.4 Area Lock Guard (Yeni mantik duzeltmesi)
Mantik hatasi giderildi:  
Boss alani bulunduysa (`area_found` / `target["_area_check_ok"]`), karakterin durmasi normal kabul edilir.  
Bu durumda pulse/evasion **kesinlikle** tetiklenmez.

Uygulama noktasi:
- `_unified_spawn_sequence` icindeki ana while dongusu
- Pulse cagrisi su kosula alindi:
  - `if not area_found and not attack_done and now < spawn_ts:`

## 3.5 Same-Layer 90sn Hard Guard + Return Source Logs
`combat_manager.py/check_strategic_wait` guclendirildi:

- Ayni katmanda bir sonraki boss `ready` penceresinde 90sn icindeyse
  bot **asla** `EXP_FARM`'a donmez.
- Bu kural AI kararinin da ustundedir (`rule_same_layer_fast90`).
- `EXP_FARM` donuslerinde kaynagi netlemek icin debug log eklendi:
  - `[RETURN_SOURCE] ... source=ai`
  - `[RETURN_SOURCE] ... source=rule`
  - `[RETURN_SOURCE] ... source=no_upcoming_boss`

---

## 4) Egitim Durumu

`train_agentic.py` tarafinda transfer learning akisi aktif:

- `--weights` argumani eklendi.
- Checkpoint varsa `model_state_dict` yukleniyor.

Hedef:
- Catastrophic Forgetting riskini dusurmek
- Yeni videolarla "cila egitimi" yapmak
- Eski tecrubeyi korumak

---

## 5) Sabit Mimari Kurallar

1. Yerel-öncelikli calisma korunur.  
2. Tier hiyerarsisi bozulmaz (Tier-1 emniyet, Tier-2 refleks, Tier-3 strateji).  
3. Kritik kararlar loglanir ve geri izlenebilir olur.  
4. Yeni ajan degisiklikten once bu dosyayi okur.  
5. Bulut LLM'e geri donus yapilmaz.

## 6) Gelistirici Operasyon Notu

Commit/push otomasyonu eklendi:

- `git_autosync.ps1`
- `git_autosync.bat`

Kullanim:

- Otomatik mesaj + push:
  - `.\git_autosync.bat`
- Ozel mesaj + push:
  - `.\git_autosync.bat -Message "fix: strategic wait guard"`
- Sadece commit (push yok):
  - `.\git_autosync.bat -NoPush`

---

## Kisa Durum Cumlesi

LoABot v5.9 su anda yerel, cok katmanli, refleks + strateji hibrit mimaride calisiyor; aktif odak, navigasyon stabilitesini ve transfer-learning surekliligini uretim seviyesinde sabitlemektir.
