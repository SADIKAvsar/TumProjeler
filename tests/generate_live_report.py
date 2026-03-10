import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.bot as mb
import modules.utils as u
import cv2, numpy as np, mss
import types

OUT_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(OUT_DIR, exist_ok=True)

cfg = u.load_config(u.CONFIG_FILE)
if not cfg:
    print('Config yüklenemedi:', u.CONFIG_FILE); raise SystemExit(1)

general = cfg.get('module_general', {})
ui_regions = general.get('ui_regions', {})
locations = general.get('locations', {})

# Dummy bot with bound find_image_on_screen
class Dummy: pass
bot = Dummy(); bot.ui_regions = ui_regions
bot.log = lambda msg: print('[BOT LOG]', msg)
bot.find_image_on_screen = types.MethodType(mb.LoABot.find_image_on_screen, bot)

# Capture full screen once
full_screenshot_path = os.path.join(OUT_DIR, 'screenshot_full.png')
with mss.mss() as sct:
    monitor = sct.monitors[0]
    img = np.array(sct.grab(monitor))
    cv2.imwrite(full_screenshot_path, cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
print('Full screen saved to', full_screenshot_path)

# Prepare annotated image (BGR)
annot = cv2.cvtColor(img.copy(), cv2.COLOR_BGRA2BGR)

# Optional global confidence override (set to None to use per-anchor confidence)
GLOBAL_CONF_OVERRIDE = 0.6

report_lines = []
report_lines.append(f"Live anchor detection report - {time.asctime()}\nConfig: {u.CONFIG_FILE}\n")

for name, data in locations.items():
    anchors = []
    single = data.get('anchor')
    if single:
        anchors.append(single)
    else:
        multi = data.get('anchors')
        if isinstance(multi, list): anchors.extend(multi)
    if not anchors:
        report_lines.append(f"{name}: NO_ANCHOR_CONFIGURED")
        continue
    found_any = False
    for i, anchor in enumerate(anchors):
        image_file = anchor.get('image_file')
        region_key = anchor.get('region_key')
        conf = float(anchor.get('confidence', 0.8))
        if GLOBAL_CONF_OVERRIDE is not None:
            conf = float(GLOBAL_CONF_OVERRIDE)
        if not image_file or not region_key:
            report_lines.append(f"{name}[{i}]: MISSING_IMAGE_OR_REGION")
            continue
        region = ui_regions.get(region_key)
        if not region:
            report_lines.append(f"{name}[{i}]: REGION_KEY_NOT_FOUND ({region_key})")
            continue

        tpl = u.load_image(image_file)
        if tpl is None:
            report_lines.append(f"{name}[{i}]: TEMPLATE_NOT_FOUND ({image_file})")
            continue

        res = bot.find_image_on_screen(image_file, region=region, confidence=conf)
        if res:
            cx, cy, score = res
            th, tw = tpl.shape[:2]
            left = int(cx - tw//2); top = int(cy - th//2)
            right = left + tw; bottom = top + th
            cv2.rectangle(annot, (left, top), (right, bottom), (0,255,0), 3)
            cv2.putText(annot, f"{name}:{score:.2f}", (left, max(top-8,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            report_lines.append(f"{name}[{i}]: FOUND image={image_file} region={region_key} score={score:.3f} bbox=({left},{top},{right},{bottom})")
            found_any = True
        else:
            report_lines.append(f"{name}[{i}]: NOT_FOUND image={image_file} region={region_key} conf={conf}")

    if not found_any:
        report_lines.append(f"{name}: NO_ANCHOR_FOUND_ON_SCREEN")

# Save annotated image and report
annotated_path = os.path.join(OUT_DIR, 'annotated_full.png')
cv2.imwrite(annotated_path, annot)
report_path = os.path.join(OUT_DIR, 'location_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print('\nReport saved to', report_path)
print('Annotated image saved to', annotated_path)
print('Done.')
