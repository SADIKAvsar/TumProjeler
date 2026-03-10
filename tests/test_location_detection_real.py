import sys, os, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.bot as mb
import modules.location_manager as lm_mod
import modules.utils as u
import cv2, numpy as np


def run_test_for_location(location_key="EXP_FARM"):
	cfg = u.load_config(u.CONFIG_FILE)
	general = cfg.get("module_general", {})
	locations = general.get("locations", {})
	loc_def = locations.get(location_key)
	if not loc_def:
		print("Location not in config:", location_key); return
	anchor = loc_def.get("anchor") or {}
	image_file = anchor.get("image_file")
	region_key = anchor.get("region_key")
	if not image_file or not region_key:
		print("No anchor/image or region for", location_key); return

	ui_regions = general.get("ui_regions", {})
	search_region = ui_regions.get(region_key)
	if not search_region:
		print("Region key not found in ui_regions:", region_key); return

	# Dummy bot
	class Dummy: pass
	bot = Dummy(); bot.ui_regions = ui_regions
	bot.log = lambda msg: print("[BOT LOG]", msg)

	# bind LoABot.find_image_on_screen as instance method to dummy
	bot.find_image_on_screen = types.MethodType(mb.LoABot.find_image_on_screen, bot)

	# Prepare fake sct that returns a screen image containing the template at top-left
	template = u.load_image(image_file)
	if template is None:
		print("Template image not found in image/:", image_file); return

	tpl_bgra = cv2.cvtColor(template, cv2.COLOR_GRAY2BGRA)

	class FakeSct:
		def __enter__(self): return self
		def __exit__(self, exc_type, exc, tb): return False
		def grab(self, monitor):
			h = monitor.get('height')
			w = monitor.get('width')
			screen = np.zeros((h, w, 4), dtype=np.uint8)
			th, tw = tpl_bgra.shape[:2]
			# place template at top-left
			screen[0:th, 0:tw, :] = tpl_bgra
			return screen

	# Monkeypatch mss.mss used inside find_image_on_screen
	original_mss = mb.mss.mss
	try:
		mb.mss.mss = lambda: FakeSct()

		# Call find_image_on_screen directly
		res = bot.find_image_on_screen(image_file, region=search_region, confidence=0.9)
		print("find_image_on_screen result:", res)

		# Now test location manager detection
		lm = lm_mod.LocationManager(bot, general)
		detected = lm.update_visual_location()
		print("_detect_visual_location ->", detected, "name:", lm.get_region_name())
	finally:
		mb.mss.mss = original_mss


if __name__ == '__main__':
	run_test_for_location('EXP_FARM')
