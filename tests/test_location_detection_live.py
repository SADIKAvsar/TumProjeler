import sys, os, types, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import modules.bot as mb
import modules.location_manager as lm_mod
import modules.utils as u

# Dummy bot for testing (no clicks)
class Dummy:
    pass

bot = Dummy()
cfg = u.load_config(u.CONFIG_FILE)
if not cfg:
    print('Config yüklenemedi:', u.CONFIG_FILE); sys.exit(1)

general = cfg.get('module_general', {})
bot.ui_regions = general.get('ui_regions', {})
# bind find_image_on_screen
import types
bot.find_image_on_screen = types.MethodType(mb.LoABot.find_image_on_screen, bot)
bot.log = lambda msg: print('[BOT LOG]', msg)

locations = general.get('locations', {})

print('Başlatılıyor: gerçek ekran testi. Oyun açık olmalı ve görüntüler image/ içinde bulunmalı.')
print('Toplam konum:', len(locations))

for name, data in locations.items():
    anchors = []
    single = data.get('anchor')
    if single:
        anchors.append(single)
    else:
        multi = data.get('anchors')
        if isinstance(multi, list): anchors.extend(multi)
    if not anchors:
        print(name, '-> no anchors configured')
        continue
    for anchor in anchors:
        img = anchor.get('image_file')
        region_key = anchor.get('region_key')
        conf = float(anchor.get('confidence', 0.8))
        if not img or not region_key:
            print(name, '-> anchor missing image or region')
            continue
        region = bot.ui_regions.get(region_key)
        if not region:
            print(name, '-> region key not found:', region_key)
            continue
        print('\nAranıyor:', name, img, 'in region', region_key, 'confidence', conf)
        res = bot.find_image_on_screen(img, region=region, confidence=conf)
        print(' Sonuç:', res)
        time.sleep(0.2)

print('\nTest tamamlandı.')
