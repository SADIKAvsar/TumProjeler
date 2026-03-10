import sys, os, types

# Ensure project root (parent of tests/) is on sys.path so 'modules' package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules import bot as mb

class Dummy: pass

dummy = Dummy()
dummy.ui_regions = {}

find = types.MethodType(mb.LoABot.find_image_on_screen, dummy)


def safe_call(region):
    try:
        res = find("nonexistent_image_file_hopefully.png", region=region, confidence=0.5)
        print("REGION", region, "->", res)
    except Exception as e:
        print("ERROR for region", region, ":", e)


safe_call({"x":100, "y":200, "w":300, "h":400})
safe_call({"top":10, "left":20, "width":30, "height":40})
safe_call(None)
print("TEST_DONE")
