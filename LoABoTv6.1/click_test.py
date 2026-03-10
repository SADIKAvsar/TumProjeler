# -*- coding: utf-8 -*-
import time, pyautogui, os
from utils import log_to_file

pyautogui.FAILSAFE = False
time.sleep(3)  # 3s içinde imleci gözle
x, y = pyautogui.position()
log_to_file(f"CLICK_TEST: current mouse pos {x},{y}")
try:
    pyautogui.click(x=x, y=y)
    log_to_file("CLICK_TEST: pyautogui.click executed")
except Exception as e:
    log_to_file(f"CLICK_TEST ERROR: {e}")
print("Done")