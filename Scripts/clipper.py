"""Moggo-style clipper v2: long video -> ranked vertical clips with speaker tracking,
animated captions, on-screen hook, and ready-to-post metadata, delivered to Telegram."""
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path("work")
ROOT.mkdir(exist_ok=True)

URLS = [u for u in re.split(r"[\s,]+", os.environ["VIDEO_URL"]) if u.startswith("http")]
CLIPS_COUNT = int(os.environ.get("CLIPS_COUNT", "5"))
CAPTION_STYLE = os.environ.get("CAPTION_STYLE", "highlight")
LAYOUT = os.environ.get("LAYOUT", "auto")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL") or "small"
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODELS = list(dict.fromkeys(
    [os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), "gemini-2.5-flash-lite"]))
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")

PLATFORMS = {
    "youtube": (["youtube.com", "youtu.be"], "FROM_YOUTUBE", "YT_COOKIES"),
    "tiktok": (["tiktok.com"], "FROM_TIKTOK", "TT_COOKIES"),
    "facebook": (["facebook.com", "fb.watch", "fb.com"], "FROM_FACEBOOK", "FB_COOKIES"),
    "instagram": (["instagram.com"], "FROM_INSTAGRAM", "IG_COOKIES"),
}

ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart"]


def run(cmd):
    print("+", " ".join(map(str, cmd))[:300], flush=True)
    subprocess.run([str(c) for c in cmd], check=True)

