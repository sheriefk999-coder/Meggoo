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


def tg(method, files=None, **data):
    if not (TG_TOKEN and TG_CHAT):
        return None
    data["chat_id"] = TG_CHAT
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
                          data=data, files=files, timeout=300)
        if not r.ok:
            print("telegram:", r.status_code, r.text[:200])
        return r
    except requests.RequestException as e:
        print("telegram error:", e)


# ---------------- platform + download ----------------
def pick_platform(url):
    checked = [n for n, (_, flag, _) in PLATFORMS.items() if os.environ.get(flag) == "true"]
    if os.environ.get("FROM_OTHER") == "true":
        checked.append("other")
    detected = next((n for n, (doms, _, _) in PLATFORMS.items() if any(d in url for d in doms)), "other")
    if len(checked) == 1 and checked[0] != detected:
        print(f"⚠️ اخترت {checked[0]} بس اللينك شكله {detected} — هستخدم اختيارك")
        return checked[0]
    return detected


def download(url, platform, work):
    cmd = ["yt-dlp", "--no-playlist", "--no-simulate", "--print", "after_move:title",
           "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
           "--merge-output-format", "mp4", "-o", str(work / "source.%(ext)s"), url]
    if platform in PLATFORMS:
        cookies = os.environ.get(PLATFORMS[platform][2], "").strip()
        if cookies:
            (work / "cookies.txt").write_text(cookies)
            cmd[1:1] = ["--cookies", str(work / "cookies.txt")]
            print(f"🍪 using {platform} cookies")
    title = subprocess.check_output(cmd, text=True).strip().splitlines()[-1]
    src = next(work.glob("source.*"))
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", work / "audio.wav"])
    return src, title


def probe(src):
    d = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(src)]))
    s = d["streams"][0]
    num, den = (s.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1) or 30.0
    return s["width"], s["height"], float(d["format"]["duration"]), min(fps, 60.0)


# ---------------- transcription ----------------
_whisper = None


def transcribe(work):
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                                cpu_threads=os.cpu_count() or 4)
    segments, info = _whisper.transcribe(str(work / "audio.wav"), word_timestamps=True,
                                         vad_filter=True, beam_size=5)
    segs, words = [], []
    for s in segments:
        segs.append((s.start, s.end, s.text.strip()))
        words += [(w.start, w.end, w.word.strip()) for w in (s.words or []) if w.word.strip()]
    print(f"📝 language={info.language} segments={len(segs)} words={len(words)}")
    return segs, words


# ---------------- viral moment selection ----------------
PROMPT = """You are a top short-form video editor (TikTok, YouTube Shorts, Reels).
From the timestamped transcript below, pick the {n} moments most likely to go viral.

Judge each moment on: a strong hook in the first 3 seconds, emotion or surprise,
a clear payoff, and whether it makes sense without the rest of the video.
Rules:
- 20 to 60 seconds each, no overlap
- start exactly where a sentence starts; end on a finished thought, never mid-sentence
- write title, hook, description and hashtags in the same language and dialect as the speaker
- hook: max 6 words, shown on screen in the first 3 seconds, must create curiosity
- score: honest 1-10 virality estimate

Return ONLY JSON:
{{"clips": [{{"start": 12.3, "end": 48.0, "score": 8, "title": "...", "hook": "...",
"description": "one or two sentences", "hashtags": ["#..."], "reason": "why it works"}}]}}

Transcript:
{transcript}"""


def gemini_json(prompt):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}}
    last = None
    for model in GEMINI_MODELS:
        for attempt in range(4):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}",
                    json=body, timeout=300)
                if r.status_code == 404:
                    last = f"{model} not found"
                    break
                if r.status_code in (429, 500, 503):
                    last = r.text[:200]
                    time.sleep(15 * (attempt + 1))
                    continue
                r.raise_for_status()
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M)
                return json.loads(txt)
            except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
                last = str(e)
                time.sleep(5)
    raise RuntimeError(f"Gemini failed: {last}")


def select_clips(segs, duration):
    chunks, cur, size = [], [], 0
    for s, e, t in segs:
        line = f"[{s:.1f}-{e:.1f}] {t}"
        if size + len(line) > 60000 and cur:
            chunks.append(cur)
            cur, size = [], 0
        cur.append((s, e, line))
        size += len(line)
    if cur:
        chunks.append(cur)

    cands = []
    for ch in chunks:
        span = ch[-1][1] - ch[0][0]
        k = max(2, math.ceil(CLIPS_COUNT * span / max(duration, 1)) + 1)
        data = gemini_json(PROMPT.format(n=k, transcript="\n".join(c[2] for c in ch)))
        cands += data.get("clips", []) if isinstance(data, dict) else data

    starts = [s for s, _, _ in segs]
    ends = [e for _, e, _ in segs]
    good = []
    for c in cands:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        ns = min(starts, key=lambda x: abs(x - s))
        ne = min(ends, key=lambda x: abs(x - e))
        s = ns if abs(ns - s) <= 3 else s
        e = ne if abs(ne - e) <= 3 else e
        s, e = max(0.0, s - 0.15), min(duration, e + 0.35)
        if 15 <= e - s <= 90:
            c.update(start=s, end=e)
            good.append(c)

    good.sort(key=lambda c: -float(c.get("score") or 0))
    picked = []
    for c in good:
        if all(min(c["end"], p["end"]) - max(c["start"], p["start"]) < 2 for p in picked):
            picked.append(c)
        if len(picked) == CLIPS_COUNT:
            break
    return picked


# ---------------- face tracking ----------------
class FaceDetector:
    def __init__(self):
        self.yn = None
        p = Path("models/yunet.onnx")
        if p.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yn = cv2.FaceDetectorYN.create(str(p), "", (320, 320), 0.7, 0.3, 50)
            except cv2.error as e:
                print("YuNet failed:", e)
        if self.yn is None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("face detector:", "YuNet" if self.yn is not None else "Haar")

    def detect(self, frame):
        h, w = frame.shape[:2]
        sc = 640 / w if w > 640 else 1.0
        img = cv2.resize(frame, (int(w * sc), int(h * sc))) if sc != 1.0 else frame
        out = []
        if self.yn is not None:
            self.yn.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.yn.detect(img)
            for f in (faces if faces is not None else []):
                x, y, fw, fh = f[:4]
                out.append(((x + fw / 2) / sc, fw * fh / sc / sc))
        else:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            m = img.shape[0] // 14
            for (x, y, fw, fh) in self.haar.detectMultiScale(g, 1.15, 5, minSize=(m, m)):
                out.append(((x + fw / 2) / sc, fw * fh / sc / sc))
        return out


def analyze(src, cs, ce, fps, det):
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_MSEC, cs * 1000)
    every = max(1, int(round(fps * 0.25)))
    times, faces = [], []
    for n in range(int((ce - cs) * fps)):
        if not cap.grab():
            break
        if n % every:
            continue
        ok, frame = cap.retrieve()
        if ok:
            times.append(n / fps)
            faces.append(det.detect(frame))
    cap.release()
    return times, faces


def plan_track(times, faces, W, cw):
    n = max(1, len(faces))
    face_ratio = sum(1 for f in faces if f) / n
    multi_ratio = sum(1 for f in faces if len(f) >= 2 and
                      max(c for c, _ in f) - min(c for c, _ in f) > cw * 0.8) / n
    xs, cur = [], None
    for f in faces:
        if not f:
            xs.append(None)
            continue
        biggest = max(f, key=lambda c: c[1])
        if cur is None:
            cur = biggest[0]
        else:
            near = min(f, key=lambda c: abs(c[0] - cur))
            pick = biggest if biggest[1] > near[1] * 1.6 else near
            if abs(pick[0] - cur) > cw * 0.12:  # dead zone: ignore small jitter
                cur = pick[0]
        xs.append(cur)
    last, filled = None, []
    for x in xs:
        last = x if x is not None else last
        filled.append(last)
    first = next((x for x in filled if x is not None), W / 2)
    arr = np.array([first if x is None else x for x in filled], float)
    k = 5
    for _ in range(2):  # double moving average = smooth pan
        if len(arr) >= k:
            arr = np.convolve(np.pad(arr, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return np.array(times), arr, face_ratio, multi_ratio


# ---------------- captions (ASS) ----------------
def ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def esc(t):
    return str(t).replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


CAP_STYLES = {
    "highlight": "Style: Cap,Noto Sans Arabic,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,470,1",
    "box": "Style: Cap,Noto Sans Arabic,76,&H00FFFFFF,&H00FFFFFF,&HB4000000,&HB4000000,-1,0,0,0,100,100,0,0,3,14,0,2,80,80,470,1",
    "minimal": "Style: Cap,Noto Sans Arabic,70,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,4,2,2,80,80,470,1",
}
HOOK_STYLE = "Style: Hook,Noto Sans Arabic,68,&H00000000,&H00000000,&H0000E5FF,&H0000E5FF,-1,0,0,0,100,100,0,0,3,18,0,8,80,80,200,1"
ACTIVE = {
    "highlight": (r"{\c&H00E5FF&\fscx118\fscy118\t(0,140,\fscx100\fscy100)}", r"{\c&HFFFFFF&\fscx100\fscy100}"),
    "box": (r"{\c&H66FF33&}", r"{\c&HFFFFFF&}"),
}


def make_ass(words, clip, path):
    cs, ce = clip["start"], clip["end"]
    ws = [(s - cs, e - cs, esc(w)) for s, e, w in words if s >= cs and e <= ce]
    style = CAPTION_STYLE if CAPTION_STYLE in CAP_STYLES else "highlight"
    lines = []
    if CAPTION_STYLE != "none":
        size = 4 if style == "minimal" else 3
        for g in range(0, len(ws), size):
            grp = ws[g:g + size]
            if style == "minimal":
                lines.append(f"Dialogue: 0,{ass_time(grp[0][0])},{ass_time(grp[-1][1] + 0.2)},Cap,,0,0,0,,"
                             + " ".join(w for _, _, w in grp))
                continue
            on, off = ACTIVE[style]
            for i, (s, e, _) in enumerate(grp):
                end = grp[i + 1][0] if i + 1 < len(grp) else e + 0.25
                text = " ".join(on + w + off if j == i else w for j, (_, _, w) in enumerate(grp))
                lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(end)},Cap,,0,0,0,,{text}")
    if clip.get("hook"):
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(3.2)},Hook,,0,0,0,,"
                     + r"{\fad(150,300)}" + esc(clip["hook"]))
    if not lines:
        return None
    header = "\n".join([
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        CAP_STYLES[style], HOOK_STYLE, "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text", ""])
    path.write_text(header + "\n".join(lines), encoding="utf-8")
    return path


# ---------------- rendering ----------------
def render_track(src, clip, out, ass, W, H, fps, times, xs):
    cs, dur = clip["start"], clip["end"] - clip["start"]
    cw = int(H * 9 / 16) // 2 * 2
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1080x1920",
           "-r", f"{fps:.3f}", "-i", "-", "-ss", f"{cs:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
           "-map", "0:v", "-map", "1:a?"]
    if ass:
        cmd += ["-vf", f"ass={ass}"]
    p = subprocess.Popen(cmd + ENC + [str(out)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_MSEC, cs * 1000)
    interp = cv2.INTER_AREA if cw > 1080 else cv2.INTER_CUBIC
    for n in range(int(dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        fw = frame.shape[1]
        cx = float(np.interp(n / fps, times, xs)) if len(times) else fw / 2
        x = int(min(max(cx - cw / 2, 0), fw - cw))
        p.stdin.write(cv2.resize(frame[:, x:x + cw], (1080, 1920), interpolation=interp).tobytes())
    cap.release()
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed while rendering tracked clip")


def render_filter(src, clip, out, ass, layout):
    cs, dur = clip["start"], clip["end"] - clip["start"]
    sub = f",ass={ass}" if ass else ""
    if layout == "blur":
        fc = ("[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
              f"gblur=sigma=30[bg];[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2{sub}[v]")
    else:  # already vertical
        fc = f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920{sub}[v]"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{cs:.3f}", "-t", f"{dur:.3f}", "-i", src,
         "-filter_complex", fc, "-map", "[v]", "-map", "0:a?"] + ENC + [out])


def fit_telegram(path):
    if path.stat().st_size <= 49 * 1024 * 1024:
        return path
    small = path.with_name(path.stem + "_tg.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "30", "-maxrate", "5M", "-bufsize", "10M", "-c:a", "aac", "-b:a", "128k", small])
    return small


# ---------------- pipeline ----------------
def process(n, url, det):
    work = ROOT / str(n)
    work.mkdir(parents=True, exist_ok=True)
    platform = pick_platform(url)
    tg("sendMessage", text=f"⏳ بدأت أشتغل على ({platform}):\n{url}")

    src, title = download(url, platform, work)
    W, H, duration, fps = probe(src)
    segs, words = transcribe(work)
    if not segs:
        raise RuntimeError("مفيش كلام اتفرّغ من الفيديو")
    clips = select_clips(segs, duration)
    if not clips:
        raise RuntimeError("Gemini مرجّعش مقاطع صالحة")
    tg("sendMessage", text=f"✂️ لقيت {len(clips)} مقاطع في «{title}» — بجهّزهم...")

    landscape = W / H > 0.6
    cw = int(H * 9 / 16) // 2 * 2
    meta = []
    for i, c in enumerate(clips, 1):
        out = work / f"clip_{i}.mp4"
        ass = make_ass(words, c, work / f"clip_{i}.ass")
        if not landscape:
            layout = "vertical"
            render_filter(src, c, out, ass, layout)
        elif LAYOUT == "blur":
            layout = "blur"
            render_filter(src, c, out, ass, layout)
        else:
            times, faces = analyze(src, c["start"], c["end"], fps, det)
            t, xs, face_ratio, multi_ratio = plan_track(times, faces, W, cw)
            layout = "track" if LAYOUT == "track" or (face_ratio >= 0.3 and multi_ratio < 0.35) else "blur"
            print(f"clip {i}: faces={face_ratio:.0%} multi={multi_ratio:.0%} -> {layout}")
            if layout == "track":
                render_track(src, c, out, ass, W, H, fps, t, xs)
            else:
                render_filter(src, c, out, ass, layout)

        tags = " ".join(h if str(h).startswith("#") else f"#{h}" for h in c.get("hashtags", []))
        dur = round(c["end"] - c["start"])
        caption = (f"🎬 {i}/{len(clips)} — {c.get('title', '')}\n⭐ {c.get('score', '?')}/10 · ⏱ {dur}s\n\n"
                   f"{c.get('description', '')}\n\n{tags}")
        sendable = fit_telegram(out)
        with open(sendable, "rb") as f:
            tg("sendVideo", files={"video": f}, caption=caption[:1024], width=1080, height=1920,
               supports_streaming="true")
        meta.append({**c, "file": out.name, "layout": layout, "source_title": title, "url": url})

    (work / "clips.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tg("sendMessage", text=f"✅ خلصت «{title}» — {len(clips)} مقاطع")


def main():
    if not URLS:
        sys.exit("مفيش لينك صالح")
    det = FaceDetector()
    failed = 0
    for n, url in enumerate(URLS, 1):
        try:
            process(n, url, det)
        except Exception as e:  # keep going with the next link
            traceback.print_exc()
            failed += 1
            tg("sendMessage", text=f"❌ فشل:\n{url}\n{str(e)[:400]}")
    if failed == len(URLS):
        sys.exit(1)


if __name__ == "__main__":
    main()
