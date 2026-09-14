"""Watch the fight once, hold a key while the box is on the wrong person.

Clip-by-clip labelling costs about five seconds a clip and asks the hardest
question in the footage - *which technique was that* - on athletes sixty pixels
tall. A hundred and forty clips is a quarter of an hour, and the session that
prompted this came back saying most clips were not even the right person, which
the pack could not record and which makes every technique answer worthless
anyway.

This asks the easy question instead, continuously. The fight plays with the
analysis drawn over it; hold **a** while fighter A's box is on the wrong person
and **b** while B's is. Silence means correct. Watching a two-minute bout takes
two minutes and produces identity ground truth at frame resolution for every
moment of it - against twelve minutes for a sparse, clip-level sample of a
harder question.

It measures the thing that is actually broken. Detection finds a fighter in
roughly half the sampled frames of the reference bout, and the most confident
box in the picture is a person who has not moved in a hundred and sixteen
seconds. Coverage counts whether *something* was tracked. This counts whether
it was the right something, which is the number that decides whether any
technique label is worth collecting at all.

    tools/mark_identity.py --job label_f1 --video fights/1.mp4

Answers save as each mark ends, so closing the tab loses nothing. Bound to
127.0.0.1: nothing here is authenticated because nothing here is reachable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MARKS_DIR = PROJECT_ROOT / "labelpack"
MAX_BODY_BYTES = 4 * 1024 * 1024


def load_tracking(job: str) -> list[dict]:
    """The analysed frames, reduced to what the overlay needs."""
    path = OUTPUTS / job / "tracking.jsonl"
    if not path.exists():
        raise SystemExit(f"no tracking record at {path}")
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        entry = {"t": round(float(row.get("time_seconds") or 0.0), 4)}
        for name, key in (("A", "fighter_A"), ("B", "fighter_B")):
            observation = (row.get(key) or {}).get("observation") or {}
            box = observation.get("box")
            if box and len(box) == 4:
                entry[name] = [round(float(v), 1) for v in box]
        frames.append(entry)
    frames.sort(key=lambda f: f["t"])
    return frames


PAGE = """<!doctype html><meta charset="utf-8"><title>WarriorIQ - is it the right person?</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#070b12;color:#dbe3ef;font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
 header{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid #1e2b3f;flex-wrap:wrap}
 h1{font-size:15px;margin:0;letter-spacing:-.01em}
 .pill{padding:4px 9px;border:1px solid #2b4163;border-radius:7px;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase}
 .ok{color:#65d4ab;border-color:#2f6d58}.warn{color:#ffb020;border-color:#6b5321}.bad{background:#b3261e;color:#fff;border-color:#b3261e}
 main{padding:16px 18px;max-width:1100px}
 .stage{position:relative;display:inline-block;max-width:100%}
 video{display:block;max-width:100%;border:1px solid #24344c;border-radius:10px;background:#02050a}
 canvas{position:absolute;inset:0;pointer-events:none}
 .keys{display:flex;gap:10px;margin:14px 0 6px;flex-wrap:wrap}
 .key{flex:1;min-width:220px;padding:12px 14px;border:1px solid #2b3d58;border-radius:10px;background:#0b1524}
 .key.live{border-color:#b3261e;background:#2a0f0d}
 .key b{display:block;font-size:13px}
 .key small{color:#8a9ab0}
 kbd{display:inline-block;min-width:20px;padding:2px 6px;border:1px solid #3a4c6b;border-radius:5px;background:#16233a;font:inherit;font-weight:800;text-align:center}
 .bar{height:8px;border-radius:4px;background:#16233a;overflow:hidden;margin-top:12px}
 .bar i{display:block;height:100%;background:#5e82e9;width:0}
 table{border-collapse:collapse;margin-top:16px;font-size:12px}
 td,th{padding:5px 12px 5px 0;text-align:left;color:#9fb0c6}
 th{color:#7d8ea6;font-weight:700}
 .A{color:#55a7ff}.B{color:#f1b84b}
 p.help{color:#8a9ab0;max-width:760px}
</style>
<header>
  <h1>Is the box on the right person?</h1>
  <span class="pill" id="job"></span>
  <span class="pill warn" id="state">not saved yet</span>
  <span class="pill" id="clock">0.0s</span>
</header>
<main>
  <p class="help">Play it and just watch. <b>Hold <kbd>a</kbd></b> whenever the blue box is on the wrong
  person &mdash; the referee, a spectator, the other fighter, or nothing at all. <b>Hold <kbd>b</kbd></b>
  for the yellow box. Let go when it is right again. Silence means correct, so most of the time you
  press nothing. <kbd>space</kbd> plays and pauses; <kbd>&larr;</kbd> and <kbd>&rarr;</kbd> jump two seconds.</p>
  <div class="stage"><video id="v" controls preload="auto" src="video"></video><canvas id="c"></canvas></div>
  <div class="keys">
    <div class="key" id="kA"><b class="A">A &mdash; hold <kbd>a</kbd> while wrong</b><small id="sA">right so far</small></div>
    <div class="key" id="kB"><b class="B">B &mdash; hold <kbd>b</kbd> while wrong</b><small id="sB">right so far</small></div>
  </div>
  <div class="bar"><i id="prog"></i></div>
  <table><thead><tr><th>fighter</th><th>marked wrong</th><th>share of what you watched</th><th>spans</th></tr></thead>
  <tbody><tr><td class="A">A</td><td id="wA">0.0s</td><td id="pA">&mdash;</td><td id="nA">0</td></tr>
  <tr><td class="B">B</td><td id="wB">0.0s</td><td id="pB">&mdash;</td><td id="nB">0</td></tr></tbody></table>
</main>
<script>
const JOB = __JOB__, FRAMES = __FRAMES__;
const v = document.getElementById("v"), c = document.getElementById("c"), ctx = c.getContext("2d");
const marks = { A: [], B: [] };          // closed [start, end] spans, seconds
const open = { A: null, B: null };       // a mark being held right now
let watchedTo = 0, saveState = "unproven";

function nearest(t){
  // The analysis samples a subset of frames, so the overlay shows the nearest
  // one rather than interpolating: an invented box between two real ones would
  // be a box nobody can be asked to judge.
  let lo = 0, hi = FRAMES.length - 1, best = null;
  while (lo <= hi){ const mid = (lo + hi) >> 1;
    if (FRAMES[mid].t <= t){ best = FRAMES[mid]; lo = mid + 1; } else hi = mid - 1; }
  if (!best) return null;
  return Math.abs(best.t - t) <= 0.5 ? best : null;
}
function fit(){ const r = v.getBoundingClientRect();
  c.width = Math.max(1, Math.round(r.width * devicePixelRatio));
  c.height = Math.max(1, Math.round(r.height * devicePixelRatio));
  c.style.width = r.width + "px"; c.style.height = r.height + "px"; draw(); }
function draw(){
  ctx.clearRect(0, 0, c.width, c.height);
  if (!v.videoWidth) return;
  const s = Math.min(c.width / v.videoWidth, c.height / v.videoHeight);
  const ox = (c.width - v.videoWidth * s) / 2, oy = (c.height - v.videoHeight * s) / 2;
  const f = nearest(v.currentTime);
  for (const [name, colour] of [["A", "#55a7ff"], ["B", "#f1b84b"]]){
    const b = f && f[name]; if (!b) continue;
    ctx.strokeStyle = open[name] !== null ? "#ff4438" : colour;
    ctx.lineWidth = Math.max(2, 3 * devicePixelRatio);
    ctx.strokeRect(ox + b[0] * s, oy + b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s);
    ctx.fillStyle = ctx.strokeStyle; ctx.font = `${Math.round(13 * devicePixelRatio)}px system-ui`;
    ctx.fillText(name, ox + b[0] * s + 3, oy + b[1] * s - 4);
  }
}
const total = (list) => list.reduce((sum, s) => sum + (s[1] - s[0]), 0);
function refresh(){
  document.getElementById("clock").textContent = v.currentTime.toFixed(1) + "s";
  document.getElementById("prog").style.width = (100 * watchedTo / Math.max(0.1, v.duration || 1)) + "%";
  for (const name of ["A", "B"]){
    const live = open[name] === null ? 0 : Math.max(0, v.currentTime - open[name]);
    const wrong = total(marks[name]) + live;
    document.getElementById("w" + name).textContent = wrong.toFixed(1) + "s";
    document.getElementById("p" + name).textContent =
      watchedTo > 1 ? (100 * wrong / watchedTo).toFixed(0) + "%" : "\\u2014";
    document.getElementById("n" + name).textContent = marks[name].length + (open[name] === null ? 0 : 1);
    document.getElementById("k" + name).classList.toggle("live", open[name] !== null);
    document.getElementById("s" + name).textContent =
      open[name] !== null ? "marking wrong…" : (wrong > 0 ? "wrong for " + wrong.toFixed(1) + "s" : "right so far");
  }
  document.getElementById("state").className = "pill " +
    (saveState === "server" ? "ok" : saveState === "unproven" ? "warn" : "bad");
  document.getElementById("state").textContent =
    saveState === "server" ? "saving to disk" : saveState === "unproven" ? "not saved yet" : "NOT SAVING";
}
function persist(){
  fetch("/" + encodeURIComponent(JOB) + "/marks", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ job: JOB, watched_seconds: watchedTo, duration: v.duration || 0, marks }),
  }).then((r) => { saveState = r.ok ? "server" : "none"; refresh(); })
    .catch(() => { saveState = "none"; refresh(); });
}
function start(name){ if (open[name] === null){ open[name] = v.currentTime; refresh(); draw(); } }
function stop(name){
  if (open[name] === null) return;
  const a = open[name], b = v.currentTime; open[name] = null;
  if (b - a > 0.08) marks[name].push([Number(a.toFixed(2)), Number(b.toFixed(2))]);
  persist(); refresh(); draw();
}
addEventListener("keydown", (e) => {
  if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "a"){ start("A"); e.preventDefault(); }
  else if (k === "b"){ start("B"); e.preventDefault(); }
  else if (k === " "){ v.paused ? v.play() : v.pause(); e.preventDefault(); }
  else if (e.key === "ArrowLeft"){ v.currentTime = Math.max(0, v.currentTime - 2); e.preventDefault(); }
  else if (e.key === "ArrowRight"){ v.currentTime += 2; e.preventDefault(); }
});
addEventListener("keyup", (e) => {
  const k = e.key.toLowerCase();
  if (k === "a") stop("A"); else if (k === "b") stop("B");
});
// Releasing a key while the window is not focused never fires keyup, which
// would leave a mark open and running until the end of the fight.
addEventListener("blur", () => { stop("A"); stop("B"); });
v.addEventListener("pause", () => { stop("A"); stop("B"); });
v.addEventListener("loadedmetadata", fit);
v.addEventListener("seeked", () => { stop("A"); stop("B"); draw(); });
v.addEventListener("timeupdate", () => { watchedTo = Math.max(watchedTo, v.currentTime); refresh(); });
v.addEventListener("play", () => {
  const tick = () => { if (!v.paused){ draw(); refresh(); requestAnimationFrame(tick); } }; tick();
});
addEventListener("resize", fit);
document.getElementById("job").textContent = JOB;
refresh();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    job = ""
    video = Path()
    frames: list[dict] = []

    def log_message(self, *_args) -> None:            # quiet; the page is the UI
        pass

    def _send(self, code: int, body: bytes, kind: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_video(self) -> None:
        """With Range support, because a <video> cannot be scrubbed without it."""
        data = self.video.read_bytes()
        span = self.headers.get("Range", "")
        match = re.match(r"bytes=(\d*)-(\d*)", span) if span else None
        if not match:
            return self._send(200, data, "video/mp4", {"Accept-Ranges": "bytes"})
        start = int(match.group(1) or 0)
        end = int(match.group(2) or len(data) - 1)
        end = min(end, len(data) - 1)
        if start > end:
            return self._send(416, b"", "video/mp4")
        chunk = data[start:end + 1]
        self._send(206, chunk, "video/mp4", {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{len(data)}",
        })

    def do_GET(self) -> None:                         # noqa: N802 - stdlib's spelling
        path = self.path.split("?", 1)[0].strip("/")
        if path in ("", self.job):
            page = (PAGE.replace("__JOB__", json.dumps(self.job))
                        .replace("__FRAMES__", json.dumps(self.frames)))
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if path in ("video", f"{self.job}/video"):
            return self._send_video()
        self._send(404, b"not here", "text/plain; charset=utf-8")

    def do_POST(self) -> None:                        # noqa: N802 - stdlib's spelling
        parts = self.path.split("?", 1)[0].strip("/").split("/")
        if len(parts) != 2 or parts[1] != "marks" or parts[0] != self.job:
            return self._send(404, b"not here", "text/plain; charset=utf-8")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send(400, b"bad length", "text/plain; charset=utf-8")
        if length <= 0 or length > MAX_BODY_BYTES:
            return self._send(400, b"bad length", "text/plain; charset=utf-8")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._send(400, b"bad json", "text/plain; charset=utf-8")
        payload["video"] = str(self.video)
        destination = MARKS_DIR / self.job / f"{self.job}-identity-marks.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.part")
        temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        temporary.replace(destination)
        self._send(200, b"ok", "text/plain; charset=utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="job id under outputs/")
    parser.add_argument("--video", required=True, help="the fight video that job analysed")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"no video at {video}")
    frames = load_tracking(args.job)
    if not frames:
        raise SystemExit("tracking record has no frames")

    Handler.job, Handler.video, Handler.frames = args.job, video, frames
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/{args.job}"
    covered = sum(1 for f in frames if "A" in f or "B" in f)
    print(f"{len(frames)} analysed frames, {covered} with a fighter box")
    print(f"watch it once and hold a / b while the box is wrong:  {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
