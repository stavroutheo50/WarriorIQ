/* Pick the fighters while the video is still uploading (2026-10-27).
 *
 * A 3-minute phone video took 11 minutes to upload at 0.5 MB/s, and picking
 * the fighters only began after it. This takes frames from the file on the
 * phone, asks the server who is in them (/api/upload/<id>/people), and lets
 * the two fighters be tapped while the bytes are still travelling. When the
 * upload finishes, the frame goes to /api/selection-frame and the boxes to
 * /api/start - the fighter page's own calls - and the analysis starts.
 *
 * Anything this cannot do (a browser that cannot play the file, a frame the
 * server will not take, a refused start) sends the person to the fighter page
 * exactly as before, so it can only save time, never cost an upload.
 */
(() => {
  const panel = document.getElementById('earlyPick');
  if (!panel || !window.fetch || !window.HTMLCanvasElement) return;
  const canvas = document.getElementById('earlyPickCanvas');
  const status = document.getElementById('earlyPickStatus');
  const title = document.getElementById('earlyPickTitle');
  const redo = document.getElementById('earlyPickRedo');
  const other = document.getElementById('earlyPickOther');
  const fallback = document.getElementById('earlyPickFallback');
  const ctx = canvas.getContext('2d');
  const SAMPLE_SHARES = [0.08, 0.2, 0.35, 0.5, 0.65];
  const DETECT_WIDTH = 960;

  let job = null, nameA = 'You', video = null, url = null;
  let frames = [], shown = 0, boxA = null, boxB = null, drawnA = false, drawnB = false;
  let upload = null, finishing = false, abandoned = false, drag = null;

  const say = (text) => { status.textContent = text; };
  const current = () => frames[shown];
  const wants = () => !abandoned && frames.length > 0;
  const done = () => Boolean(boxA && boxB);

  const blobOf = (source, width, height, quality) => new Promise((resolve) => {
    const c = document.createElement('canvas'); c.width = width; c.height = height;
    c.getContext('2d').drawImage(source, 0, 0, width, height);
    c.toBlob(resolve, 'image/jpeg', quality);
  });
  const seek = (time) => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('seek')), 8000);
    video.addEventListener('seeked', () => { clearTimeout(timer); resolve(); }, { once: true });
    video.currentTime = time;
  });

  async function grab(time) {
    await seek(time);
    const w = video.videoWidth, h = video.videoHeight;
    // The full picture is what /api/selection-frame keeps; a smaller copy is
    // enough to find people and costs the phone's uplink less.
    const full = await blobOf(video, w, h, 0.9);
    const scale = Math.min(1, DETECT_WIDTH / w);
    const small = await blobOf(video, Math.round(w * scale), Math.round(h * scale), 0.85);
    const bitmap = await createImageBitmap(full);
    const response = await fetch(`/api/upload/${job}/people`, {
      method: 'POST', headers: wiqCsrfHeaders({ 'Content-Type': 'image/jpeg' }), body: small,
    });
    const found = response.ok ? await response.json() : { people: [], score: 0 };
    const people = (found.people || []).map((p) => ({ box: p.box.map((v) => v / scale) }));
    return { time, full, bitmap, width: w, height: h, people, score: Number(found.score) || 0,
             suggested: found.suggested || null };
  }

  function layout() {
    const frame = current(); if (!frame) return 1;
    // Drawn at the screen's own pixel density: a phone at 3x showed the
    // dotted boxes as a blur too faint to see.
    const width = canvas.parentElement.clientWidth || 320;
    const k = (width * (window.devicePixelRatio || 1)) / frame.width;
    canvas.width = Math.round(frame.width * k); canvas.height = Math.round(frame.height * k);
    canvas.style.height = `${Math.round(width * frame.height / frame.width)}px`;
    return k;
  }
  function draw() {
    const frame = current(); if (!frame) return;
    const k = layout();
    ctx.drawImage(frame.bitmap, 0, 0, canvas.width, canvas.height);
    const px = canvas.width / (canvas.clientWidth || canvas.width);
    const outline = (box) => ctx.strokeRect(box[0] * k, box[1] * k, (box[2] - box[0]) * k, (box[3] - box[1]) * k);
    // White dashes over a dark line, so the boxes show on a bright ring too.
    frame.people.forEach(({ box }) => {
      ctx.setLineDash([]); ctx.lineWidth = 3.5 * px; ctx.strokeStyle = 'rgba(2,5,11,.55)'; outline(box);
      ctx.setLineDash([6 * px, 4 * px]); ctx.lineWidth = 2 * px; ctx.strokeStyle = '#ffffff'; outline(box);
    });
    ctx.setLineDash([]); ctx.lineWidth = 3 * px; ctx.font = `600 ${13 * px}px system-ui, sans-serif`;
    [[boxA, '#4ea1ff', `A · ${nameA}`], [boxB, '#f5b942', 'B · Opponent']].forEach(([box, colour, label]) => {
      if (!box) return;
      ctx.strokeStyle = colour; ctx.strokeRect(box[0] * k, box[1] * k, (box[2] - box[0]) * k, (box[3] - box[1]) * k);
      const tw = ctx.measureText(label).width + 10 * px;
      ctx.fillStyle = colour; ctx.fillRect(box[0] * k, Math.max(0, box[1] * k - 20 * px), tw, 20 * px);
      ctx.fillStyle = '#081020'; ctx.fillText(label, box[0] * k + 5 * px, Math.max(14 * px, box[1] * k - 6 * px));
    });
    if (drag && drag.moved) {
      ctx.strokeStyle = boxA ? '#f5b942' : '#4ea1ff'; ctx.lineWidth = 2 * px;
      ctx.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0);
    }
    describe();
  }
  function shortBox() {
    if (!boxA || !boxB) return null;
    const ha = boxA[3] - boxA[1], hb = boxB[3] - boxB[1];
    if (Math.min(ha, hb) >= 0.6 * Math.max(ha, hb)) return null;
    return ha < hb ? 'A' : 'B';
  }
  function describe() {
    other.hidden = frames.length < 2; redo.hidden = !boxA;
    if (!boxA) say(current().people.length ? `Tap ${nameA}, or drag a box around them.` : `Drag a box around ${nameA}, head to feet.`);
    else if (!boxB) say('Now tap the opponent, or drag a box around them.');
    else if (shortBox()) say(`${shortBox() === 'A' ? nameA : 'The opponent'}'s box is much shorter than the other. Check it covers the whole fighter, head to feet; if not, press Start over.`);
    else say(upload ? 'Starting the analysis…' : 'Done. The analysis starts by itself the moment the upload finishes.');
  }

  const point = (event) => {
    const r = canvas.getBoundingClientRect();
    return [(event.clientX - r.left) * canvas.width / r.width, (event.clientY - r.top) * canvas.height / r.height];
  };
  canvas.addEventListener('pointerdown', (event) => {
    if (done() || finishing) return;
    const [x, y] = point(event); drag = { x0: x, y0: y, x1: x, y1: y, moved: false };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!drag) return;
    const [x, y] = point(event); drag.x1 = x; drag.y1 = y;
    drag.moved = drag.moved || Math.hypot(x - drag.x0, y - drag.y0) > 12;
    if (drag.moved) draw();
  });
  canvas.addEventListener('pointerup', () => {
    if (!drag) return;
    const k = canvas.width / current().width, d = drag; drag = null;
    let box = null, drawn = false;
    if (d.moved) {
      box = [Math.min(d.x0, d.x1) / k, Math.min(d.y0, d.y1) / k, Math.max(d.x0, d.x1) / k, Math.max(d.y0, d.y1) / k];
      drawn = true;
      if (box[2] - box[0] < 8 || box[3] - box[1] < 16) box = null;
    } else {
      const x = d.x0 / k, y = d.y0 / k;
      const hits = current().people.filter(({ box: b }) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]);
      hits.sort((p, q) => (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]) - (q.box[2] - q.box[0]) * (q.box[3] - q.box[1]));
      if (hits.length) box = [...hits[0].box];
    }
    if (box) {
      if (!boxA) { boxA = box; drawnA = drawn; } else if (!boxB) { boxB = box; drawnB = drawn; }
    }
    draw();
    if (done() && upload) start();
  });
  redo.addEventListener('click', () => { boxA = boxB = null; drawnA = drawnB = false; draw(); });
  other.addEventListener('click', () => { shown = (shown + 1) % frames.length; boxA = boxB = null; draw(); });
  window.addEventListener('resize', () => { if (frames.length) draw(); });

  async function begin(file, jobId, name) {
    job = jobId; nameA = (name || '').trim() || 'You';
    if (typeof createImageBitmap !== 'function') { abandoned = true; return; }
    url = URL.createObjectURL(file);
    video = document.createElement('video');
    video.muted = true; video.playsInline = true; video.preload = 'auto'; video.src = url;
    try {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('metadata')), 10000);
        video.addEventListener('loadeddata', () => { clearTimeout(timer); resolve(); }, { once: true });
        video.addEventListener('error', () => { clearTimeout(timer); reject(new Error('decode')); }, { once: true });
      });
      if (!(video.videoWidth > 0) || !isFinite(video.duration)) throw new Error('no picture');
      panel.hidden = false; say('Finding your fighters in the video…');
      for (const share of SAMPLE_SHARES) {
        if (upload && !frames.length) break;            // the upload won the race: use the fighter page
        frames.push(await grab(Math.max(0.5, share * video.duration)));
        if (frames.length === 1) draw();
      }
      // The clearest moment first, as the fighter page does (core/person_detect.py).
      const best = Math.max(...frames.map((f) => f.score));
      if (!boxA && best > 0) {
        const pick = frames.findIndex((f) => f.score >= 0.9 * best);
        const chosen = frames.splice(pick, 1)[0]; frames.unshift(chosen); shown = 0; draw();
      }
    } catch (error) {
      if (!frames.length) { abandoned = true; panel.hidden = true; }
    }
  }

  async function start() {
    if (finishing || !upload || !done()) return;
    finishing = true; say('Starting the analysis…');
    const frame = current(), id = upload.job_id;
    try {
      let response = await fetch(`/api/selection-frame/${id}/image?seconds=${frame.time.toFixed(3)}`, {
        method: 'POST', headers: wiqCsrfHeaders({ 'Content-Type': 'image/jpeg' }), body: frame.full,
      });
      if (!response.ok) throw new Error('frame');
      response = await fetch(`/api/start/${id}`, {
        method: 'POST', headers: wiqCsrfHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ fighter_a_box: boxA, fighter_b_box: boxB, focus_fighter: 'A',
          fighter_a_drawn: drawnA, fighter_b_drawn: drawnB,
          both_detected: !drawnA && !drawnB }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.progress_url) throw new Error(data.detail || 'start');
      if (url) URL.revokeObjectURL(url);
      location.assign(data.progress_url);
    } catch (error) {
      // The fighter page can always take it from here.
      say(error.message && !['frame', 'start'].includes(error.message) ? `${error.message} Opening the fighter page…` : 'Opening the fighter page…');
      location.assign(upload.next_url);
    }
  }

  // Called by the upload page when the bytes have all arrived.
  function uploaded(payload) {
    upload = payload;
    if (fallback) { fallback.querySelector('a').href = payload.next_url; fallback.hidden = false; }
    title.textContent = 'Upload done';
    if (done()) start(); else draw();
  }

  // state() is read by the browser test (tests/test_early_pick.py), nothing else.
  const state = () => ({ frames: frames.map((f) => ({ time: f.time, width: f.width, height: f.height, people: f.people, score: f.score })), shown, boxA, boxB, upload: Boolean(upload) });
  window.wiqEarlyPick = { begin, uploaded, wants, state };
})();
