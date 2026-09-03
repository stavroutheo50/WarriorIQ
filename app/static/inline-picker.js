/* WarriorIQ — choose the moment and the two fighters before anything uploads.
 *
 * The old order was: wait for the upload, then pick a frame, then pick the
 * fighters, then wait again. On a phone the first wait is minutes - an 881MB
 * 4K bout is about five minutes of re-encoding and three and a half of
 * transfer - and it is dead time, after which the person is asked to do the
 * only part that actually needs them.
 *
 * The browser already holds the file, so none of that waiting is necessary to
 * make the choices. This runs the whole selection against the local copy
 * first, then re-encodes and uploads with nobody waiting on the result: the
 * analysis starts on its own when the bytes land.
 *
 * The frame is chosen the same way the server used to choose it - the earliest
 * moment in the opening where the middle of the mat is moving - because the
 * first frame of a round is the referee standing between the fighters with
 * both arms out, and a box drawn there lands on the referee.
 *
 * Boxes are reported as fractions of the frame, never pixels. The video being
 * drawn on is the phone's 4K original; the video being analysed is a 1280
 * re-encode, and pixel coordinates from one are meaningless on the other.
 */
(function () {
  "use strict";

  var SEARCH_SECONDS = 20;      // how far in to look for an opening moment
  var SAMPLES = 10;             // seeks are slow on a phone; keep it modest
  var MIN_BOX = 0.02;           // a box smaller than this is a stray tap

  function el(id) { return document.getElementById(id); }

  window.wiqInlinePicker = function (file, options) {
    var onDone = options.onDone;
    var panel = el("inlinePicker");
    var video = el("pickVideo");
    var canvas = el("pickCanvas");
    var hint = el("pickHint");
    var useFrame = el("pickUseFrame");
    var reset = el("pickReset");
    var start = el("pickStart");
    var stepLabel = el("pickStep");
    if (!panel || !video || !canvas) return false;

    var url = URL.createObjectURL(file);
    var ctx = canvas.getContext("2d");
    var boxes = [null, null];     // fighter A, fighter B, in frame fractions
    var drawing = null;
    var phase = "frame";

    panel.hidden = false;
    video.src = url;
    video.muted = true;
    video.playsInline = true;

    function say(text) { if (hint) hint.textContent = text; }
    function setStep(text) { if (stepLabel) stepLabel.textContent = text; }

    /* ---- step 1: find a moment where the two are actually working ------- */
    function scoreAt(seconds) {
      return new Promise(function (resolve) {
        var settled = false;
        var done = function (value) { if (!settled) { settled = true; resolve(value); } };
        var onSeek = function () {
          video.removeEventListener("seeked", onSeek);
          try {
            var w = 160, h = Math.max(2, Math.round(160 * video.videoHeight / video.videoWidth));
            var probe = document.createElement("canvas");
            probe.width = w; probe.height = h;
            var pctx = probe.getContext("2d", { willReadFrequently: true });
            pctx.drawImage(video, 0, 0, w, h);
            var a = pctx.getImageData(Math.round(w * 0.25), Math.round(h * 0.3),
                                      Math.round(w * 0.5), Math.round(h * 0.45)).data;
            // Compare against the frame a moment later: motion, not brightness.
            video.currentTime = Math.min(video.duration - 0.05, seconds + 0.12);
            video.addEventListener("seeked", function again() {
              video.removeEventListener("seeked", again);
              pctx.drawImage(video, 0, 0, w, h);
              var b = pctx.getImageData(Math.round(w * 0.25), Math.round(h * 0.3),
                                        Math.round(w * 0.5), Math.round(h * 0.45)).data;
              var total = 0;
              for (var i = 0; i < a.length; i += 4) total += Math.abs(a[i] - b[i]);
              done(total / (a.length / 4));
            }, { once: true });
          } catch (error) { done(0); }
        };
        video.addEventListener("seeked", onSeek, { once: true });
        video.currentTime = seconds;
        setTimeout(function () { done(0); }, 4000);
      });
    }

    async function suggestFrame() {
      say("Looking for a clear moment…");
      var limit = Math.min(SEARCH_SECONDS, Math.max(1, video.duration * 0.25));
      var best = 0, bestScore = -1, scores = [];
      for (var i = 0; i < SAMPLES; i++) {
        var t = (limit / SAMPLES) * i + limit / (SAMPLES * 2);
        var score = await scoreAt(t);
        scores.push([t, score]);
        if (score > bestScore) { bestScore = score; best = t; }
      }
      // The earliest clear moment, not the busiest: everything before the
      // chosen frame goes unanalysed.
      for (var j = 0; j < scores.length; j++) {
        if (scores[j][1] >= bestScore * 0.7) { best = scores[j][0]; break; }
      }
      video.currentTime = best;
      say("Scrub if you want a different moment, then continue.");
    }

    /* ---- step 2: draw a box on each fighter ----------------------------- */
    function fit() {
      var rect = video.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      paint();
    }

    function paint() {
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      var colours = ["#ff5470", "#4aa8ff"];
      var names = ["A", "B"];
      for (var i = 0; i < 2; i++) {
        if (boxes[i]) drawBox(boxes[i], colours[i], names[i]);
      }
      if (drawing) drawBox(drawing, colours[boxes[0] ? 1 : 0], boxes[0] ? "B" : "A");
    }

    function drawBox(box, colour, name) {
      var x = box[0] * canvas.width, y = box[1] * canvas.height;
      var w = (box[2] - box[0]) * canvas.width, h = (box[3] - box[1]) * canvas.height;
      ctx.lineWidth = 3;
      ctx.strokeStyle = colour;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = colour;
      ctx.fillRect(x, y - 20, 26, 20);
      ctx.fillStyle = "#04101f";
      ctx.font = "bold 14px system-ui, sans-serif";
      ctx.fillText(name, x + 8, y - 5);
    }

    function pointAt(event) {
      var rect = canvas.getBoundingClientRect();
      return [
        Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
        Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
      ];
    }

    var origin = null;
    canvas.addEventListener("pointerdown", function (event) {
      if (phase !== "boxes") return;
      canvas.setPointerCapture(event.pointerId);
      origin = pointAt(event);
      drawing = [origin[0], origin[1], origin[0], origin[1]];
      event.preventDefault();
    });
    canvas.addEventListener("pointermove", function (event) {
      if (phase !== "boxes" || !origin) return;
      var here = pointAt(event);
      drawing = [
        Math.min(origin[0], here[0]), Math.min(origin[1], here[1]),
        Math.max(origin[0], here[0]), Math.max(origin[1], here[1]),
      ];
      paint();
      event.preventDefault();
    });
    canvas.addEventListener("pointerup", function (event) {
      if (phase !== "boxes" || !drawing) return;
      var box = drawing;
      drawing = null; origin = null;
      if (box[2] - box[0] < MIN_BOX || box[3] - box[1] < MIN_BOX) { paint(); return; }
      boxes[boxes[0] ? 1 : 0] = box;
      paint();
      updateBoxHint();
      event.preventDefault();
    });

    function updateBoxHint() {
      if (!boxes[0]) { say("Drag a box around the FIRST fighter."); }
      else if (!boxes[1]) { say("Now drag a box around the OTHER fighter."); }
      else { say("Both fighters marked. Start whenever you are ready."); }
      if (start) start.disabled = !(boxes[0] && boxes[1]);
    }

    /* ---- wiring --------------------------------------------------------- */
    video.addEventListener("loadedmetadata", function () {
      fit();
      void suggestFrame();
    }, { once: true });

    if (useFrame) {
      useFrame.onclick = function () {
        video.pause();
        phase = "boxes";
        setStep("Step 2 of 2 · Mark the two fighters");
        video.removeAttribute("controls");
        canvas.classList.add("is-active");
        useFrame.hidden = true;
        if (reset) reset.hidden = false;
        fit();
        updateBoxHint();
      };
    }
    if (reset) {
      reset.onclick = function () { boxes = [null, null]; paint(); updateBoxHint(); };
    }
    if (start) {
      start.onclick = function () {
        if (!boxes[0] || !boxes[1]) return;
        start.disabled = true;
        onDone({
          seconds: video.currentTime,
          fighter_a_box: boxes[0],
          fighter_b_box: boxes[1],
          revoke: function () { try { URL.revokeObjectURL(url); } catch (e) {} },
        });
      };
    }
    window.addEventListener("resize", fit);
    setStep("Step 1 of 2 · Find a clear moment");
    return true;
  };
})();
