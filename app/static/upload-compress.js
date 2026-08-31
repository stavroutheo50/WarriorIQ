/* WarriorIQ — shrink oversized fight video in the browser before uploading.
 *
 * A phone films at 1080p or 4K. The analysis stack runs pose inference at 1280
 * and samples at most 15 frames a second, so everything above that is data the
 * uploader waits on and the analysis never uses. On a mobile uplink that is the
 * difference between forty seconds and six minutes.
 *
 * The rule is deliberately conservative: only re-encode when the source is
 * LARGER than the analyser's own inference size, and target exactly that size.
 * Footage at or below 1280 is passed through untouched, because core/config.py
 * treats a small source as low-resolution and infers it at 1280 — shrinking
 * those would genuinely cost accuracy.
 *
 * Audio is dropped. Nothing in the analysis reads it.
 *
 * Every failure path returns null, and the caller uploads the original file.
 * A visitor must never lose an upload because an optimisation misfired.
 */
(function () {
  "use strict";

  var TARGET_LONG_EDGE = 1280;
  var MIN_BYTES = 60 * 1024 * 1024;   // below this the wait is not worth a re-encode
  var BITRATE = 3200000;              // ample for 720p-class fight footage

  function supported() {
    return typeof window.MediaRecorder === "function" &&
      typeof HTMLCanvasElement.prototype.captureStream === "function" &&
      typeof MediaRecorder.isTypeSupported === "function";
  }

  function pickMime() {
    var types = [
      "video/mp4;codecs=avc1",
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm"
    ];
    for (var i = 0; i < types.length; i++) {
      if (MediaRecorder.isTypeSupported(types[i])) return types[i];
    }
    return null;
  }

  function loadMetadata(video, url) {
    return new Promise(function (resolve, reject) {
      var settled = false;
      var done = function (ok) { if (!settled) { settled = true; ok ? resolve() : reject(new Error("metadata")); } };
      video.onloadedmetadata = function () { done(true); };
      video.onerror = function () { done(false); };
      setTimeout(function () { done(false); }, 15000);
      video.src = url;
    });
  }

  /**
   * @returns {Promise<File|null>} a smaller file, or null to upload the original.
   */
  window.wiqShrinkVideo = async function (file, onProgress) {
    if (!supported() || !file || file.size < MIN_BYTES) return null;

    var mime = pickMime();
    if (!mime) return null;

    var url = URL.createObjectURL(file);
    var video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";

    try {
      await loadMetadata(video, url);

      var srcW = video.videoWidth, srcH = video.videoHeight;
      var longEdge = Math.max(srcW, srcH);
      // Already at or below what the analyser wants: leave it completely alone.
      if (!longEdge || longEdge <= TARGET_LONG_EDGE) return null;
      if (!isFinite(video.duration) || video.duration <= 0) return null;

      var scale = TARGET_LONG_EDGE / longEdge;
      var canvas = document.createElement("canvas");
      canvas.width = Math.max(2, Math.round(srcW * scale / 2) * 2);
      canvas.height = Math.max(2, Math.round(srcH * scale / 2) * 2);
      var ctx = canvas.getContext("2d", { alpha: false });
      if (!ctx) return null;

      var stream = canvas.captureStream(30);
      var recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: BITRATE });
      var chunks = [];
      recorder.ondataavailable = function (event) { if (event.data && event.data.size) chunks.push(event.data); };

      var stopped = new Promise(function (resolve) { recorder.onstop = resolve; });
      recorder.start(1000);

      var drawing = true;
      var draw = function () {
        if (!drawing) return;
        try { ctx.drawImage(video, 0, 0, canvas.width, canvas.height); } catch (e) { /* transient */ }
        if (onProgress && video.duration) onProgress(Math.min(0.99, video.currentTime / video.duration));
        requestAnimationFrame(draw);
      };

      await video.play();
      draw();

      await new Promise(function (resolve) {
        video.onended = resolve;
        // A stall must not hang the upload for ever; 3x realtime is generous.
        setTimeout(resolve, Math.min(30 * 60 * 1000, video.duration * 3000 + 20000));
      });

      drawing = false;
      if (recorder.state !== "inactive") recorder.stop();
      await stopped;

      if (!chunks.length) return null;
      var blob = new Blob(chunks, { type: mime.split(";")[0] });

      // If the re-encode did not actually save meaningful bandwidth, the
      // original is the safer thing to send: it is the untouched recording.
      if (blob.size >= file.size * 0.85) return null;

      var ext = mime.indexOf("mp4") !== -1 ? ".mp4" : ".webm";
      var base = (file.name || "fight").replace(/\.[^.]+$/, "");
      return new File([blob], base + ext, { type: blob.type, lastModified: Date.now() });
    } catch (error) {
      return null;
    } finally {
      drawing = false;
      try { video.pause(); } catch (e) { /* ignore */ }
      URL.revokeObjectURL(url);
    }
  };
})();
