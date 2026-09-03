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

  /* This re-encode runs at PLAYBACK SPEED: preparing a five-minute fight costs
   * five minutes before the upload starts. Whether that is worth paying has
   * nothing to do with how long the fight is - it depends on how much transfer
   * it saves, so that is what gets measured.
   *
   * A capped-by-duration rule was tried and was wrong in the direction that
   * hurts most. It switched the re-encode off for long files, which are
   * exactly the ones that need it: an 881MB 4K bout went up untouched at
   * roughly 0.6MB/s, which is about twenty-five minutes. The same fight
   * shrunk is near 120MB - five minutes of encoding and three and a half of
   * transfer.
   *
   * A conservative mobile uplink, deliberately pessimistic. Guessing too low
   * only declines a re-encode that would have helped a little; guessing too
   * high spends real minutes encoding for a saving the connection did not
   * need. ~6 Mbps. */
  var ASSUMED_UPLINK_BYTES_PER_SECOND = 750 * 1024;
  /* However good the arithmetic looks, nobody is holding a phone still for a
   * quarter of an hour, and a backgrounded tab truncates the recording. */
  var MAX_SECONDS = 900;

  function supported() {
    return typeof window.MediaRecorder === "function" &&
      typeof HTMLCanvasElement.prototype.captureStream === "function" &&
      typeof MediaRecorder.isTypeSupported === "function";
  }

  /* MP4/H.264 only, and null rather than anything else.
   *
   * This used to fall back to VP9 or VP8 WebM, and that is what broke uploads
   * from a phone. Desktop Chrome can record MP4, so a laptop produced a file
   * the server read happily. Android Chrome's MediaRecorder usually cannot, so
   * it fell through to WebM - and the server's OpenCV build reports a WebM as
   * 0x0 pixels, -1 frames and -1 seconds, then either fails to read a frame or
   * blocks trying. The upload was rejected with "could not prepare this video",
   * for a file the phone had recorded perfectly well and WarriorIQ had itself
   * converted into something unreadable.
   *
   * A phone that cannot record MP4 now simply skips the re-encode. The
   * original camera recording is H.264 in an MP4, which the server reads. */
  function pickMime() {
    return MediaRecorder.isTypeSupported("video/mp4;codecs=avc1")
      ? "video/mp4;codecs=avc1"
      : null;
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
      if (video.duration > MAX_SECONDS) return null;

      // Is re-encoding worth the wait on this particular file? The encode runs
      // at playback speed, so it costs about the fight's own length; what it
      // buys is everything it takes off the transfer.
      var estimatedBytes = video.duration * BITRATE / 8;
      var savedSeconds = (file.size - estimatedBytes) / ASSUMED_UPLINK_BYTES_PER_SECOND;
      // A clear win, not a marginal one: the person is staring at a progress
      // bar either way, and only one of the two is visibly moving their file.
      if (savedSeconds < video.duration * 1.25) return null;

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

      // Did the playback actually finish? If the phone locked or the browser
      // was backgrounded, playback and rAF both stopped, the timeout above
      // fired, and these chunks are a fragment of the fight. Uploading that
      // would silently analyse the first minute of someone's bout and report
      // it as the whole thing, which is far worse than a slower upload.
      if (video.duration && video.currentTime < video.duration * 0.98) return null;

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
