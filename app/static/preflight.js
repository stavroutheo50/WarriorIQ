/* Measure a fight video in the browser, before it is uploaded.
 *
 * The landing page promises "a quick quality check catches hard-to-see footage
 * before the analysis starts". The check only ever read the file's size and
 * duration; videoWidth was never touched. So the one thing that decides
 * whether an analysis can work - how big the fighters are in the picture -
 * was first measured on the worker, after the filmer had already waited out
 * an upload they could not have known was pointless.
 *
 * What this can and cannot do
 * ---------------------------
 * It cannot run the pose model, so it cannot find a person. What it can do is
 * find the part of the picture that MOVES, which in a fight video is the
 * fighters. Frame differencing across a handful of decoded frames gives the
 * height of the moving region as a share of the frame, and a share is exactly
 * the quantity that matters: what starves the detector is not a fighter's
 * height in the source, it is their height once the frame has been scaled into
 * the network's input.
 *
 *     a fighter 90 px tall, at different source widths
 *        640 -> 198 px in the network    fine
 *       1280 -> 144 px                   drops people
 *       1920 ->  96 px                   below the floor
 *       3840 ->  48 px                   hopeless
 *
 * The same 90 px gets worse as the source gets better, because the downscale
 * to a clamped inference size shrinks them further. Measuring pixels alone
 * would therefore give the wrong answer on exactly the phone footage this is
 * for.
 *
 * Where it refuses to guess
 * -------------------------
 * A panning camera moves every pixel, and a moving crowd moves most of them.
 * The height of that region is not a fighter's height, so when the moving
 * region covers too much - or too little - of the frame, this reports that it
 * could not measure rather than returning a number it would have to invent.
 * Resolution and duration are still exact and are still reported.
 *
 * Nothing here blocks an upload. It is a warning, shown while the filmer can
 * still walk closer and film it again.
 */
(function (global) {
  'use strict';

  // How many frames to decode. Eight seeks is about a second of work on a
  // phone and gives seven difference pairs to take a median over.
  var SAMPLES = 8;
  /* Work at this width.
   *
   * 192 was too coarse for the case this check exists for. A fighter 8% of
   * the height of a 1080p frame lands about three pixels wide at 192, which
   * is under any sane row threshold, so the far-away clip - the one that most
   * needs flagging - measured as "nothing is moving". At 320 the same fighter
   * is five pixels across and registers. Seven difference passes over
   * 320x180 is about 400k comparisons, which is still nothing.
   */
  var WORK_WIDTH = 320;
  // Per-channel difference that counts as movement rather than sensor noise.
  var MOVED = 18;
  /* A row counts as part of the subject when this share of it moved.
   *
   * Deliberately low. This is a share of the frame's width, and a distant
   * fighter is a very small share of it - setting this by what looks like a
   * sensible fraction of a frame quietly excludes every subject the check is
   * meant to catch. Noise is held off by MOVED above rather than by this.
   */
  var ROW_SHARE = 0.015;
  /* Below this share of differing pixels, two samples are the same decoded
   * frame rather than a still moment.
   *
   * Seeking lands on the nearest keyframe, and phone video is written with
   * sparse ones - so several of the sampled times decode to one frame. On a
   * synthetic clip of a known subject, five of seven pairs came back
   * identical; the two real pairs measured the subject exactly right, and the
   * median over all seven was zero, which reads as "nothing in this video
   * moves". A duplicate pair is an absence of evidence, so it is dropped
   * before the median rather than counted as a zero.
   */
  var IDENTICAL = 0.0005;
  /* How solidly the moving region must fill its own bounding box before its
   * height is treated as a fighter's height.
   *
   * A fighter sweeping against a still background fills a good share of the
   * box that contains them. A panned background or a restless crowd scatters
   * thin changes across most of the frame, which produces a tall box that no
   * single thing occupies. Measured on a synthetic pan the density was 0.07
   * against 0.35-0.6 for the subject clips.
   */
  var MIN_DENSITY = 0.18;

  function median(values) {
    if (!values.length) return 0;
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  // Decode one frame at `time` into the canvas and hand back its pixels.
  function grab(video, canvas, context, time) {
    return new Promise(function (resolve, reject) {
      var done = false;
      var finish = function (ok) {
        if (done) return;
        done = true;
        video.removeEventListener('seeked', onSeeked);
        video.removeEventListener('error', onError);
        ok ? resolve(context.getImageData(0, 0, canvas.width, canvas.height))
           : reject(new Error('seek failed'));
      };
      var onSeeked = function () {
        try {
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          finish(true);
        } catch (err) { finish(false); }
      };
      var onError = function () { finish(false); };
      video.addEventListener('seeked', onSeeked);
      video.addEventListener('error', onError);
      // A seek that never lands must not hang the page behind it.
      setTimeout(function () { finish(false); }, 4000);
      try { video.currentTime = time; } catch (err) { finish(false); }
    });
  }

  /* What moved between two frames: its vertical extent, and how solidly it
   * fills the box it occupies.
   *
   * Density is what separates a fighter from a moving camera, and the share of
   * changed pixels on its own is not. Panning across a scene whose background
   * is mostly flat changes only the textured parts of it, so the share stays
   * low while the changed pixels are scattered over the entire frame. Measured
   * on a synthetic pan, that read as a subject 99% of the frame tall - a
   * confident wrong answer of exactly the kind this check exists to avoid.
   *
   * A fighter moving against a still background fills its own bounding box;
   * a panned background, or a moving crowd, scatters thinly across a much
   * larger one. So the box is measured in both directions and the fill is
   * compared against it.
   */
  function motionBetween(before, after, width, height) {
    var rows = new Uint32Array(height);
    var columns = new Uint32Array(width);
    var moved = 0;
    var a = before.data, b = after.data;
    for (var y = 0; y < height; y++) {
      var count = 0;
      for (var x = 0; x < width; x++) {
        var i = (y * width + x) * 4;
        // Luma difference: a colour cast between frames should not read as
        // movement, and the green channel carries most of the luminance.
        var diff = Math.abs((a[i] * 3 + a[i + 1] * 6 + a[i + 2]) - (b[i] * 3 + b[i + 1] * 6 + b[i + 2])) / 10;
        if (diff > MOVED) { count++; columns[x]++; }
      }
      rows[y] = count;
      moved += count;
    }
    var rowFloor = Math.max(1, Math.round(width * ROW_SHARE));
    var columnFloor = Math.max(1, Math.round(height * ROW_SHARE));
    var top = -1, bottom = -1, left = -1, right = -1;
    for (var r = 0; r < height; r++) {
      if (rows[r] >= rowFloor) { if (top < 0) top = r; bottom = r; }
    }
    for (var c = 0; c < width; c++) {
      if (columns[c] >= columnFloor) { if (left < 0) left = c; right = c; }
    }
    var boxHeight = top < 0 ? 0 : bottom - top + 1;
    var boxWidth = left < 0 ? 0 : right - left + 1;
    var area = boxHeight * boxWidth;
    return {
      share: moved / (width * height),
      extent: boxHeight / height,
      spread: boxWidth / width,
      density: area ? moved / area : 0
    };
  }

  /* A usable duration, even when the container does not declare one.
   *
   * A WebM written by a phone or a re-encoder frequently carries no duration,
   * and the browser then reports Infinity. Read naively that becomes a zero
   * length, every sample seeks to t=0, and eight identical frames measure as
   * "nothing is moving" - a confident wrong answer about exactly the footage
   * this check exists for.
   *
   * Seeking far past the end makes the browser scan to the real end and fill
   * the duration in. The position is put back afterwards so the caller starts
   * from a known place.
   */
  function resolveDuration(video) {
    return new Promise(function (resolve) {
      if (isFinite(video.duration) && video.duration > 0) { resolve(video.duration); return; }
      var done = false;
      var finish = function (value) {
        if (done) return;
        done = true;
        video.removeEventListener('durationchange', onChange);
        try { video.currentTime = 0; } catch (err) { /* already at the start */ }
        resolve(value);
      };
      var onChange = function () {
        if (isFinite(video.duration) && video.duration > 0) finish(video.duration);
      };
      video.addEventListener('durationchange', onChange);
      try { video.currentTime = 1e7; } catch (err) { finish(0); }
      setTimeout(function () {
        finish(isFinite(video.duration) && video.duration > 0 ? video.duration : 0);
      }, 3000);
    });
  }

  // Frames per second, counted rather than read: the container's declared rate
  // is often absent in a browser and wrong in a re-encode. Resolves null when
  // the browser has no per-frame callback, because a guess here would produce
  // a warning about a number nobody measured.
  /* Frame rate read out of an MP4/MOV container, exactly.
   *
   * Counting presented frames is unreliable (see frameRate below), and phone
   * footage is overwhelmingly MP4 or MOV, where the answer is written down:
   * the media header carries a timescale and a duration, and the time-to-
   * sample table carries how many samples there are. Frames per second is
   * then a division rather than an estimate.
   *
   * Only box headers are read, and only from the slices they point at, so a
   * 130 MB file is never pulled into memory. Returns null for anything that
   * is not an ISO base media file, or whose tables are not where they should
   * be - the caller then falls back to counting, or to saying nothing.
   */
  function containerFrameRate(file) {
    var HEADER = 8;
    var read = function (start, end) {
      return file.slice(start, end).arrayBuffer().then(function (buffer) {
        return new DataView(buffer);
      });
    };
    // Walk the boxes at one level, calling `onBox` with (type, start, size).
    var walk = function (from, to, onBox) {
      if (from >= to) return Promise.resolve(null);
      return read(from, Math.min(from + HEADER, to)).then(function (view) {
        if (view.byteLength < HEADER) return null;
        var size = view.getUint32(0);
        var type = String.fromCharCode(view.getUint8(4), view.getUint8(5),
                                       view.getUint8(6), view.getUint8(7));
        // size 1 means a 64-bit length follows; size 0 means "to end of file".
        if (size === 0) size = to - from;
        if (size < HEADER) return null;
        var found = onBox(type, from, size);
        if (found) return found;
        return walk(from + size, to, onBox);
      }).catch(function () { return null; });
    };
    var findChild = function (from, to, wanted) {
      return walk(from, to, function (type, start, size) {
        return type === wanted ? { start: start, size: size } : null;
      });
    };
    // mdhd: version, flags, times, then timescale and duration.
    var readMdhd = function (box) {
      return read(box.start, box.start + Math.min(box.size, 40)).then(function (v) {
        var version = v.getUint8(HEADER);
        return version === 1
          ? { timescale: v.getUint32(HEADER + 20), duration: Number(v.getBigUint64(HEADER + 24)) }
          : { timescale: v.getUint32(HEADER + 12), duration: v.getUint32(HEADER + 16) };
      });
    };
    // stts: an entry count, then (sample count, delta) pairs.
    var readStts = function (box) {
      return read(box.start, box.start + Math.min(box.size, 4096)).then(function (v) {
        var entries = v.getUint32(HEADER + 4);
        var samples = 0;
        for (var i = 0; i < entries && HEADER + 8 + i * 8 + 8 <= v.byteLength; i++) {
          samples += v.getUint32(HEADER + 8 + i * 8);
        }
        return samples;
      });
    };
    return findChild(0, file.size, 'moov').then(function (moov) {
      if (!moov) return null;
      var moovEnd = moov.start + moov.size;
      // The first trak carrying an stts with plausible video timing wins; an
      // audio track has an stts too, which is why the rate is sanity checked.
      var tryTrak = function (from) {
        if (from >= moovEnd) return null;
        return walk(from, moovEnd, function (type, start, size) {
          return type === 'trak' ? { start: start, size: size } : null;
        }).then(function (trak) {
          if (!trak) return null;
          var trakEnd = trak.start + trak.size;
          return findChild(trak.start + HEADER, trakEnd, 'mdia').then(function (mdia) {
            if (!mdia) return tryTrak(trakEnd);
            var mdiaEnd = mdia.start + mdia.size;
            return findChild(mdia.start + HEADER, mdiaEnd, 'mdhd').then(function (mdhd) {
              if (!mdhd) return tryTrak(trakEnd);
              return readMdhd(mdhd).then(function (header) {
                return findChild(mdia.start + HEADER, mdiaEnd, 'minf').then(function (minf) {
                  if (!minf) return tryTrak(trakEnd);
                  return findChild(minf.start + HEADER, minf.start + minf.size, 'stbl');
                }).then(function (stbl) {
                  if (!stbl) return tryTrak(trakEnd);
                  return findChild(stbl.start + HEADER, stbl.start + stbl.size, 'stts');
                }).then(function (stts) {
                  if (!stts) return tryTrak(trakEnd);
                  return readStts(stts).then(function (samples) {
                    var seconds = header.duration / header.timescale;
                    var rate = seconds > 0 ? samples / seconds : 0;
                    // Audio tracks land in the thousands; a still image in
                    // the fractions. Anything outside this is not a frame
                    // rate we should be quoting at somebody.
                    if (rate >= 5 && rate <= 240) return rate;
                    return tryTrak(trakEnd);
                  });
                });
              });
            });
          });
        });
      };
      return tryTrak(moov.start + HEADER);
    }).catch(function () { return null; });
  }

  function frameRate(video) {
    return new Promise(function (resolve) {
      // A hidden tab throttles both playback and the frame callback, so the
      // count would be of the throttling rather than of the video.
      if (typeof video.requestVideoFrameCallback !== 'function'
          || (global.document && global.document.visibilityState === 'hidden')) {
        resolve(null);
        return;
      }
      var first = null, frames = 0, stop = false;
      var wallStart = (global.performance || Date).now();
      var settle = function (value) {
        if (stop) return;
        stop = true;
        try { video.pause(); } catch (err) { /* already stopped */ }
        resolve(value);
      };
      var tick = function (now, meta) {
        if (stop) return;
        if (first === null) first = meta.mediaTime;
        frames++;
        var elapsed = meta.mediaTime - first;
        if (elapsed >= 0.5 && frames > 1) {
          /* Only trust a count that kept up with the clock.
           *
           * requestVideoFrameCallback fires for *presented* frames. An
           * offscreen or throttled video presents a fraction of them, and the
           * count then measures the browser's scheduling rather than the
           * file: a 30 fps clip measured 1 fps this way. Comparing how far
           * the video advanced against how long it actually took separates
           * "this video is 12 fps" from "this tab was not being drawn", and
           * only the first is something to warn a filmer about.
           */
          var wall = ((global.performance || Date).now() - wallStart) / 1000;
          var keptUp = wall > 0 && (elapsed / wall) >= 0.8;
          settle(keptUp ? (frames - 1) / elapsed : null);
          return;
        }
        video.requestVideoFrameCallback(tick);
      };
      video.requestVideoFrameCallback(tick);
      var played = video.play();
      if (played && played.catch) played.catch(function () { settle(null); });
      // Short: an unmeasured frame rate costs one missing sentence, while a
      // long stall here delays every check behind it.
      setTimeout(function () { settle(null); }, 1500);
    });
  }

  /* Measure `file`. Resolves an object; never rejects, because a check that
   * throws would block an upload the server would happily have taken. */
  function measure(file, limits) {
    var url = URL.createObjectURL(file);
    var video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.playsInline = true;
    video.src = url;

    var result = {
      measured: false, width: 0, height: 0, duration: 0, fps: null,
      portrait: false, subjectShare: 0, subjectPx: 0, networkPx: 0,
      framingMeasured: false, reason: '',
      // How much evidence the framing number rests on. A read from one usable
      // pair is not the same fact as a read from seven.
      pairsCompared: 0, pairsDuplicate: 0, density: 0, fpsSource: ''
    };
    var release = function () {
      try { video.removeAttribute('src'); video.load(); } catch (err) { /* nothing to undo */ }
      URL.revokeObjectURL(url);
    };

    return new Promise(function (resolve) {
      var giveUp = setTimeout(function () {
        result.reason = 'timeout';
        release();
        resolve(result);
      }, 20000);

      video.onerror = function () {
        clearTimeout(giveUp);
        result.reason = 'undecodable';
        release();
        resolve(result);
      };

      video.onloadedmetadata = function () {
        result.width = video.videoWidth || 0;
        result.height = video.videoHeight || 0;
        result.portrait = result.height > result.width;
        result.measured = result.width > 0 && result.height > 0;
        if (!result.measured) {
          clearTimeout(giveUp);
          result.reason = 'no dimensions';
          release();
          resolve(result);
          return;
        }

        var aspect = result.height / result.width;
        var canvas = document.createElement('canvas');
        canvas.width = WORK_WIDTH;
        canvas.height = Math.max(2, Math.round(WORK_WIDTH * aspect));
        // willReadFrequently: this reads back every frame it draws, and
        // without the hint some browsers keep the surface on the GPU and pay
        // a full stall per getImageData.
        var context = canvas.getContext('2d', { willReadFrequently: true });

        resolveDuration(video).then(function (span) {
          result.duration = span;
          // The container's own answer first; counting presented frames only
          // when the file does not carry one.
          return containerFrameRate(file).then(function (declared) {
            if (declared) { result.fpsSource = 'container'; return [span, declared]; }
            return frameRate(video).then(function (counted) {
              result.fpsSource = counted ? 'counted' : '';
              return [span, counted];
            });
          });
        }).then(function (pair) {
          var span = pair[0];
          result.fps = pair[1];
          // Sample the middle 80%: the opening and closing seconds of a phone
          // clip are usually the filmer walking up and putting the phone down.
          var from = span * 0.1, to = span * 0.9;
          var times = [];
          for (var i = 0; i < SAMPLES; i++) {
            times.push(span ? from + (to - from) * (i / (SAMPLES - 1 || 1)) : 0);
          }

          var frames = [];
          var next = times.reduce(function (chain, at) {
            return chain.then(function () {
              return grab(video, canvas, context, at)
                .then(function (pixels) { frames.push(pixels); })
                .catch(function () { /* a frame that will not decode is skipped */ });
            });
          }, Promise.resolve());

          next.then(function () {
            clearTimeout(giveUp);
            var extents = [], shares = [], densities = [], duplicates = 0;
            for (var i = 1; i < frames.length; i++) {
              var m = motionBetween(frames[i - 1], frames[i], canvas.width, canvas.height);
              if (m.share < IDENTICAL) { duplicates++; continue; }
              shares.push(m.share);
              extents.push(m.extent);
              densities.push(m.density);
            }
            result.pairsCompared = shares.length;
            result.pairsDuplicate = duplicates;
            var share = median(shares);
            if (span <= 0) {
              // Without a length every sample lands on the same frame, so a
              // zero motion reading here means nothing was sampled - not that
              // nothing moved. Two different facts, and saying the wrong one
              // would send the filmer off to fix a problem they do not have.
              result.reason = 'no duration';
            } else if (frames.length < 2) {
              result.reason = 'no frames decoded';
            } else if (!extents.length) {
              // Every pair decoded to the same frame. Either the clip really
              // is static, or every seek landed on one keyframe; both mean
              // there is nothing here to size a fighter from.
              result.reason = 'nothing moving';
            } else if (median(densities) < MIN_DENSITY) {
              // A tall box nothing actually fills: the camera moved, or the
              // background did. Either way its height is not a fighter.
              result.reason = 'camera moving';
            } else if (share > limits.max_motion_share) {
              // Everything moved, so the moving region is the room, not a
              // fighter. Almost always a panning or handheld camera.
              result.reason = 'camera moving';
            } else if (share < limits.min_motion_share) {
              result.reason = 'nothing moving';
            } else {
              result.density = median(densities);
              result.subjectShare = median(extents);
              result.subjectPx = result.subjectShare * result.height;
              var longEdge = Math.max(result.width, result.height);
              // The same clamp core/preflight_client.py applies, so the
              // browser's number and the worker's agree.
              var wanted = limits.target_subject_px * longEdge / Math.max(1, result.subjectPx);
              var stepped = Math.round(wanted / 32) * 32;
              var size = Math.max(limits.min_inference_size,
                                  Math.min(limits.max_inference_size, stepped));
              result.networkPx = result.subjectPx * size / longEdge;
              result.framingMeasured = true;
            }
            release();
            resolve(result);
          });
        });
      };
    });
  }

  global.wiqPreflight = { measure: measure };
})(window);
