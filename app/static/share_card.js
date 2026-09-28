/* Stats-only story card (1080x1920) for Instagram, TikTok, WhatsApp and the rest.
 *
 * Drawn in the browser from the numbers the result page already shows
 * (core.report.share_card), so nothing is uploaded and no link is created.
 * No names and no video frames. On a phone "Share" opens the system share
 * sheet with the image; elsewhere the image is downloaded.
 */
(function () {
  "use strict";
  var W = 1080, H = 1920;
  var FONT = '"InterVar", Inter, "Segoe UI", system-ui, -apple-system, sans-serif';
  var INK = "#f4f7ff", MUTED = "#9aa8c2", CYAN = "#4fd7ff", RED = "#ff718f", LINE = "#243653";
  var PLURAL = { punch: "Punches", kick: "Kicks", knee: "Knees" };

  function font(weight, size) { return weight + " " + size + "px " + FONT; }

  function wrap(ctx, text, x, y, maxWidth, lineHeight, maxLines) {
    var words = String(text || "").split(/\s+/), line = "", lines = 0;
    for (var i = 0; i < words.length; i++) {
      var next = line ? line + " " + words[i] : words[i];
      if (ctx.measureText(next).width > maxWidth && line) {
        if (lines === maxLines - 1) { ctx.fillText(line + "…", x, y); return y + lineHeight; }
        ctx.fillText(line, x, y); y += lineHeight; lines++; line = words[i];
      } else { line = next; }
    }
    if (line) { ctx.fillText(line, x, y); y += lineHeight; }
    return y;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  function draw(canvas, card, side) {
    var ctx = canvas.getContext("2d"), me = card.fighters[side] || { strikes: {}, total: 0 };
    var other = side === "A" ? "B" : "A";
    ctx.fillStyle = "#070a16"; ctx.fillRect(0, 0, W, H);
    var glow = ctx.createRadialGradient(160, 220, 0, 160, 220, 900);
    glow.addColorStop(0, "rgba(79,215,255,0.28)"); glow.addColorStop(1, "rgba(79,215,255,0)");
    ctx.fillStyle = glow; ctx.fillRect(0, 0, W, H);
    glow = ctx.createRadialGradient(W - 120, H - 260, 0, W - 120, H - 260, 900);
    glow.addColorStop(0, "rgba(255,113,143,0.22)"); glow.addColorStop(1, "rgba(255,113,143,0)");
    ctx.fillStyle = glow; ctx.fillRect(0, 0, W, H);

    var x = 96, y = 190;
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = INK; ctx.font = font(800, 52); ctx.fillText("WARRIOR", x, y);
    var wordmark = ctx.measureText("WARRIOR ").width;
    ctx.fillStyle = CYAN; ctx.fillText("IQ", x + wordmark, y);

    y += 90;
    ctx.font = font(600, 36); ctx.fillStyle = MUTED;
    ctx.fillText(String(card.sport || "Fight").toUpperCase() + "  ·  FIGHT ANALYSIS", x, y);

    y += 330;
    ctx.fillStyle = INK; ctx.font = font(800, 300); ctx.fillText(String(me.total), x - 12, y);
    y += 80;
    ctx.font = font(600, 48); ctx.fillStyle = MUTED; ctx.fillText("strikes thrown", x, y);

    y += 110;
    var families = ["punch", "kick", "knee"].filter(function (f) { return f in (me.strikes || {}); }), top = 1;
    families.forEach(function (f) { top = Math.max(top, me.strikes[f]); });
    families.forEach(function (f) {
      ctx.font = font(700, 44); ctx.fillStyle = INK; ctx.fillText(PLURAL[f] || f, x, y);
      ctx.textAlign = "right"; ctx.fillText(String(me.strikes[f]), W - x, y); ctx.textAlign = "left";
      roundRect(ctx, x, y + 26, W - 2 * x, 18, 9); ctx.fillStyle = LINE; ctx.fill();
      roundRect(ctx, x, y + 26, Math.max(18, (W - 2 * x) * me.strikes[f] / top), 18, 9);
      ctx.fillStyle = CYAN; ctx.fill();
      y += 128;
    });

    if (card.score) {
      y += 10;
      ctx.font = font(600, 36); ctx.fillStyle = MUTED; ctx.fillText("Estimated score", x, y);
      y += 110;
      ctx.font = font(800, 110); ctx.fillStyle = INK;
      ctx.fillText(card.score[side] + " – " + card.score[other], x, y);
      y += 60;
    }

    [["Strength", me.strength, CYAN], ["Working on", me.working_on, RED]].forEach(function (row) {
      if (!row[1] || y > H - 420) return;
      y += 40;
      roundRect(ctx, x, y, W - 2 * x, 190, 28); ctx.fillStyle = "rgba(255,255,255,0.05)"; ctx.fill();
      ctx.fillStyle = row[2]; ctx.fillRect(x, y + 34, 8, 122);
      ctx.font = font(600, 32); ctx.fillStyle = MUTED; ctx.fillText(row[0], x + 44, y + 70);
      ctx.font = font(700, 42); ctx.fillStyle = INK; wrap(ctx, row[1], x + 44, y + 130, W - 2 * x - 88, 50, 2);
      y += 190;
    });

    ctx.font = font(500, 30); ctx.fillStyle = MUTED;
    wrap(ctx, card.note, x, H - 190, W - 2 * x, 42, 2);
    ctx.font = font(700, 38); ctx.fillStyle = INK; ctx.fillText("warrioriq.eu", x, H - 96);
  }

  function blobOf(canvas) {
    return new Promise(function (resolve) { canvas.toBlob(resolve, "image/png"); });
  }

  function init() {
    var data = document.getElementById("shareCardData"), dialog = document.getElementById("shareCardDialog");
    var open = document.querySelector("[data-share-card-open]");
    if (!data || !dialog || !open) return;
    var card = JSON.parse(data.textContent), canvas = dialog.querySelector("canvas");
    var side = dialog.dataset.side === "B" ? "B" : "A";
    function render() { draw(canvas, card, side); }
    dialog.querySelectorAll("input[name=shareSide]").forEach(function (input) {
      input.checked = input.value === side;
      input.addEventListener("change", function () { side = input.value; render(); });
    });
    open.addEventListener("click", function () { render(); dialog.showModal(); });
    dialog.querySelector("[data-share-card-close]").addEventListener("click", function () { dialog.close(); });
    dialog.querySelector("[data-share-card-send]").addEventListener("click", async function () {
      var blob = await blobOf(canvas);
      var file = new File([blob], "warrioriq-fight.png", { type: "image/png" });
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        try { await navigator.share({ files: [file], title: "My fight on WarriorIQ" }); }
        catch (err) { if (err && err.name !== "AbortError" && window.wiqToast) window.wiqToast("Could not open sharing. Download the image instead.", "error"); }
        return;
      }
      var url = URL.createObjectURL(blob), a = document.createElement("a");
      a.href = url; a.download = "warrioriq-fight.png"; document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
      if (window.wiqToast) window.wiqToast("Image saved. Add it to your story from your photos.", "success");
    });
  }

  window.wiqShareCard = { draw: draw };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
