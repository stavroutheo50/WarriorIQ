/* Fight Camp story card (1080x1920): rank, level, streak and training results.
 *
 * Drawn in the browser from the numbers the Fight Camp page shows (#campCardData),
 * so nothing is uploaded. No name and no video. On a phone "Share" opens the
 * system share sheet with the image; elsewhere the image is downloaded.
 */
(function () {
  "use strict";
  var W = 1080, H = 1920;
  var FONT = '"InterVar", Inter, "Segoe UI", system-ui, -apple-system, sans-serif';
  var INK = "#f4f7ff", MUTED = "#9aa8c2", CYAN = "#4fd7ff", GOLD = "#ffc44d", GREEN = "#52d6a3";

  function font(weight, size) { return weight + " " + size + "px " + FONT; }

  function draw(canvas, card) {
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#070a16"; ctx.fillRect(0, 0, W, H);
    var glow = ctx.createRadialGradient(180, 260, 0, 180, 260, 950);
    glow.addColorStop(0, "rgba(255,196,77,0.22)"); glow.addColorStop(1, "rgba(255,196,77,0)");
    ctx.fillStyle = glow; ctx.fillRect(0, 0, W, H);
    var x = 96, y = 190;
    ctx.fillStyle = INK; ctx.font = font(800, 52); ctx.fillText("WARRIOR", x, y);
    ctx.fillStyle = CYAN; ctx.fillText("IQ", x + ctx.measureText("WARRIOR").width, y);  // one word, as on the site
    y += 90; ctx.font = font(600, 36); ctx.fillStyle = MUTED; ctx.fillText("FIGHT CAMP", x, y);
    y += 300; ctx.font = font(800, 190); ctx.fillStyle = GOLD; ctx.fillText(String(card.rank || "Rookie"), x - 8, y);
    y += 90; ctx.font = font(600, 50); ctx.fillStyle = INK;
    ctx.fillText("Level " + card.level + "  ·  " + card.points + " points", x, y);
    y += 140;
    function plural(n, one, many) { return (n === 1 ? one : many); }
    [[plural(card.streak_weeks, "week in a row training", "weeks in a row training"), card.streak_weeks, CYAN],
     [plural(card.sessions, "training session counted", "training sessions counted"), card.sessions, CYAN],
     ["this week, of " + card.week_target, card.week_sessions, GREEN],
     [plural(card.improved, "mission a fight proved improved", "missions a fight proved improved"), card.improved, GREEN]].forEach(function (row) {
      ctx.font = font(800, 120); ctx.fillStyle = row[2]; ctx.fillText(String(row[1] || 0), x, y + 100);
      var w = ctx.measureText(String(row[1] || 0)).width;
      ctx.font = font(600, 40); ctx.fillStyle = MUTED; ctx.fillText(row[0], x + w + 32, y + 92);
      y += 190;
    });
    ctx.font = font(500, 30); ctx.fillStyle = MUTED;
    ctx.fillText("Points from uploaded training and fights WarriorIQ measured.", x, H - 190);
    ctx.font = font(700, 38); ctx.fillStyle = INK; ctx.fillText("warrioriq.eu", x, H - 96);
  }

  function init() {
    var data = document.getElementById("campCardData"), dialog = document.getElementById("campShareDialog");
    var open = document.querySelector("[data-camp-share-open]");
    if (!data || !dialog || !open) return;
    var card = JSON.parse(data.textContent), canvas = dialog.querySelector("canvas");
    open.addEventListener("click", function () { draw(canvas, card); dialog.showModal(); });
    dialog.querySelector("[data-camp-share-close]").addEventListener("click", function () { dialog.close(); });
    dialog.querySelector("[data-camp-share-send]").addEventListener("click", function () {
      canvas.toBlob(async function (blob) {
        var file = new File([blob], "warrioriq-camp.png", { type: "image/png" });
        if (navigator.canShare && navigator.canShare({ files: [file] })) {
          try { await navigator.share({ files: [file], title: "My Fight Camp on WarriorIQ" }); }
          catch (err) { if (err && err.name !== "AbortError" && window.wiqToast) window.wiqToast("Could not open sharing. Download the image instead.", "error"); }
          return;
        }
        var url = URL.createObjectURL(blob), a = document.createElement("a");
        a.href = url; a.download = "warrioriq-camp.png"; document.body.appendChild(a); a.click(); a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
        if (window.wiqToast) window.wiqToast("Image saved. Add it to your story from your photos.", "success");
      }, "image/png");
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
