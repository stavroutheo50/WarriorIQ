/* Stats-only story card (1080x1920) for Instagram, TikTok, WhatsApp and the rest.
 *
 * Drawn in the browser from the numbers the result page already shows
 * (core.report.share_card), so nothing is uploaded for it. No names and no
 * video frames. On a phone "Share" opens the system share sheet with the
 * image; elsewhere the image is downloaded.
 *
 * The dialog can also make a link: a public page with the same numbers
 * (app.main.story_page), which is what a post needs to be tapped. Only
 * made when asked, one per fighter, and it can be turned off.
 */
(function () {
  "use strict";
  var W = 1080, H = 1920;
  var FONT = '"InterVar", Inter, "Segoe UI", system-ui, -apple-system, sans-serif';
  var INK = "#f4f7ff", MUTED = "#a9b2c3", BLUE = "#8cc2ff", RED = "#ff718f";
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

  // The shield from the site's header, drawn once it has loaded (init()
  // redraws an open card when it arrives).
  var LOGO = new Image();
  LOGO.src = "/static/warrioriq-logo-192.png";

  // A glass panel, as on the site (static/glass.css): an almost clear fill,
  // a bright top edge and faint sides.
  function glass(ctx, x, y, w, h, r) {
    roundRect(ctx, x, y, w, h, r);
    var fill = ctx.createLinearGradient(0, y, 0, y + h);
    fill.addColorStop(0, "rgba(255,255,255,0.10)"); fill.addColorStop(1, "rgba(255,255,255,0.035)");
    ctx.fillStyle = fill; ctx.fill();
    var edge = ctx.createLinearGradient(0, y, 0, y + h);
    edge.addColorStop(0, "rgba(255,255,255,0.42)"); edge.addColorStop(0.18, "rgba(255,255,255,0.10)");
    edge.addColorStop(1, "rgba(255,255,255,0.06)");
    ctx.lineWidth = 2; ctx.strokeStyle = edge; ctx.stroke();
  }

  function bar(ctx, x, y, w, share) {
    roundRect(ctx, x, y, w, 16, 8); ctx.fillStyle = "rgba(255,255,255,0.10)"; ctx.fill();
    roundRect(ctx, x, y, Math.max(16, w * Math.max(0, Math.min(1, share))), 16, 8);
    var g = ctx.createLinearGradient(x, 0, x + w, 0);
    g.addColorStop(0, "rgba(140,194,255,0.65)"); g.addColorStop(1, BLUE);
    ctx.fillStyle = g; ctx.fill();
  }

  function draw(canvas, card, side) {
    var ctx = canvas.getContext("2d"), me = card.fighters[side] || { strikes: null, total: null, movement: [] };
    var other = side === "A" ? "B" : "A", x = 84, w = W - 2 * x;
    ctx.fillStyle = "#0a0f18"; ctx.fillRect(0, 0, W, H);
    [[180, 360, "rgba(64,120,255,0.38)"], [W - 140, H - 520, "rgba(150,92,255,0.30)"], [W - 260, 700, "rgba(56,210,255,0.14)"]]
      .forEach(function (light) {
        var g = ctx.createRadialGradient(light[0], light[1], 0, light[0], light[1], 760);
        g.addColorStop(0, light[2]); g.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
      });

    // Header: the shield and the wordmark, as in the site's bar.
    ctx.textBaseline = "alphabetic";
    if (LOGO.complete && LOGO.naturalWidth) ctx.drawImage(LOGO, x, 72, 104, 104);
    // One word, as the site header spells it (QA, 2026-10-07: "WARRIOR IQ").
    ctx.fillStyle = INK; ctx.font = font(800, 58); ctx.fillText("WARRIOR", x + 128, 144);
    ctx.fillText("IQ", x + 128 + ctx.measureText("WARRIOR").width, 144);
    ctx.font = font(600, 32); ctx.fillStyle = MUTED;
    ctx.fillText(String(card.sport || "Fight").toUpperCase() + "  ·  FIGHT ANALYSIS", x, 246);

    var y = 290;
    var counted = me.total !== null && me.total !== undefined && me.strikes;
    if (counted) {
      var families = ["punch", "kick", "knee"].filter(function (f) { return f in me.strikes; }), top = 1;
      families.forEach(function (f) { top = Math.max(top, me.strikes[f]); });
      var h = 330 + families.length * 96;
      glass(ctx, x, y, w, h, 44);
      ctx.fillStyle = INK; ctx.font = font(800, 230); ctx.fillText(String(me.total), x + 48, y + 236);
      ctx.font = font(600, 44); ctx.fillStyle = MUTED; ctx.fillText("strikes thrown", x + 52, y + 300);
      var row = y + 386;
      families.forEach(function (f) {
        ctx.font = font(700, 40); ctx.fillStyle = INK; ctx.fillText(PLURAL[f] || f, x + 52, row);
        ctx.textAlign = "right"; ctx.fillText(String(me.strikes[f]), x + w - 52, row); ctx.textAlign = "left";
        bar(ctx, x + 52, row + 24, w - 104, me.strikes[f] / top);
        row += 96;
      });
      y += h + 28;
    }

    if (card.score) {
      glass(ctx, x, y, w, 190, 40);
      ctx.font = font(600, 32); ctx.fillStyle = MUTED; ctx.fillText("Estimated score", x + 52, y + 64);
      ctx.font = font(800, 104); ctx.fillStyle = INK;
      ctx.fillText(card.score[side] + " – " + card.score[other], x + 48, y + 162);
      y += 214;
    }

    // What was measured from movement: tiles beside the counts, or the main
    // panel when strike counts are switched off. Each on its own 0-100 scale.
    var moves = (me.movement || []).slice(0, 4);
    if (moves.length) {
      var gap = 20, tw = (w - gap) / 2, th = 150;
      moves.forEach(function (m, i) {
        var tx = x + (i % 2) * (tw + gap), ty = y + Math.floor(i / 2) * (th + gap);
        glass(ctx, tx, ty, tw, th, 36);
        ctx.font = font(600, 28); ctx.fillStyle = MUTED; ctx.fillText(m.label, tx + 36, ty + 50);
        ctx.font = font(800, 64); ctx.fillStyle = INK; ctx.fillText(String(m.value), tx + 34, ty + 114);
        var valueWidth = ctx.measureText(String(m.value)).width;
        ctx.font = font(600, 30); ctx.fillStyle = MUTED; ctx.fillText((m.unit || "").trim(), tx + 40 + valueWidth, ty + 114);
        bar(ctx, tx + 36, ty + 124, tw - 72, m.value / 100);
      });
      y += Math.ceil(moves.length / 2) * (th + gap) + 4;
    }

    [["Strength", me.strength, BLUE], ["Working on", me.working_on, RED]].forEach(function (item) {
      if (!item[1] || y > H - 360) return;
      glass(ctx, x, y, w, 132, 34);
      ctx.fillStyle = item[2]; roundRect(ctx, x + 30, y + 28, 8, 76, 4); ctx.fill();
      ctx.font = font(600, 28); ctx.fillStyle = MUTED; ctx.fillText(item[0], x + 64, y + 54);
      ctx.font = font(700, 38); ctx.fillStyle = INK; wrap(ctx, item[1], x + 64, y + 102, w - 110, 44, 1);
      y += 152;
    });

    ctx.font = font(500, 28); ctx.fillStyle = MUTED;
    wrap(ctx, card.note, x, H - 200, w, 40, 2);
    if (LOGO.complete && LOGO.naturalWidth) ctx.drawImage(LOGO, x, H - 138, 56, 56);
    ctx.font = font(700, 40); ctx.fillStyle = INK; ctx.fillText("warrioriq.eu", x + 72, H - 96);
  }

  function toast(message, kind) { if (window.wiqToast) window.wiqToast(message, kind); }

  function initLink(box, currentSide) {
    var job = box.dataset.storyJob, links = {};
    try {
      JSON.parse(document.getElementById("storyLinks").textContent).forEach(function (link) { links[link.side] = link; });
    } catch (err) { links = {}; }
    var url = box.querySelector("[data-story-url]"), create = box.querySelector("[data-story-create]");
    var share = box.querySelector("[data-story-share]"), revoke = box.querySelector("[data-story-revoke]");
    var showName = box.querySelector("[data-story-show-name]"), name = box.querySelector("[data-story-name]");
    var profile = box.querySelector("[data-story-profile]");

    function show() {
      var link = links[currentSide()];
      url.value = link ? link.url : "";
      create.textContent = link ? "Copy link" : "Create link";
      share.hidden = !(link && navigator.share);
      revoke.hidden = !Object.keys(links).length;
      if (link && link.name) { showName.checked = true; name.value = link.name; }
      if (profile) profile.textContent = link && link.on_profile ? "Remove from my profile" : "Post to my profile";
    }

    async function copy(text) {
      try {
        if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); return true; }
      } catch (err) { /* fall through to selecting it */ }
      url.focus(); url.select();
      return false;
    }

    function post(path, body) {
      return fetch(path, { method: "POST", body: body, credentials: "same-origin",
                           headers: window.wiqCsrfHeaders ? window.wiqCsrfHeaders() : {} });
    }

    // Always asks the server, which hands back the same address for the same
    // fighter: the name on the page follows whatever is ticked now.
    create.addEventListener("click", async function () {
      var side = currentSide(), body = new FormData();
      body.append("side", side);
      body.append("name", showName.checked ? name.value : "");
      create.disabled = true;
      try {
        var response = await post("/story/" + encodeURIComponent(job), body);
        if (!response.ok) throw new Error(String(response.status));
        var made = await response.json();
        links[side] = { side: side, url: made.url, name: showName.checked ? name.value : null,
                        on_profile: !!(links[side] && links[side].on_profile) };
        show();
        toast(await copy(made.url) ? "Link copied. Paste it into your story's Link sticker, or send it."
                                    : "Your link is ready: copy it from the box.", "success");
      } catch (err) {
        toast("Could not make the link. Try again.", "error");
      } finally {
        create.disabled = false;
      }
    });
    share.addEventListener("click", async function () {
      var link = links[currentSide()];
      if (!link) return;
      try { await navigator.share({ title: "My fight on WarriorIQ", url: link.url }); }
      catch (err) { if (err && err.name !== "AbortError") toast("Could not open sharing. Copy the link instead.", "error"); }
    });
    if (profile) profile.addEventListener("click", async function () {
      var side = currentSide(), link = links[side], posting = !(link && link.on_profile), body = new FormData();
      body.append("side", side);
      body.append("name", showName.checked ? name.value : "");
      body.append("posted", posting ? "1" : "0");
      profile.disabled = true;
      try {
        var response = await post("/story/" + encodeURIComponent(job) + "/profile", body);
        if (!response.ok) throw new Error(String(response.status));
        var made = await response.json();
        if (made.url) links[side] = { side: side, url: made.url, name: showName.checked ? name.value : null, on_profile: made.posted };
        show();
        if (!posting) toast("Removed from your profile.", "success");
        else if (made.profile_url) toast("Posted to your profile.", "success");
        else toast("Posted. Choose a username under Profile so your athlete page can show it.", "success");
      } catch (err) {
        toast("Could not update your profile. Try again.", "error");
      } finally {
        profile.disabled = false;
      }
    });
    revoke.addEventListener("click", async function () {
      if (!window.confirm("Turn off your links to this fight? Anyone who opens one will see it was turned off.")) return;
      try {
        var response = await post("/story/" + encodeURIComponent(job) + "/revoke", new FormData());
        if (!response.ok) throw new Error(String(response.status));
        links = {};
        show();
        toast("Links turned off.", "success");
      } catch (err) {
        toast("Could not turn the links off. Try again.", "error");
      }
    });
    show();
    return { sideChanged: show };
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
    // The logo usually arrives after the first draw: put it on the card.
    LOGO.addEventListener("load", function () { if (dialog.open) render(); });
    var box = dialog.querySelector("[data-story-job]");
    var link = box ? initLink(box, function () { return side; }) : null;
    dialog.querySelectorAll("input[name=shareSide]").forEach(function (input) {
      input.checked = input.value === side;
      input.addEventListener("change", function () { side = input.value; render(); if (link) link.sideChanged(); });
    });
    open.addEventListener("click", function () {
      render();
      // The dialog is written inside "See all the details", which is folded
      // shut when "Fix this first" leads the report. Inside a closed fold it
      // has no size, so the modal opened invisible and blocked every button on
      // the page (2026-10-13). It is moved out of the fold before it opens.
      if (dialog.parentElement !== document.body) document.body.appendChild(dialog);
      dialog.classList.remove("wiq-reveal");
      dialog.showModal();
      // showModal focuses the first control, which sits below the card, so
      // the dialog opened scrolled ~400px down with its top cut off on a
      // phone (QA, 2026-10-04). Start at the top, on the title.
      var title = document.getElementById("shareCardTitle");
      if (title) { title.setAttribute("tabindex", "-1"); title.focus({ preventScroll: true }); }
      dialog.scrollTop = 0;
    });
    dialog.querySelectorAll("[data-share-card-close]").forEach(function (button) {
      button.addEventListener("click", function () { dialog.close(); });
    });
    // A tap on the dimmed page around the card closes it too.
    dialog.addEventListener("click", function (event) { if (event.target === dialog) dialog.close(); });
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
      if (window.wiqToast) window.wiqToast("Image saved. Share it from your photos.", "success");
    });
  }

  window.wiqShareCard = { draw: draw };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
