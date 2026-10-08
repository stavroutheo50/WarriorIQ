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
    var ctx = canvas.getContext("2d"), me = card.fighters[side] || { strikes: null, total: null, movement: [] };
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
    // One word, as the site header spells it (QA, 2026-10-07: "WARRIOR IQ").
    var wordmark = ctx.measureText("WARRIOR").width;
    ctx.fillStyle = CYAN; ctx.fillText("IQ", x + wordmark, y);

    y += 90;
    ctx.font = font(600, 36); ctx.fillStyle = MUTED;
    ctx.fillText(String(card.sport || "Fight").toUpperCase() + "  ·  FIGHT ANALYSIS", x, y);

    if (me.total !== null && me.total !== undefined && me.strikes) {
      y += 330;
      ctx.fillStyle = INK; ctx.font = font(800, 300); ctx.fillText(String(me.total), x - 12, y);
      y += 80;
      ctx.font = font(600, 48); ctx.fillStyle = MUTED; ctx.fillText("strikes thrown", x, y);

      y += 110;
      var families = ["punch", "kick", "knee"].filter(function (f) { return f in me.strikes; }), top = 1;
      families.forEach(function (f) { top = Math.max(top, me.strikes[f]); });
      families.forEach(function (f) {
        ctx.font = font(700, 44); ctx.fillStyle = INK; ctx.fillText(PLURAL[f] || f, x, y);
        ctx.textAlign = "right"; ctx.fillText(String(me.strikes[f]), W - x, y); ctx.textAlign = "left";
        roundRect(ctx, x, y + 26, W - 2 * x, 18, 9); ctx.fillStyle = LINE; ctx.fill();
        roundRect(ctx, x, y + 26, Math.max(18, (W - 2 * x) * me.strikes[f] / top), 18, 9);
        ctx.fillStyle = CYAN; ctx.fill();
        y += 128;
      });
    } else {
      // Strike counts are switched off: the card carries what was measured
      // from movement instead, each row on its own 0-100 scale.
      y += 140;
      (me.movement || []).forEach(function (row) {
        ctx.font = font(700, 44); ctx.fillStyle = INK; ctx.fillText(row.label, x, y);
        ctx.textAlign = "right"; ctx.fillText(String(row.value) + (row.unit || ""), W - x, y); ctx.textAlign = "left";
        roundRect(ctx, x, y + 26, W - 2 * x, 18, 9); ctx.fillStyle = LINE; ctx.fill();
        roundRect(ctx, x, y + 26, Math.max(18, (W - 2 * x) * row.value / 100), 18, 9);
        ctx.fillStyle = CYAN; ctx.fill();
        y += 128;
      });
    }

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
    var box = dialog.querySelector("[data-story-job]");
    var link = box ? initLink(box, function () { return side; }) : null;
    dialog.querySelectorAll("input[name=shareSide]").forEach(function (input) {
      input.checked = input.value === side;
      input.addEventListener("change", function () { side = input.value; render(); if (link) link.sideChanged(); });
    });
    open.addEventListener("click", function () {
      render();
      dialog.showModal();
      // showModal focuses the first control, which sits below the card, so
      // the dialog opened scrolled ~400px down with its top cut off on a
      // phone (QA, 2026-10-04). Start at the top, on the title.
      var title = document.getElementById("shareCardTitle");
      if (title) { title.setAttribute("tabindex", "-1"); title.focus({ preventScroll: true }); }
      dialog.scrollTop = 0;
    });
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
