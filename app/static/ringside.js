// The free Ringside Scorer (app/templates/ringside.html). This file records
// taps and draws the page; every score comes from the server
// (/api/ringside/score, core/ringside.py), which uses the report's scorer.
(function () {
  'use strict';
  var SPORTS = window.WIQ_SCORER || [];
  // v2: rounds hold points and deductions per corner (owner, 2026-11-09), not
  // the technique taps v1 stored, so a v1 fight in this browser is not reused.
  var KEY = 'wiq-ringside-v2';
  var $ = function (id) { return document.getElementById(id); };
  var state = load() || fresh();
  var pending = null, lastScore = null;

  function fresh() {
    return { sport: SPORTS[0] && SPORTS[0].key, ruleset: null, names: { A: '', B: '' },
             rounds: [blankRound()], current: 0, history: [], view: 'setup' };
  }
  function blankRound() { return { A: { points: 0, deductions: 0 }, B: { points: 0, deductions: 0 } }; }
  function load() { try { return JSON.parse(localStorage.getItem(KEY) || 'null'); } catch (e) { return null; } }
  function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* private mode */ } }
  function sport() { return SPORTS.find(function (s) { return s.key === state.sport; }) || SPORTS[0]; }
  function rules() {
    var s = sport();
    return s.rulesets.find(function (r) { return r.key === state.ruleset; }) || s.rulesets[0];
  }
  function name(side) { return (state.names[side] || '').trim() || (side === 'A' ? 'Blue' : 'Red'); }

  // ---- setup ---------------------------------------------------------------
  function drawSetup() {
    document.querySelectorAll('[data-sport]').forEach(function (chip) {
      chip.setAttribute('aria-checked', String(chip.dataset.sport === state.sport));
    });
    var s = sport(), select = $('scorerRuleset');
    select.innerHTML = '';
    s.rulesets.forEach(function (r) {
      var option = document.createElement('option');
      option.value = r.key; option.textContent = r.label; select.appendChild(option);
    });
    if (!s.rulesets.some(function (r) { return r.key === state.ruleset; })) state.ruleset = s.rulesets[0].key;
    select.value = state.ruleset;
    $('scorerRulesetField').hidden = s.rulesets.length < 2;
    $('scorerSummary').textContent = rules().summary;
    $('nameA').value = state.names.A; $('nameB').value = state.names.B;
  }
  document.querySelectorAll('[data-sport]').forEach(function (chip) {
    chip.addEventListener('click', function () { state.sport = chip.dataset.sport; state.ruleset = null; save(); drawSetup(); });
  });
  $('scorerRuleset').addEventListener('change', function (e) { state.ruleset = e.target.value; save(); drawSetup(); });
  ['A', 'B'].forEach(function (side) {
    $('name' + side).addEventListener('input', function (e) { state.names[side] = e.target.value; save(); });
  });
  $('scorerStart').addEventListener('click', function () {
    state.rounds = [blankRound()]; state.current = 0; state.history = []; state.view = 'live'; lastScore = null;
    save(); show();
  });

  // ---- scoring -------------------------------------------------------------
  function drawPads() {
    ['A', 'B'].forEach(function (side) {
      var pad = $('pad' + side);
      pad.innerHTML = '';
      var heading = document.createElement('p');
      heading.className = 'ringside-pad-name'; heading.textContent = name(side); pad.appendChild(heading);
      var row = document.createElement('div');
      row.className = 'ringside-plus';
      [1, 2].forEach(function (value) {
        var b = document.createElement('button');
        b.type = 'button'; b.className = 'ringside-tap ringside-point';
        b.textContent = '+' + value;
        b.setAttribute('aria-label', '+' + value + ' for ' + name(side));
        b.addEventListener('click', function () { tap(side, 'points', value, b); });
        row.appendChild(b);
      });
      pad.appendChild(row);
      var minus = document.createElement('button');
      minus.type = 'button'; minus.className = 'ringside-tap ringside-deduction';
      minus.textContent = 'Point deduction';
      minus.setAttribute('aria-label', 'Take a point off ' + name(side));
      minus.addEventListener('click', function () { tap(side, 'deductions', 1, minus); });
      pad.appendChild(minus);
    });
  }
  function tap(side, kind, value, element) {
    var round = state.rounds[state.current];
    round[side][kind] += value;
    state.history.push({ round: state.current, side: side, kind: kind, value: value });
    if (navigator.vibrate) { try { navigator.vibrate(18); } catch (e) { /* not allowed */ } }
    if (element) { element.classList.add('is-hit'); setTimeout(function () { element.classList.remove('is-hit'); }, 120); }
    save(); drawBoard(); requestScore();
  }
  $('scorerUndo').addEventListener('click', function () {
    var last = state.history.pop();
    if (!last) return;
    var corner = state.rounds[last.round][last.side];
    corner[last.kind] = Math.max(0, corner[last.kind] - last.value);
    state.current = last.round;
    save(); drawBoard(); requestScore();
  });
  $('scorerNextRound').addEventListener('click', function () {
    if (state.rounds.length >= 15) return;
    state.rounds.push(blankRound()); state.current = state.rounds.length - 1;
    save(); drawBoard(); requestScore();
  });
  $('scorerFinish').addEventListener('click', function () { state.view = 'result'; save(); show(); requestScore(); });
  $('scorerBack').addEventListener('click', function () { state.view = 'live'; save(); show(); });
  $('scorerNew').addEventListener('click', function () { state.view = 'setup'; save(); show(); });

  function tapped(side) {
    return state.rounds.reduce(function (n, round) { return n + round[side].points; }, 0);
  }
  function drawBoard() {
    $('boardNameA').textContent = name('A'); $('boardNameB').textContent = name('B');
    $('boardRound').textContent = 'Round ' + (state.current + 1);
    $('boardRules').textContent = rules().label;
    ['A', 'B'].forEach(function (side) {
      var n = tapped(side);
      $('boardLanded' + side).textContent = n + (n === 1 ? ' point' : ' points') + ' tapped';
      $('boardScore' + side).textContent = lastScore ? lastScore.totals[side] : '–';
    });
  }

  function requestScore() {
    clearTimeout(pending);
    pending = setTimeout(function () {
      fetch('/api/ringside/score', {
        method: 'POST', credentials: 'same-origin',
        headers: window.wiqCsrfHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ ruleset: rules().key, rounds: state.rounds })
      }).then(function (response) {
        if (!response.ok) throw new Error('score ' + response.status);
        return response.json();
      }).then(function (data) {
        lastScore = data; $('scorerOffline').hidden = true; drawBoard(); drawResult();
      }).catch(function () { $('scorerOffline').hidden = false; });
    }, 180);
  }

  // ---- result and sharing --------------------------------------------------
  function verdict() {
    if (!lastScore) return '';
    var t = lastScore.totals;
    if (t.A === t.B) return 'Level at ' + t.A + '–' + t.B;
    var w = t.A > t.B ? 'A' : 'B';
    return name(w) + ' wins ' + Math.max(t.A, t.B) + '–' + Math.min(t.A, t.B);
  }
  function drawResult() {
    if (!lastScore) return;
    $('resultRules').textContent = sport().label + ' · ' + rules().label;
    $('resultHeadline').textContent = verdict();
    $('resultNameA').textContent = name('A'); $('resultNameB').textContent = name('B');
    $('resultColA').textContent = name('A'); $('resultColB').textContent = name('B');
    $('resultScore').textContent = lastScore.totals.A + ' – ' + lastScore.totals.B;
    var body = $('resultRounds'); body.innerHTML = '';
    lastScore.rounds.forEach(function (row) {
      var tr = document.createElement('tr');
      [String(row.round), row.A + ' (' + row.points_A + ' tapped' + (row.deductions_A ? ', -' + row.deductions_A : '') + ')',
       row.B + ' (' + row.points_B + ' tapped' + (row.deductions_B ? ', -' + row.deductions_B : '') + ')']
        .forEach(function (text) { var td = document.createElement('td'); td.textContent = text; tr.appendChild(td); });
      body.appendChild(tr);
    });
    $('scorerWhatsApp').href = 'https://wa.me/?text=' + encodeURIComponent(shareText());
  }
  function shareText() {
    return name('A') + ' vs ' + name('B') + ' (' + rules().label + '): ' + verdict() +
      '. Scored with WarriorIQ Ringside Scorer: ' + location.origin + '/scorer';
  }
  function card() {
    var c = document.createElement('canvas'); c.width = 1080; c.height = 1350;
    var g = c.getContext('2d');
    var bg = g.createLinearGradient(0, 0, 0, 1350); bg.addColorStop(0, '#0b1424'); bg.addColorStop(1, '#05080f');
    g.fillStyle = bg; g.fillRect(0, 0, 1080, 1350);
    g.textAlign = 'center';
    g.fillStyle = '#a9b2c3'; g.font = '600 40px system-ui, sans-serif';
    g.fillText(sport().label + ' · ' + rules().label, 540, 150);
    g.font = '800 64px system-ui, sans-serif';
    g.fillStyle = '#8cc2ff'; g.fillText(name('A'), 540, 300);
    g.fillStyle = '#a9b2c3'; g.font = '600 44px system-ui, sans-serif'; g.fillText('vs', 540, 375);
    g.font = '800 64px system-ui, sans-serif'; g.fillStyle = '#ff6470'; g.fillText(name('B'), 540, 460);
    g.fillStyle = '#ffffff'; g.font = '900 200px system-ui, sans-serif';
    g.fillText(lastScore.totals.A + '–' + lastScore.totals.B, 540, 720);
    g.font = '700 50px system-ui, sans-serif'; g.fillText(verdict(), 540, 820);
    g.font = '500 38px system-ui, sans-serif'; g.fillStyle = '#c3cad8';
    lastScore.rounds.slice(0, 6).forEach(function (row, i) {
      g.fillText('Round ' + row.round + ':  ' + row.A + ' – ' + row.B, 540, 920 + i * 56);
    });
    g.fillStyle = '#a9b2c3'; g.font = '600 36px system-ui, sans-serif';
    g.fillText('Scored with WarriorIQ · warrioriq.eu/scorer', 540, 1290);
    return c;
  }
  $('scorerShare').addEventListener('click', function () {
    if (!lastScore) return;
    card().toBlob(function (blob) {
      var file = blob && new File([blob], 'warrioriq-scorecard.png', { type: 'image/png' });
      if (file && navigator.canShare && navigator.canShare({ files: [file] })) {
        navigator.share({ files: [file], text: shareText() }).catch(function () {});
      } else if (navigator.share) {
        navigator.share({ text: shareText() }).catch(function () {});
      } else if (blob) {
        var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'warrioriq-scorecard.png';
        document.body.appendChild(a); a.click(); a.remove();
      }
    }, 'image/png');
  });

  // ---- views ---------------------------------------------------------------
  function show() {
    $('scorerSetup').hidden = state.view !== 'setup';
    $('scorerLive').hidden = state.view !== 'live';
    $('scorerResult').hidden = state.view !== 'result';
    // While scoring, the board sits at the top of the screen, not under the title.
    $('ringside').classList.toggle('is-scoring', state.view === 'live');
    if (state.view === 'setup') drawSetup();
    else { drawPads(); drawBoard(); requestScore(); }
  }
  show();
}());
