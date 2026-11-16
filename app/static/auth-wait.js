// The page waiting for an email link to be used, perhaps on another device
// (owner, 2026-11-16: "if I accept it on my phone, I want to continue on my PC
// where I left off"). Asks the server every few seconds, and at once when the
// tab comes back into view; when the link has been used, goes on.
(() => {
  const box = document.querySelector('[data-auth-wait]');
  if (!box) return;
  const started = Date.now();
  let busy = false;
  let timer = null;
  async function check() {
    if (busy || Date.now() - started > 60 * 60 * 1000) return;
    busy = true;
    try {
      const response = await fetch('/auth/wait', {credentials: 'same-origin', cache: 'no-store'});
      if (response.ok) {
        const state = await response.json();
        if (state.state === 'done' && state.go) {
          box.textContent = box.dataset.doneText || 'Done. Taking you back...';
          window.location.assign(state.go);
          return;
        }
        if (state.state === 'none') return;      // nothing to wait for: stop asking
      }
    } catch (error) { /* offline for a moment: try again */ }
    finally { busy = false; }
    timer = setTimeout(check, 3000);
  }
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') { clearTimeout(timer); check(); }
  });
  timer = setTimeout(check, 3000);
})();
