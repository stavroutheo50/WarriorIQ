/* The tracking clip on the homepage and the sample report (owner, 2026-11-01).
 *
 * It plays by itself and cannot be paused or controlled: no controls and no
 * taps on it (glass.css), and anything that stops it - the tab coming back,
 * a phone holding autoplay back until the page is touched - starts it again.
 * Only a reduced-motion setting on the device keeps it still, on its poster.
 */
(() => {
  document.querySelectorAll('video[data-tracking-clip]').forEach(clip => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { clip.removeAttribute('autoplay'); clip.pause(); return; }
    const play = () => { if (clip.paused) clip.play().catch(() => {}); };
    clip.addEventListener('pause', () => setTimeout(play, 50));
    document.addEventListener('visibilitychange', () => { if (!document.hidden) play(); });
    // A phone in low-power mode refuses autoplay until the page is touched.
    ['touchstart', 'pointerdown', 'scroll'].forEach(type => window.addEventListener(type, play, { passive: true, once: true }));
    play();
  });
})();
