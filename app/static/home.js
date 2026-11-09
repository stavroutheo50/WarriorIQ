/* The home page's motion (owner, 2026-11-09; app/templates/index.html).
 *
 * Sections fade up as they come into view, and each section clip plays only
 * while it is on screen, so a page of videos never runs all at once. Nothing
 * is hidden until this script runs: without it every section shows as it is.
 * A reduced-motion setting keeps everything still on its poster.
 */
(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const items = [...document.querySelectorAll('.hx-reveal')];
  const clips = [...document.querySelectorAll('video[data-home-clip]')];
  if (reduced || !('IntersectionObserver' in window)) {
    items.forEach(item => item.classList.add('is-in'));
    if (!reduced) clips.forEach(clip => clip.play().catch(() => {}));
    return;
  }
  document.documentElement.classList.add('hx-motion');
  const reveal = new IntersectionObserver(entries => entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add('is-in');
    reveal.unobserve(entry.target);
  }), { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
  items.forEach(item => {
    // Anything already on screen or scrolled past shows at once.
    if (item.getBoundingClientRect().top < window.innerHeight) item.classList.add('is-in');
    else reveal.observe(item);
  });
  const play = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting) entry.target.play().catch(() => {});
    else entry.target.pause();
  }), { threshold: 0.15 });
  clips.forEach(clip => play.observe(clip));
})();
