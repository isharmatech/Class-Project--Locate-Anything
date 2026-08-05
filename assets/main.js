(function () {
  // Dark/light theme toggle
  const toggle = document.querySelector('[data-theme-toggle]');
  const root = document.documentElement;
  let theme = matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'dark'; // default dark for this tech aesthetic
  root.setAttribute('data-theme', theme);

  function setIcon() {
    toggle.innerHTML = theme === 'dark'
      ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
      : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
  }
  setIcon();

  if (toggle) {
    toggle.addEventListener('click', () => {
      theme = theme === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', theme);
      toggle.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode');
      setIcon();
    });
  }

  // Sticky header shadow on scroll
  const header = document.getElementById('site-header');
  window.addEventListener('scroll', () => {
    if (header) header.classList.toggle('header--scrolled', window.scrollY > 8);
  }, { passive: true });

  // Reveal-on-scroll
  const revealEls = document.querySelectorAll('.reveal');
  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
  revealEls.forEach((el) => io.observe(el));

  // Benchmark bar animation
  const benchFills = document.querySelectorAll('.bench-fill');
  const benchIo = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const el = entry.target;
        el.style.width = el.dataset.width + '%';
        benchIo.unobserve(el);
      }
    });
  }, { threshold: 0.3 });
  benchFills.forEach((el) => benchIo.observe(el));

  // Scrollspy for TOC
  const tocLinks = document.querySelectorAll('.toc a');
  const sections = Array.from(tocLinks).map((a) => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const spyIo = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const id = '#' + entry.target.id;
      const link = document.querySelector('.toc a[href="' + id + '"]');
      if (!link) return;
      if (entry.isIntersecting) {
        tocLinks.forEach((l) => l.classList.remove('active'));
        link.classList.add('active');
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });
  sections.forEach((s) => spyIo.observe(s));

  // Lazy-load the embedded Gradio iframe only when the "Try It Yourself"
  // section approaches the viewport, then reveal it once loaded. The live
  // share URL comes from assets/config.js (window.DEMO_CONFIG.GRADIO_LIVE_URL).
  // If that URL is empty or the iframe fails to load (the Space is asleep),
  // fall back to the "open the Space" message.
  const gradioEmbed = document.getElementById('gradio-embed');
  const gradioFrame = document.getElementById('gradio-frame');
  const embedLoading = document.getElementById('embed-loading');
  const embedFallback = document.getElementById('embed-fallback');
  const cfg = window.DEMO_CONFIG || {};
  const liveUrl = (cfg.GRADIO_LIVE_URL || '').trim();
  const timeoutMs = cfg.IFRAME_LOAD_TIMEOUT_MS || 20000;

  function showFallback() {
    if (gradioEmbed) gradioEmbed.hidden = true;
    if (embedLoading) embedLoading.hidden = true;
    if (embedFallback) embedFallback.hidden = false;
  }

  if (gradioEmbed && gradioFrame) {
    if (!liveUrl) {
      // No live link configured yet — show the fallback immediately.
      showFallback();
    } else {
      let started = false;
      let loaded = false;
      const startLoading = () => {
        if (started) return;
        started = true;
        gradioFrame.src = liveUrl;
      };
      const embedIo = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            startLoading();
            embedIo.unobserve(entry.target);
          }
        });
      }, { rootMargin: '200px 0px' });
      embedIo.observe(gradioEmbed);

      const hideLoader = () => {
        if (loaded) return;
        loaded = true;
        gradioEmbed.classList.add('is-loaded');
      };
      gradioFrame.addEventListener('load', hideLoader);
      // If the iframe errors or the share link is unreachable, fall back.
      gradioFrame.addEventListener('error', showFallback);
      // Safety net: if the share link hasn't rendered in time, show the fallback
      // so the section is never stuck on a spinner.
      window.setTimeout(() => {
        if (!loaded) showFallback();
      }, timeoutMs);
    }
  } else if (embedFallback) {
    // No embed markup present — make sure fallback is visible.
    showFallback();
  }
})();
