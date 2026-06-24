/* Persistent Alpha top-nav + team-profile modal, injected app-wide
   (see BobosHome._inject_alpha_nav).

   1) TEAM MODAL: clicking any /team/<org> link opens the team profile as a big
      card overlay (iframe of the team page) with a blurred backdrop + X, instead
      of navigating away. Runs everywhere EXCEPT inside the modal's own iframe.

   2) NAV BAR: the single source of truth for the Alpha navigation bar — rendered
      identically on every Alpha page. Shown always on /alpha and /team/*, in
      Alpha mode on every other page, never on the classic home (/), and never
      inside the modal iframe. */
(function () {
  var inIframe = false;
  try { inIframe = (window.self !== window.top); } catch (e) { inIframe = true; }

  // ---- 1) Team-profile modal (app-wide, but not inside the modal iframe) ----
  if (!inIframe && !window.__teamModalSetup) {
    window.__teamModalSetup = true;
    setupTeamModal();
  }

  // ---- 2) Nav bar ----
  var p = location.pathname;
  if (inIframe) return;                               // no nav inside the modal iframe
  if (p === '/') return;                              // classic home renders its own
  var ALWAYS = (p === '/alpha' || p.indexOf('/team/') === 0);
  if (!ALWAYS) {
    try { if (localStorage.getItem('bobo_ui') !== 'alpha') return; } catch (e) { return; }
  }
  if (document.querySelector('.alpha-navbar')) return;

  // Ensure the nav's fonts (Plus Jakarta Sans / DM Sans) load on EVERY page.
  // Some pages don't load them, so the bar fell back to a system font with
  // different letter widths — shifting the tab positions between pages.
  if (!document.getElementById('anav-fonts')) {
    var ff = document.createElement('link');
    ff.id = 'anav-fonts'; ff.rel = 'stylesheet';
    ff.href = 'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&family=DM+Sans:wght@400;500;700&display=swap';
    document.head.appendChild(ff);
  }

  var links = [
    ['/alpha', 'Home'],
    ['/mapelo/', 'BenPom'],
    ['/vct/', 'Leaderboards'],
    ['/highs/', 'All-Time Highs/Lows'],
    ['/articles/', 'Articles'],
    ['/mapelo/pythagorean/', 'Pythagorean Ratings'],
    ['/mapelo/rankings/', 'Historical Rankings']
  ];

  // Active = the link whose base path is the LONGEST prefix of the current path,
  // so e.g. /mapelo/modern/ lights ONLY "BenPom" sub-pages correctly, and "Home"
  // (/alpha) lights only on the home page.
  var activeI = -1, bestLen = 0;
  links.forEach(function (l, i) {
    var base = l[0].split('#')[0];
    if (base && p.indexOf(base) === 0 && base.length > bestLen) {
      bestLen = base.length; activeI = i;
    }
  });

  var css =
    // Reserve the scrollbar gutter on every Alpha page so page width (and thus
    // the fixed nav + centered content) doesn't jump horizontally when moving
    // between a page that scrolls and one that doesn't.
    'html{scrollbar-gutter:stable;}' +
    '.alpha-navbar{position:fixed;top:0;left:0;right:0;z-index:99999;display:flex;align-items:center;gap:14px;' +
    'padding:8px 18px;background:rgba(255,255,255,.93);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);' +
    "border-bottom:1px solid #eceef2;box-shadow:0 2px 12px #0000000d;font-family:'DM Sans',system-ui,sans-serif;}" +
    ".alpha-navbar .an-brand{display:flex;align-items:center;gap:7px;font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1rem;color:#16121d;text-decoration:none;flex-shrink:0;}" +
    '.alpha-navbar .an-brand img{height:1.35em;width:auto;}' +
    '.alpha-navbar .an-links{display:flex;gap:5px;overflow-x:auto;flex:1;scrollbar-width:none;}' +
    '.alpha-navbar .an-links::-webkit-scrollbar{display:none;}' +
    '.alpha-navbar .an-link{flex:0 0 auto;font-size:.8rem;font-weight:700;color:#6b6478;text-decoration:none;padding:6px 12px;border-radius:999px;white-space:nowrap;transition:color .15s,background .15s;}' +
    '.alpha-navbar .an-link:hover{color:#16121d;background:#f3eefb;}' +
    '.alpha-navbar .an-link.active{color:#fff;background:#7c4dd6;}' +
    '.alpha-navbar .an-switch{display:flex;align-items:center;gap:8px;cursor:pointer;flex-shrink:0;}' +
    '.alpha-navbar .an-switch span{font-size:.74rem;font-weight:700;color:#9a93a6;}' +
    '.alpha-navbar .an-switch span.on{color:#16121d;}' +
    '.alpha-navbar .an-track{position:relative;width:38px;height:21px;border-radius:999px;background:#7c4dd6;}' +
    '.alpha-navbar .an-knob{position:absolute;top:2px;left:2px;width:17px;height:17px;border-radius:50%;background:#fff;transform:translateX(17px);box-shadow:0 1px 4px #0003;}' +
    '@media(max-width:600px){.alpha-navbar{gap:9px;padding:7px 12px;}.alpha-navbar .an-switch span{display:none;}}';

  var st = document.createElement('style');
  st.textContent = css;
  document.head.appendChild(st);

  var bar = document.createElement('div');
  bar.className = 'alpha-navbar';

  var brand = document.createElement('a');
  brand.className = 'an-brand';
  brand.href = '/alpha';
  brand.innerHTML = '<img src="/logo.svg" alt="B">Bobo gg';
  bar.appendChild(brand);

  var lwrap = document.createElement('div');
  lwrap.className = 'an-links';
  links.forEach(function (l, i) {
    var a = document.createElement('a');
    a.className = 'an-link';
    a.href = l[0];
    a.textContent = l[1];
    if (i === activeI) a.className += ' active';
    lwrap.appendChild(a);
  });
  bar.appendChild(lwrap);

  var sw = document.createElement('div');
  sw.className = 'an-switch';
  sw.title = 'Switch to the classic layout';
  sw.innerHTML = '<span>Classic</span><div class="an-track"><div class="an-knob"></div></div><span class="on">Alpha</span>';
  sw.addEventListener('click', function () {
    try { localStorage.setItem('bobo_ui', 'classic'); } catch (e) {}
    location.href = '/';
  });
  bar.appendChild(sw);

  document.body.insertBefore(bar, document.body.firstChild);

  function pad() { document.body.style.paddingTop = bar.offsetHeight + 'px'; }
  pad();
  window.addEventListener('resize', pad);
  setTimeout(pad, 300);

  // ---- team modal implementation ----
  function setupTeamModal() {
    var mcss =
      '#teamModal{position:fixed;inset:0;z-index:100000;display:none;align-items:center;justify-content:center;padding:26px;' +
      'background:rgba(18,11,28,.55);backdrop-filter:blur(8px) saturate(1.1);-webkit-backdrop-filter:blur(8px) saturate(1.1);}' +
      '#teamModal.on{display:flex;}' +
      '#teamModal .tm-card{position:relative;width:min(1080px,96vw);height:70vh;max-height:92vh;background:#fff;border-radius:20px;' +
      'overflow:hidden;box-shadow:0 34px 100px #00000066;animation:tmIn .22s cubic-bezier(.2,.8,.3,1);}' +
      '#teamModal .tm-card.tm-narrow{width:min(580px,94vw);border-radius:24px;}' +
      '#teamModal .tm-card.tm-narrow .tm-x{background:transparent;box-shadow:none;color:#9e96a8;width:auto;height:auto;font-size:1.5rem;top:12px;right:16px;}' +
      '#teamModal .tm-card.tm-narrow .tm-x:hover{background:transparent;color:#16121d;transform:none;}' +
      '@keyframes tmIn{from{opacity:0;transform:scale(.97) translateY(10px)}to{opacity:1;transform:none}}' +
      '#teamModal .tm-frame{width:100%;height:100%;border:0;display:block;background:#fff;}' +
      '#teamModal .tm-load{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#9a93a6;' +
      "font-family:'DM Sans',system-ui,sans-serif;font-size:.85rem;font-weight:600;}" +
      '#teamModal .tm-x{position:absolute;top:13px;right:13px;z-index:3;width:36px;height:36px;border:0;border-radius:50%;' +
      "background:#fff;color:#16121d;font-size:1.35rem;line-height:1;cursor:pointer;box-shadow:0 4px 14px #0004;" +
      'display:flex;align-items:center;justify-content:center;transition:background .15s,transform .15s;}' +
      '#teamModal .tm-x:hover{background:#f3eefb;transform:scale(1.06);}' +
      '@media(max-width:600px){#teamModal{padding:10px;}#teamModal .tm-card{width:100vw;height:96vh;border-radius:16px;}}';
    var s = document.createElement('style'); s.textContent = mcss; document.head.appendChild(s);

    var ov = document.createElement('div'); ov.id = 'teamModal';
    ov.innerHTML = '<div class="tm-card"><button class="tm-x" aria-label="Close">&times;</button>' +
                   '<div class="tm-load">Loading&hellip;</div>' +
                   '<iframe class="tm-frame" title="Team profile"></iframe></div>';
    document.body.appendChild(ov);
    var frame = ov.querySelector('.tm-frame'), load = ov.querySelector('.tm-load');
    var card = ov.querySelector('.tm-card');
    frame.addEventListener('load', function () { if (frame.src && frame.src.indexOf('about:blank') < 0) load.style.display = 'none'; });

    // The team page inside the iframe posts its content height so the card can
    // auto-size to fit (no internal scrolling), capped at 92% of the viewport.
    window.addEventListener('message', function (e) {
      if (e.source !== frame.contentWindow) return;
      if (e.data && typeof e.data === 'object' && typeof e.data.__teamH === 'number') {
        card.style.height = Math.min(e.data.__teamH, Math.round(window.innerHeight * 0.92)) + 'px';
      }
    });

    // Open the modal on an arbitrary same-origin URL (team profile OR the
    // standalone /vct/player card). Both pages post {__teamH} so the card
    // auto-sizes identically.
    function openCard(href) {
      // Player cards use the compact /vct/-modal size; team profiles use the big card.
      card.classList.toggle('tm-narrow', href.indexOf('/vct/player') === 0);
      load.style.display = 'flex';
      frame.src = href;
      ov.classList.add('on');
      document.documentElement.style.overflow = 'hidden';
    }
    function openTeam(org) { openCard('/team/' + encodeURIComponent(org)); }
    function closeTeam() {
      ov.classList.remove('on');
      frame.src = 'about:blank';
      document.documentElement.style.overflow = '';
    }
    ov.addEventListener('click', function (e) {
      if (e.target === ov || (e.target.closest && e.target.closest('.tm-x'))) closeTeam();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && ov.classList.contains('on')) closeTeam();
    });

    // Intercept clicks on /team/<org> links AND /vct/player?... links (capture
    // phase, so it beats other handlers like the leaderboard row toggle). Let
    // modified clicks through so cmd/ctrl-click still opens the full page in a
    // new tab. Both open the same overlay; the player-card link opens as-is so
    // its query string (profile/stat/event) is preserved.
    document.addEventListener('click', function (e) {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
      var a = e.target.closest && e.target.closest('a[href]');
      if (!a) return;
      if (a.target === '_blank') return;
      var href = a.getAttribute('href') || '';
      var m = href.match(/^\/team\/([^?#]+)/);
      if (m) { e.preventDefault(); e.stopPropagation(); openTeam(decodeURIComponent(m[1])); return; }
      if (href.indexOf('/vct/player') === 0) { e.preventDefault(); e.stopPropagation(); openCard(href); }
    }, true);
  }
})();
