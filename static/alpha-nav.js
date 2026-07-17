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
  if (p === '/classic') return;                       // classic home renders its own
  var ALWAYS = (p === '/' || p === '/alpha' || p.indexOf('/team/') === 0);
  if (!ALWAYS) {
    // Alpha is the default — show the nav everywhere except when the user has
    // explicitly chosen the classic layout.
    try { if (localStorage.getItem('bobo_ui') === 'classic') return; } catch (e) {}
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

  // A nav entry is either a link [href, label] or a dropdown {label, items:[...]}.
  var links = [
    ['/', 'Home'],
    ['/mapelo/', 'BenPom'],
    ['/articles/', 'Articles'],
    ['/vct/', 'Leaderboards'],
    ['/highs/', 'All-Time Highs/Lows'],
    {label: 'Historical VCT Tools', items: [
      ['/mapelo/rankings/', 'Rankings'],
      ['/mapelo/matchup/', 'Matchup Simulator']
    ]},
    ['/mapelo/pythagorean/', 'Pythagorean Ratings'],
    ['/match-data/', 'Match Data']
  ];

  // Active = the entry (or dropdown child) whose base path is the LONGEST prefix
  // of the current path, so e.g. /mapelo/modern/ lights ONLY "BenPom", and a
  // dropdown child (e.g. /mapelo/rankings/) lights its parent "Historical".
  var activeI = -1, activeChild = -1, bestLen = 0;
  links.forEach(function (l, i) {
    if (Array.isArray(l)) {
      var base = l[0].split('#')[0];
      if (base && p.indexOf(base) === 0 && base.length > bestLen) {
        bestLen = base.length; activeI = i; activeChild = -1;
      }
    } else if (l.items) {
      l.items.forEach(function (c, ci) {
        var cb = c[0].split('#')[0];
        if (cb && p.indexOf(cb) === 0 && cb.length > bestLen) {
          bestLen = cb.length; activeI = i; activeChild = ci;
        }
      });
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
    '.alpha-navbar .an-dd-tog{background:none;border:none;cursor:pointer;font-family:inherit;display:inline-flex;align-items:center;gap:5px;}' +
    '.alpha-navbar .an-caret{font-size:.92rem;opacity:.8;line-height:1;}' +
    ".an-dd-menu{position:fixed;z-index:100000;background:#fff;border:1px solid #eceef2;border-radius:13px;box-shadow:0 14px 38px #0000001f;padding:6px;min-width:188px;display:none;font-family:'DM Sans',system-ui,sans-serif;}" +
    '.an-dd-menu.open{display:block;}' +
    '.an-dd-menu a{display:block;padding:9px 13px;border-radius:9px;font-size:.8rem;font-weight:700;color:#6b6478;text-decoration:none;white-space:nowrap;transition:color .12s,background .12s;}' +
    '.an-dd-menu a:hover{background:#f3eefb;color:#16121d;}' +
    '.an-dd-menu a.active{color:#7c4dd6;background:#f3eefb;}' +
    // Phones: the 7 nav items can't fit one row at a readable size, so shrink
    // the text and let the links WRAP (brand + toggle on row 1, pills below)
    // instead of horizontal-scrolling — every option stays visible. pad() keys
    // off bar.offsetHeight, so the taller wrapped nav reserves its own space.
    '@media(max-width:600px){' +
      '.alpha-navbar{gap:7px 9px;padding:7px 11px;flex-wrap:wrap;}' +
      '.alpha-navbar .an-brand{font-size:.9rem;}' +
      '.alpha-navbar .an-links{flex:1 1 100%;order:3;overflow-x:visible;flex-wrap:wrap;gap:5px;}' +
      '.alpha-navbar .an-link{font-size:.7rem;padding:5px 9px;}' +
      '.alpha-navbar .an-caret{font-size:.82rem;}' +
    '}';

  var st = document.createElement('style');
  st.textContent = css;
  document.head.appendChild(st);

  var bar = document.createElement('div');
  bar.className = 'alpha-navbar';

  var brand = document.createElement('a');
  brand.className = 'an-brand';
  brand.href = '/';
  brand.innerHTML = '<img src="/logo.svg" alt="B">Bobo gg';
  bar.appendChild(brand);

  var lwrap = document.createElement('div');
  lwrap.className = 'an-links';
  links.forEach(function (l, i) {
    if (Array.isArray(l)) {
      var a = document.createElement('a');
      a.className = 'an-link' + (i === activeI ? ' active' : '');
      a.href = l[0];
      a.textContent = l[1];
      lwrap.appendChild(a);
      return;
    }
    // Dropdown: a toggle in the bar + a fixed menu appended to <body> (so it's
    // not clipped by the bar's horizontal-scroll overflow or backdrop-filter).
    var tog = document.createElement('button');
    tog.type = 'button';
    tog.className = 'an-link an-dd-tog' + (i === activeI ? ' active' : '');
    tog.appendChild(document.createTextNode(l.label));
    var caret = document.createElement('span');
    caret.className = 'an-caret'; caret.textContent = '▾';
    tog.appendChild(caret);
    lwrap.appendChild(tog);

    var menu = document.createElement('div');
    menu.className = 'an-dd-menu';
    l.items.forEach(function (c, ci) {
      var ca = document.createElement('a');
      ca.className = 'an-dd-item' + (i === activeI && ci === activeChild ? ' active' : '');
      ca.href = c[0];
      ca.textContent = c[1];
      menu.appendChild(ca);
    });
    document.body.appendChild(menu);

    var hideT;
    function showMenu() {
      clearTimeout(hideT);
      var r = tog.getBoundingClientRect();
      menu.style.left = Math.round(r.left) + 'px';
      menu.style.top = Math.round(r.bottom + 6) + 'px';
      menu.classList.add('open');
    }
    function hideMenu() { hideT = setTimeout(function () { menu.classList.remove('open'); }, 140); }
    tog.addEventListener('mouseenter', showMenu);
    tog.addEventListener('mouseleave', hideMenu);
    menu.addEventListener('mouseenter', function () { clearTimeout(hideT); });
    menu.addEventListener('mouseleave', hideMenu);
    tog.addEventListener('click', function (e) {
      e.preventDefault();
      if (menu.classList.contains('open')) menu.classList.remove('open'); else showMenu();
    });
  });
  bar.appendChild(lwrap);

  document.body.insertBefore(bar, document.body.firstChild);

  function pad() { document.body.style.paddingTop = bar.offsetHeight + 'px'; }
  pad();
  window.addEventListener('resize', pad);
  setTimeout(pad, 300);

  // The bar now renders before the page content (injected at the top of <body>),
  // which means its labels can paint in a fallback font and then reflow — visibly
  // shifting the tabs — when the web font swaps in. Reserve the bar's height right
  // away (done by pad() above) but keep its content invisible until the nav's own
  // fonts are ready, then reveal. Reserved space => no layout jump; cached fonts
  // (the common case when moving between pages) resolve in a few ms, so the reveal
  // is imperceptible. A short fallback guarantees the bar never stays hidden.
  if (document.fonts && document.fonts.load) {
    bar.style.visibility = 'hidden';
    var __shown = false;
    var revealBar = function () { if (__shown) return; __shown = true; bar.style.visibility = ''; pad(); };
    // Wait ONLY on the bar's own two web fonts — not document.fonts.ready, which
    // on heavy pages (Pythagorean, Historical Rankings, the simulator) blocks on
    // every chart/content font and keeps the bar hidden long enough to blip.
    // The font then triggers a layout reflow a frame AFTER fonts.load resolves,
    // so reveal two animation frames later — otherwise the bar paints in the
    // fallback font and the tabs visibly shift once the real font lands.
    var afterFonts = function () {
      if (window.requestAnimationFrame)
        requestAnimationFrame(function () { requestAnimationFrame(revealBar); });
      else revealBar();
    };
    Promise.all([
      document.fonts.load('800 1rem "Plus Jakarta Sans"'),
      document.fonts.load('700 1rem "DM Sans"')
    ]).then(afterFonts).catch(afterFonts);
    setTimeout(revealBar, 400);
  }

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
