/**
 * StreamFlix — Churn Mitigation Demo  v3
 *
 * Key behaviours:
 *  - fetchTrails()      GET /api/trails/{userId}  → multi-row render
 *  - handleSkip()       POST /api/events → immediate HUD update → debounced re-fetch
 *  - handlePlay()       POST /api/events → positive Rocchio → debounced re-fetch
 *  - Bi-directional HUD  red negative section + green positive section
 *  - Genre rotation     when a genre row is swapped the row animates in
 *  - Boost modal        admin panel at Alt+A / nav button
 */
'use strict';

// ============================================================
// Config (injected by Jinja2 into window.DEMO_CONFIG)
// ============================================================
const CFG = {
  apiBase:          window.DEMO_CONFIG?.apiBase          ?? '/api',
  // userId kept for legacy reference — runtime active user is STATE.activeUser
  userId:           window.DEMO_CONFIG?.defaultUser      ?? 'demo_horror',
  demoUsers:        (window.DEMO_CONFIG?.demoUsers ?? 'demo_horror,demo_comedy,demo_scifi')
                      .split(',').map(s => s.trim()).filter(Boolean),
  threshold:        window.DEMO_CONFIG?.negativeThreshold ?? 3,
  genreThreshold:   window.DEMO_CONFIG?.genreSkipThreshold ?? 2,
  rejectionStrength:window.DEMO_CONFIG?.rejectionStrength ?? 0.8,
  alpha:            window.DEMO_CONFIG?.alpha             ?? 0.4,
  beta:             window.DEMO_CONFIG?.beta              ?? 0.3,
  driftRate:        window.DEMO_CONFIG?.driftRate         ?? 0.05,
  genreRowsCount:   window.DEMO_CONFIG?.genreRowsCount    ?? 3,
  maxBoostedSlots:  window.DEMO_CONFIG?.maxBoostedSlots   ?? 3,
  playRowThreshold: window.DEMO_CONFIG?.playRowThreshold  ?? 1,
};

// Fixed radar axes for the "Live Profile Vector" HUD panel — must match
// RADAR_AXES in app/services/vector_projection.py so labels line up with
// the projection values returned by GET /api/trails.
const RADAR_AXES = [
  'Action', 'Comedy', 'Drama', 'Horror',
  'Romance', 'Sci-Fi', 'Thriller', 'Documentary',
];

// ============================================================
// State
// ============================================================
const STATE = {
  // Active persona (mutable — changes when user switches)
  activeUser:       window.DEMO_CONFIG?.defaultUser ?? 'demo_horror',
  // SSE EventSource instance — stored so switchPersona can close it
  sseSource:        null,
  // Persona metadata cache (populated by initPersonaSwitcher)
  personas:         [],
  // Negative signals
  negativeCount:    0,
  mitigationActive: false,
  penalizedGenre:   null,
  genreSkipCounts:  {},
  swappedGenres:    [],
  rocchioGenre:     null,
  // Positive signals
  positiveCount:       0,
  positiveGenreCounts: {},
  emergingGenreCounts: {},
  betaEffective:       0,
  // Shared
  rocchioApplied:   false,
  querySource:      'user_vector',
  profileGenres:    [],
  toastShown:       false,
  loading:          false,
  refetchTimer:     null,
  heroMovie:        null,
  // Live Profile Vector radar — current + previous projections (previous
  // is kept so we can compute per-genre deltas for the callout text).
  userVectorProjection: null,
  rfyQueryProjection:   null,
  prevUserVectorProjection: null,
  prevRfyQueryProjection:   null,
  vectorShift:          0,
  deltaCalloutTimer:    null,
};

// ============================================================
// DOM refs
// ============================================================
let D = {};
function initDOM() {
  D = {
    // HUD
    hudPanel:       document.getElementById('hud-panel'),
    negCounter:     document.getElementById('neg-counter'),
    progressFill:   document.getElementById('progress-fill'),
    statusBadge:    document.getElementById('status-badge'),
    statusDot:      document.getElementById('status-dot'),
    statusText:     document.getElementById('status-text'),
    penaltyNotice:  document.getElementById('penalty-notice'),
    penaltyGenre:   document.getElementById('penalty-genre'),
    algoSource:     document.getElementById('algo-source'),
    genreHudRows:      document.getElementById('genre-hud-rows'),
    // Positive signals HUD
    positiveSection:   document.getElementById('positive-hud-section'),
    positiveHudRows:   document.getElementById('positive-hud-rows'),
    positiveTotalBadge:document.getElementById('positive-total-badge'),
    positiveNotice:    document.getElementById('positive-notice'),
    positiveNoticeGenre: document.getElementById('positive-notice-genre'),
    rocchioBar:     document.getElementById('rocchio-bar'),
    rocchioInfo:    document.getElementById('rocchio-info'),
    // Live Profile Vector radar
    vectorRadarSvg:     document.getElementById('vector-radar-svg'),
    vectorShiftBadge:   document.getElementById('vector-shift-badge'),
    vectorDeltaCallout: document.getElementById('vector-delta-callout'),
    // Hero
    heroBg:         document.getElementById('hero-bg'),
    heroTitle:      document.getElementById('hero-title'),
    heroPlot:       document.getElementById('hero-plot'),
    heroGenres:     document.getElementById('hero-genres'),
    heroScore:      document.getElementById('hero-score'),
    heroBtnPlay:    document.getElementById('hero-btn-play'),
    heroBtnSkip:    document.getElementById('hero-btn-skip'),
    // Content
    trailsContainer: document.getElementById('trails-container'),
    loadingSkeleton: document.getElementById('loading-skeleton'),
    // Toast
    toastBanner:    document.getElementById('toast-banner'),
    toastMsg:       document.getElementById('toast-msg'),
    toastClose:     document.getElementById('toast-close'),
    // Nav
    navUserId:      document.getElementById('nav-user-id'),
    btnReset:       document.getElementById('btn-reset'),
    btnAdmin:       document.getElementById('btn-admin'),
    // Admin modal
    adminOverlay:   document.getElementById('admin-overlay'),
    adminClose:     document.getElementById('admin-close'),
    movieSearch:    document.getElementById('movie-search'),
    searchResults:  document.getElementById('search-results'),
    selectedMovie:  document.getElementById('selected-movie'),
    selectedTitle:  document.getElementById('selected-title'),
    clearSelection: document.getElementById('clear-selection'),
    boostScore:     document.getElementById('boost-score'),
    boostScoreVal:  document.getElementById('boost-score-val'),
    boostLabel:     document.getElementById('boost-label'),
    boostExpiry:    document.getElementById('boost-expiry'),
    btnBoost:       document.getElementById('btn-boost'),
    activeBoostsTbl: document.getElementById('active-boosts-tbody'),
    // Mobile HUD
    mobileCounter:  document.getElementById('mobile-neg-counter'),
    mobileStatus:   document.getElementById('mobile-status-text'),
  };

}

// ============================================================
// API
// ============================================================
async function apiFetch(path, opts = {}) {
  const res = await fetch(`${CFG.apiBase}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(e.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

// ============================================================
// Fetch trails
// ============================================================
async function fetchTrails() {
  if (STATE.loading) { scheduleRefetch(200); return; }
  STATE.loading = true;
  showSkeleton(true);

  try {
    const data = await apiFetch(`/trails/${STATE.activeUser}`);
    syncSession(data.session);
    STATE.rocchioApplied     = data.rocchio_applied      ?? false;
    STATE.querySource         = data.query_source        ?? 'user_vector';
    STATE.profileGenres       = data.profile_genres      ?? [];
    STATE.betaEffective       = data.beta_effective      ?? 0;
    if (data.positive_genre_counts)
      STATE.positiveGenreCounts = data.positive_genre_counts;
    if (data.emerging_genre_counts)
      STATE.emergingGenreCounts = data.emerging_genre_counts ?? {};

    // Live Profile Vector — keep previous projections around for delta callouts
    STATE.prevUserVectorProjection = STATE.userVectorProjection;
    STATE.prevRfyQueryProjection   = STATE.rfyQueryProjection;
    STATE.userVectorProjection     = data.user_vector_projection ?? null;
    STATE.rfyQueryProjection       = data.rfy_query_projection   ?? null;
    STATE.vectorShift              = data.vector_shift ?? 0;
    updateVectorRadar();

    // Hero = first non-boosted movie (organic pick, not an admin inject)
    const rfy = data.recommended_for_you ?? [];
    const heroMovie = rfy.find(m => !m.is_boosted) ?? rfy[0];
    renderHero(heroMovie);
    renderTrails(data, heroMovie?.id);
    updateHUD();
    updateRocchioBar();
  } catch (err) {
    console.error('fetchTrails:', err);
    showTrailError(err.message);
  } finally {
    STATE.loading = false;
    showSkeleton(false);
  }
}

function scheduleRefetch(ms = 350) {
  if (STATE.refetchTimer) clearTimeout(STATE.refetchTimer);
  STATE.refetchTimer = setTimeout(async () => {
    STATE.refetchTimer = null;
    await fetchTrails();
  }, ms);
}

// ============================================================
// Event dispatch
// ============================================================
async function sendEvent(movieId, eventType, genre, trail, allGenres) {
  try {
    const data = await apiFetch('/events', {
      method: 'POST',
      body: JSON.stringify({
        user_id: STATE.activeUser, movie_id: movieId,
        event_type: eventType, genre, trail,
        // all_genres: full genre list for play events so every genre gets attributed.
        // For skips this is ignored by the backend (skip attribution is always single genre).
        all_genres: allGenres ?? null,
      }),
    });
    syncSession(data.session);
    updateHUD();
    return data;
  } catch (err) {
    console.warn('sendEvent:', err);
  }
}

// ============================================================
// Action handlers
// ============================================================
async function handlePlay(movieId, genre, trail, allGenres) {
  highlightCard(movieId, 'play');
  await sendEvent(movieId, 'play', genre, trail, allGenres);
  // Positive signal — re-fetch so the attraction effect is visible immediately
  scheduleRefetch(350);
}

async function handleSkip(movieId, genre, trail, allGenres) {
  highlightCard(movieId, 'skip');
  const wasBeforeGenreThreshold = (STATE.genreSkipCounts[genre] ?? 0) < CFG.genreThreshold;
  // Capture BEFORE sendEvent so we can detect the first global threshold crossing
  const wasMitigationActive = STATE.mitigationActive;

  await sendEvent(movieId, 'skip', genre, trail, allGenres);

  // Toast 1: global threshold crossed → RFY re-ranked
  if (!wasMitigationActive && STATE.mitigationActive && !STATE.toastShown) {
    showToast('global', null);
    STATE.toastShown = true;
  }
  // Toast 2: per-genre threshold crossed → that genre's row removed
  if (wasBeforeGenreThreshold && (STATE.genreSkipCounts[genre] ?? 0) >= CFG.genreThreshold) {
    showToast('genre', genre);
  }

  // Debounce: re-fetch trails shortly after the skip to show updated rows.
  // The 350ms delay is a pure debounce — it is NOT a race-condition workaround.
  // The SSE stream (initProfileStream) handles the real-time re-fetch once the
  // background profile write (reject_user_profile / drift_user_profile) completes.
  scheduleRefetch(350);
}

// ============================================================
// Session sync
// ============================================================
function syncSession(session) {
  if (!session) return;
  // Negative
  STATE.negativeCount    = session.negative_count    ?? 0;
  STATE.mitigationActive = session.mitigation_active ?? false;
  STATE.penalizedGenre   = session.penalized_genre   ?? null;
  STATE.genreSkipCounts  = session.genre_skip_counts ?? {};
  STATE.swappedGenres    = session.swapped_genres    ?? [];
  STATE.rocchioGenre     = session.rocchio_genre     ?? null;
  // Positive
  STATE.positiveCount       = session.positive_count        ?? 0;
  STATE.positiveGenreCounts = session.positive_genre_counts ?? {};
  // Query source (also set by trails response)
  if (session.query_source) STATE.querySource = session.query_source;
}

// ============================================================
// HUD
// ============================================================
function updateHUD() {
  const { negativeCount, mitigationActive, penalizedGenre,
          genreSkipCounts, positiveCount, positiveGenreCounts,
          rocchioApplied, querySource } = STATE;
  const ratio = Math.min(negativeCount / CFG.threshold, 1);
  const pct   = Math.round(ratio * 100);
  const hasPos = positiveCount > 0;

  if (D.negCounter) D.negCounter.textContent = `${negativeCount} / ${CFG.threshold}`;
  if (D.mobileCounter) D.mobileCounter.textContent = `${negativeCount} / ${CFG.threshold}`;

  // Progress bar
  if (D.progressFill) {
    D.progressFill.style.width = `${pct}%`;
    D.progressFill.classList.remove('warning', 'danger');
    if      (ratio >= 1)    D.progressFill.classList.add('danger');
    else if (ratio >= 0.66) D.progressFill.classList.add('warning');
  }

  // HUD border — mitigation takes priority over positive glow
  if (D.hudPanel) {
    D.hudPanel.classList.toggle('mitigation-active', mitigationActive);
    D.hudPanel.classList.toggle('positive-active', hasPos && !mitigationActive);
  }

  // Status badge
  if (D.statusBadge && D.statusText) {
    D.statusBadge.classList.remove('optimal', 'warning', 'danger');
    D.statusDot?.classList.remove('pulse-red', 'pulse-green');
    if (mitigationActive) {
      D.statusBadge.classList.add('danger');
      D.statusDot?.classList.add('pulse-red');
      D.statusText.textContent = 'RFY Re-ranked (global skip)';
      if (D.mobileStatus) D.mobileStatus.textContent = 'Re-ranked';
    } else if (ratio >= 0.66) {
      D.statusBadge.classList.add('warning');
      D.statusText.textContent = `${negativeCount}/${CFG.threshold} — RFY shift soon`;
      if (D.mobileStatus) D.mobileStatus.textContent = 'Warning';
    } else if (hasPos) {
      D.statusBadge.classList.add('optimal');
      D.statusDot?.classList.add('pulse-green');
      D.statusText.textContent = 'RFY Attracted to Plays';
      if (D.mobileStatus) D.mobileStatus.textContent = 'Attracting';
    } else {
      D.statusBadge.classList.add('optimal');
      D.statusText.textContent = 'Optimal Experience';
      if (D.mobileStatus) D.mobileStatus.textContent = 'Optimal';
    }
  }

  // Penalty notice — now describes RFY behaviour, not genre penalty
  if (D.penaltyNotice) {
    const show = mitigationActive;
    D.penaltyNotice.classList.toggle('visible', show);
    if (show && D.penaltyGenre) {
      D.penaltyGenre.textContent = penalizedGenre ?? 'skipped content';
    }
  }

  // Algo source with human-readable labels
  if (D.algoSource) {
    const src = querySource;
    const labels = {
      'user_vector':    'profile baseline',
      'rfy_rejected':   'global skip → rejection',
      'rfy_positive':   'play attraction',
      'rfy_full':       'skip rejection + play attraction',
    };
    D.algoSource.textContent = labels[src] ?? src;
    D.algoSource.style.color =
      src === 'user_vector'   ? '#7DD3FC' :
      src === 'rfy_positive'  ? '#10B981' :
      src === 'rfy_full'      ? '#A78BFA' : '#ff6b6b';
  }

  // Per-genre negative breakdown
  renderGenreHUD(genreSkipCounts);

  // Positive signals section
  renderPositiveHUD(positiveCount, positiveGenreCounts);
}

function renderGenreHUD(counts) {
  if (!D.genreHudRows) return;
  const genres = STATE.profileGenres.length ? STATE.profileGenres : Object.keys(counts);
  if (!genres.length) { D.genreHudRows.innerHTML = ''; return; }

  D.genreHudRows.innerHTML = genres.slice(0, 6).map(g => {
    const c = counts[g] ?? 0;
    const triggered = c >= CFG.genreThreshold;
    const pct = Math.min(c / CFG.genreThreshold, 1) * 100;
    return `
    <div class="genre-hud-row">
      <span class="genre-hud-name">${escHtml(g)}</span>
      <div class="genre-hud-bar">
        <div class="genre-hud-fill${triggered ? ' triggered' : ''}" style="width:${pct}%"></div>
      </div>
      <span class="genre-hud-count">${c}/${CFG.genreThreshold}</span>
      ${triggered ? '<span class="genre-hud-flag">⚡</span>' : ''}
    </div>`;
  }).join('');
}

// ============================================================
// Live Profile Vector radar — SVG spider chart, no external libs
// ============================================================
const RADAR_CX = 100, RADAR_CY = 100;
const RADAR_MAX_R = 78;     // outer radius reserved for the polygon
const RADAR_LABEL_R = 92;   // radius for genre text labels
const RADAR_ANIM_MS = 400;

function _radarAngle(i, n) {
  return -Math.PI / 2 + (i * 2 * Math.PI) / n;   // start at top, clockwise
}

// Map raw cosine-similarity scores (both series combined) onto a shared
// [innerR, RADAR_MAX_R] radius scale so the two polygons stay visually
// comparable frame to frame.
function _radarScale(userScores, rfyScores, axes) {
  const all = axes.flatMap(g => [userScores?.[g] ?? 0, rfyScores?.[g] ?? 0]);
  let min = Math.min(...all), max = Math.max(...all);
  if (!isFinite(min) || !isFinite(max) || max - min < 1e-6) { min -= 0.1; max += 0.1; }
  const innerR = 10;
  return (v) => innerR + ((v - min) / (max - min)) * (RADAR_MAX_R - innerR);
}

function _radarPoints(scores, axes, scaleFn) {
  return axes.map((g, i) => {
    const angle = _radarAngle(i, axes.length);
    const r = scaleFn(scores?.[g] ?? 0);
    return [RADAR_CX + r * Math.cos(angle), RADAR_CY + r * Math.sin(angle)];
  });
}

function _pointsToStr(points) {
  return points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
}

function _lerpPoints(a, b, t) {
  return a.map(([ax, ay], i) => {
    const [bx, by] = b[i] ?? [ax, ay];
    return [ax + (bx - ax) * t, ay + (by - ay) * t];
  });
}

// Tween a <polygon>'s `points` attribute from its current shape to a new
// target shape over RADAR_ANIM_MS — gives the "vector bending" feel on
// every click without needing an animation library.
function _tweenPolygon(el, toPoints) {
  if (!el) return;
  const fromStr = el.getAttribute('points');
  const fromPoints = fromStr
    ? fromStr.trim().split(/\s+/).map(p => p.split(',').map(Number))
    : toPoints;
  if (fromPoints.length !== toPoints.length) {
    el.setAttribute('points', _pointsToStr(toPoints));
    return;
  }
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / RADAR_ANIM_MS, 1);
    const eased = 1 - Math.pow(1 - t, 3);   // ease-out cubic
    el.setAttribute('points', _pointsToStr(_lerpPoints(fromPoints, toPoints, eased)));
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

const SVG_NS = 'http://www.w3.org/2000/svg';

// Build one SVG element with the given attributes using the proper SVG
// namespace. Using innerHTML on an <svg> root to inject markup is
// unreliable across browsers/webviews (child elements can end up parsed
// outside the SVG namespace and simply fail to render) — createElementNS
// is the robust way to build SVG chrome dynamically.
function _svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  return el;
}

function renderVectorRadar(userScores, rfyScores) {
  const svg = D.vectorRadarSvg;
  if (!svg) return;
  const axes = RADAR_AXES;
  const n = axes.length;

  // Build static chrome (grid rings + axis lines + labels) once, using the
  // SVG DOM API directly (not innerHTML) so every element is guaranteed to
  // be created in the correct SVG namespace.
  if (!svg.dataset.built) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    // Grid rings at 25/50/75/100% of RADAR_MAX_R
    [0.25, 0.5, 0.75, 1.0].forEach(f => {
      const pts = axes.map((_, i) => {
        const a = _radarAngle(i, n);
        const r = 10 + f * (RADAR_MAX_R - 10);
        return `${(RADAR_CX + r * Math.cos(a)).toFixed(1)},${(RADAR_CY + r * Math.sin(a)).toFixed(1)}`;
      }).join(' ');
      svg.appendChild(_svgEl('polygon', { points: pts, class: 'radar-grid-ring' }));
    });

    // Axis spokes + labels
    axes.forEach((g, i) => {
      const a = _radarAngle(i, n);
      const x2 = RADAR_CX + RADAR_MAX_R * Math.cos(a);
      const y2 = RADAR_CY + RADAR_MAX_R * Math.sin(a);
      const lx = RADAR_CX + RADAR_LABEL_R * Math.cos(a);
      const ly = RADAR_CY + RADAR_LABEL_R * Math.sin(a);
      const anchor = Math.abs(Math.cos(a)) < 0.2 ? 'middle' : (Math.cos(a) > 0 ? 'start' : 'end');

      svg.appendChild(_svgEl('line', {
        x1: RADAR_CX, y1: RADAR_CY,
        x2: x2.toFixed(1), y2: y2.toFixed(1),
        class: 'radar-axis-line',
      }));

      const label = _svgEl('text', {
        x: lx.toFixed(1), y: ly.toFixed(1),
        'text-anchor': anchor, class: 'radar-axis-label',
      });
      label.textContent = g;
      svg.appendChild(label);
    });

    const userPoly = _svgEl('polygon', { class: 'radar-poly radar-poly-user', points: '' });
    userPoly.id = 'radar-poly-user';
    const rfyPoly = _svgEl('polygon', { class: 'radar-poly radar-poly-rfy', points: '' });
    rfyPoly.id = 'radar-poly-rfy';
    svg.appendChild(userPoly);
    svg.appendChild(rfyPoly);

    svg.dataset.built = '1';
  }

  const scaleFn = _radarScale(userScores, rfyScores, axes);
  const userPoints = _radarPoints(userScores, axes, scaleFn);
  const rfyPoints   = _radarPoints(rfyScores, axes, scaleFn);

  _tweenPolygon(svg.querySelector('#radar-poly-user'), userPoints);
  _tweenPolygon(svg.querySelector('#radar-poly-rfy'), rfyPoints);
}

function updateVectorShiftBadge(shift) {
  const badge = D.vectorShiftBadge;
  if (!badge) return;
  const pct = Math.round(Math.min(Math.max(shift, 0), 2) / 2 * 100);
  badge.textContent = `shift ${pct}%`;
  badge.classList.remove('shift-low', 'shift-med', 'shift-high');
  badge.classList.add(pct >= 40 ? 'shift-high' : pct >= 15 ? 'shift-med' : 'shift-low');
}

// Show a short-lived callout naming the 1-2 genres whose projection moved
// the most since the previous fetch — makes the "this click bent your
// vector toward/away from X" story explicit instead of relying on the
// audience to read the radar shape alone.
function showVectorDeltaCallout(prevScores, newScores) {
  const el = D.vectorDeltaCallout;
  if (!el || !prevScores || !newScores) return;

  const deltas = RADAR_AXES
    .map(g => ({ genre: g, delta: (newScores[g] ?? 0) - (prevScores[g] ?? 0) }))
    .filter(d => Math.abs(d.delta) > 0.01)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
    .slice(0, 2);

  if (!deltas.length) return;

  el.innerHTML = deltas.map(d => {
    const up = d.delta > 0;
    return `<span class="delta-chip ${up ? 'up' : 'down'}">${up ? '▲' : '▼'} ${escHtml(d.genre)} ${up ? '+' : ''}${d.delta.toFixed(2)}</span>`;
  }).join(' ');

  el.classList.remove('visible');
  // Force reflow so the fade-in animation re-triggers on rapid consecutive clicks
  void el.offsetWidth;
  el.classList.add('visible');

  if (STATE.deltaCalloutTimer) clearTimeout(STATE.deltaCalloutTimer);
  STATE.deltaCalloutTimer = setTimeout(() => el.classList.remove('visible'), 3000);
}

function updateVectorRadar() {
  const { userVectorProjection, rfyQueryProjection,
          prevRfyQueryProjection, prevUserVectorProjection, vectorShift } = STATE;
  if (!userVectorProjection || !rfyQueryProjection) return;

  renderVectorRadar(userVectorProjection, rfyQueryProjection);
  updateVectorShiftBadge(vectorShift);

  // Prefer describing the ephemeral RFY query's movement (changes on every
  // click) — fall back to the persistent baseline profile's movement (only
  // changes once the async drift/reject write lands).
  if (prevRfyQueryProjection) {
    showVectorDeltaCallout(prevRfyQueryProjection, rfyQueryProjection);
  } else if (prevUserVectorProjection) {
    showVectorDeltaCallout(prevUserVectorProjection, userVectorProjection);
  }
}

// ============================================================
// Positive signals HUD section
// ============================================================
function renderPositiveHUD(positiveCount, genreCounts) {
  if (!D.positiveSection) return;

  const hasPos = positiveCount > 0;
  D.positiveSection.classList.toggle('visible', hasPos);
  if (!hasPos) return;

  // Badge showing total play count in window
  if (D.positiveTotalBadge) {
    D.positiveTotalBadge.textContent = `${positiveCount} play${positiveCount > 1 ? 's' : ''}`;
  }

  // Per-genre play bars
  if (D.positiveHudRows) {
    const genres = Object.keys(genreCounts);
    if (!genres.length) {
      D.positiveHudRows.innerHTML = '<div style="font-size:10px;color:#10B981;opacity:0.7;">Embedding being processed…</div>';
    } else {
      const maxCount = Math.max(...Object.values(genreCounts), 1);
      D.positiveHudRows.innerHTML = genres.slice(0, 5).map(g => {
        const c   = genreCounts[g] ?? 0;
        const pct = Math.min(c / maxCount, 1) * 100;
        return `
        <div class="positive-hud-row">
          <span class="positive-hud-name">${escHtml(g)}</span>
          <div class="positive-hud-bar">
            <div class="positive-hud-fill" style="width:${pct}%"></div>
          </div>
          <span class="positive-hud-plays">${c} ▶</span>
        </div>`;
      }).join('');
    }
  }

  // Positive notice — show when positive Rocchio is the active source
  if (D.positiveNotice) {
    const isPos = ['rfy_positive','rfy_full'].includes(STATE.querySource);
    D.positiveNotice.classList.toggle('visible', isPos);
    if (isPos && D.positiveNoticeGenre) {
      const topGenre = Object.entries(genreCounts)
        .sort((a,b) => b[1]-a[1])[0]?.[0] ?? 'played content';
      D.positiveNoticeGenre.textContent = topGenre;
    }
  }

  // Emerging interests
  updateEmergingHUD();
}

function updateEmergingHUD() {
  const emerging    = STATE.emergingGenreCounts ?? {};
  const section     = document.getElementById('emerging-hud-section');
  const rowsEl      = document.getElementById('emerging-hud-rows');
  if (!section || !rowsEl) return;

  const entries = Object.entries(emerging).filter(([,c]) => c > 0);
  if (!entries.length) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';

  const threshold = CFG.playRowThreshold ?? 1;
  const maxCount  = Math.max(...entries.map(([,c]) => c), 1);

  rowsEl.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([genre, count]) => {
      // Bar fills toward PLAY_ROW_THRESHOLD so user can see how close they are
      const pct = Math.min(count / Math.max(threshold, 1), 1) * 100;
      return `
      <div class="emerging-hud-row">
        <span class="emerging-hud-name">${escHtml(genre)}</span>
        <div class="emerging-hud-bar">
          <div class="emerging-hud-fill" style="width:${pct}%"></div>
        </div>
        <span class="emerging-hud-plays">${count} ▶</span>
      </div>`;
    }).join('');
}

// ============================================================
// Rocchio bar
// ============================================================
function updateRocchioBar() {
  if (!D.rocchioBar) return;
  const src    = STATE.querySource;
  const active = STATE.rocchioApplied;
  D.rocchioBar.classList.toggle('visible', active);

  // Colour variant based on query source
  D.rocchioBar.classList.remove('positive', 'full');
  if (src === 'rfy_positive') D.rocchioBar.classList.add('positive');
  if (src === 'rfy_full')     D.rocchioBar.classList.add('full');

  if (active && D.rocchioInfo) {
    let detail;
    if (src === 'rfy_positive') {
      const topGenre = Object.entries(STATE.positiveGenreCounts)
        .sort((a,b) => b[1]-a[1])[0]?.[0] ?? 'played content';
      const betaStr = STATE.betaEffective > 0
        ? `β_eff=${STATE.betaEffective.toFixed(1)}`
        : `β=${CFG.beta}`;
      detail = `RFY attracted toward <strong>${escHtml(topGenre)}</strong> (${betaStr})`;
    } else if (src === 'rfy_full') {
      const betaStr = STATE.betaEffective > 0 ? STATE.betaEffective.toFixed(1) : CFG.beta;
      detail = `RFY: skip rejection (${CFG.rejectionStrength}) + play attraction (β_eff=${betaStr})`;
    } else if (src === 'rfy_rejected') {
      detail = `RFY re-ranked — ${STATE.negativeCount} skips removed from profile direction (strength=${CFG.rejectionStrength})`;
    } else {
      detail = `RFY re-ranked from global skip signals`;
    }
    D.rocchioInfo.innerHTML = detail;
  }
}

// ============================================================
// Toast
// ============================================================
function showToast(type, genre) {
  if (!D.toastBanner) return;
  if (D.toastMsg) {
    if (type === 'genre') {
      // Per-genre threshold: row removed
      D.toastMsg.innerHTML =
        `Too many <strong>${escHtml(genre)}</strong> skips — ` +
        `that genre row has been removed from your page.`;
    } else {
      // Global threshold: RFY shifts
      D.toastMsg.innerHTML =
        `Your "Recommended For You" feed has been re-ranked ` +
        `based on your skip patterns.`;
    }
  }
  D.toastBanner.classList.add('visible');
  setTimeout(dismissToast, 6000);
}
function dismissToast() {
  D.toastBanner?.classList.remove('visible');
}

// ============================================================
// HERO
// ============================================================
function renderHero(movie) {
  if (!movie) return;
  STATE.heroMovie = movie;
  const primaryGenre = movie.genres?.[0] ?? '';
  if (D.heroBg && movie.poster)  D.heroBg.style.backgroundImage = `url('${movie.poster}')`;
  if (D.heroTitle)  D.heroTitle.textContent  = movie.title ?? '';
  if (D.heroPlot)   D.heroPlot.textContent   = (movie.plot ?? '').slice(0, 180) + (movie.plot?.length > 180 ? '…' : '');
  if (D.heroGenres) D.heroGenres.innerHTML   = genreTagsHTML(movie.genres);
  if (D.heroScore)  D.heroScore.textContent  = movie.vector_score ? `${(movie.vector_score * 100).toFixed(0)}% Match` : '';
  // Hero is always in the RFY row → attribute all genres on both play and skip.
  if (D.heroBtnPlay) D.heroBtnPlay.onclick = () => handlePlay(movie.id, primaryGenre, 'rfy', movie.genres ?? [primaryGenre]);
  if (D.heroBtnSkip) D.heroBtnSkip.onclick = () => handleSkip(movie.id, primaryGenre, 'rfy', movie.genres ?? [primaryGenre]);
}

// ============================================================
// TRAILS RENDER
// ============================================================
function renderTrails(data, heroMovieId = null) {
  if (!D.trailsContainer) return;

  const rfy   = data.recommended_for_you ?? [];
  const gRows = data.genre_rows ?? [];

  // RFY grid: exclude the hero movie so it isn't duplicated.
  // Boosted movies stay at the front of the remaining list.
  const rfyMovies = heroMovieId
    ? rfy.filter(m => m.id !== heroMovieId)
    : rfy.slice(1);

  let html = '';

  // ── Rocchio bar ───────────────────────────────────────────
  html += `<div id="rocchio-bar" class="rocchio-bar${data.rocchio_applied ? ' visible' : ''}" role="status">
    <span id="rocchio-info">${data.rocchio_applied
      ? `Feed re-ranked using Rocchio projection — α = ${CFG.alpha}`
      : ''}</span>
  </div>`;

  // ── RFY row ───────────────────────────────────────────────
  const rfyBadge = data.rocchio_applied
    ? `<span class="trail-badge rocchio">⚡ Rocchio Active</span>`
    : `<span class="trail-badge personalised">✦ Personalised</span>`;

  html += `<div class="trail-row" id="row-rfy">
    <div class="trail-header">
      <h2 class="trail-title">Recommended For You</h2>
      ${rfyBadge}
      <span class="trail-subtitle">${escHtml(data.query_source ?? 'user_vector')}</span>
    </div>
    <div class="movie-row">${rfyMovies.map((m,i) => cardHTML(m, i, 'rfy')).join('')}</div>
  </div>`;

  // ── Genre rows ────────────────────────────────────────────
  for (const row of gRows) {
    const src         = row.row_source ?? 'profile';
    const isPlayed    = src === 'played';
    const isSwapped   = src === 'swapped';
    const isEmerging  = src === 'emerging';
    const isDiscovery = src === 'discovery';

    const rowClass = (isPlayed || isEmerging) ? ' swapped' : isSwapped ? ' swapped' : '';

    // Row title
    let rowTitle;
    if (isPlayed) {
      rowTitle = `Because you watched <span style="color:#10B981">${escHtml(row.genre)}</span>`;
    } else if (isEmerging) {
      rowTitle = `✨ You might like <span style="color:#a78bfa">${escHtml(row.genre)}</span>`;
    } else if (isDiscovery) {
      rowTitle = `🔭 Discover Something New`;
    } else {
      rowTitle = `${genreIconFor(row.genre)} ${escHtml(row.genre)}`;
    }

    // Badge
    let genreBadge;
    if (isPlayed) {
      genreBadge = `<span class="trail-badge" style="background:rgba(16,185,129,0.15);color:#10B981;">▶ You Watched</span>`;
    } else if (isEmerging) {
      genreBadge = `<span class="trail-badge" style="background:rgba(167,139,250,0.15);color:#a78bfa;">🌱 Emerging</span>`;
    } else if (isDiscovery) {
      genreBadge = `<span class="trail-badge" style="background:rgba(251,191,36,0.15);color:#fbbf24;">🔭 Explore</span>`;
    } else if (row.rocchio_applied) {
      genreBadge = `<span class="trail-badge rocchio">⚡ Re-ranked</span>`;
    } else if (isSwapped) {
      genreBadge = `<span class="trail-badge swapped">↺ Rotated</span>`;
    } else {
      genreBadge = `<span class="trail-badge genre">${escHtml(row.genre)}</span>`;
    }

    // Notice below title
    let notice = '';
    if (isPlayed) {
      notice = `<div class="swapped-notice" style="color:#10B981;">▶ Showing more <strong>${escHtml(row.genre)}</strong> based on your recent plays</div>`;
    } else if (isEmerging) {
      notice = `<div class="swapped-notice" style="color:#a78bfa;">🌱 Your interest in <strong>${escHtml(row.genre)}</strong> is growing — keep watching to unlock this row</div>`;
    } else if (isDiscovery) {
      notice = `<div class="swapped-notice" style="color:#fbbf24;">🔭 Content outside your usual taste — play to evolve your profile</div>`;
    } else if (row.swapped_from) {
      notice = `<div class="swapped-notice">↺ Replaced <strong>${escHtml(row.swapped_from)}</strong> — too many skips</div>`;
    }

    // Discovery row uses 'rfy' trail so play/skip events are attributed correctly
    const trailId = isDiscovery ? 'rfy' : `genre:${row.genre}`;

    html += `<div class="trail-row${rowClass}" id="row-genre-${escHtml(row.genre)}">
      <div class="trail-header">
        <h2 class="trail-title">${rowTitle}</h2>
        ${genreBadge}
      </div>
      ${notice}
      <div class="movie-row">${row.movies.map((m,i) => cardHTML(m, i, trailId)).join('')}</div>
    </div>`;
  }

  if (!rfyMovies.length && !gRows.length) {
    html = `<div class="empty-state"><div class="icon">🎬</div><h3>No content found</h3></div>`;
  }

  D.trailsContainer.innerHTML = html;

  // Re-wire rocchio bar DOM refs after innerHTML replace
  D.rocchioBar  = document.getElementById('rocchio-bar');
  D.rocchioInfo = document.getElementById('rocchio-info');
}

function genreIconFor(genre) {
  const icons = {
    Horror: '👻', Thriller: '⚡', Action: '🔫', Crime: '🕵️',
    Mystery: '🔍', Drama: '🎭', Comedy: '😂', Romance: '💕',
    'Sci-Fi': '🚀', Fantasy: '🧙', Adventure: '🌍', Animation: '🎨',
    Western: '🤠', War: '⚔️', History: '📜', Biography: '📖',
    Family: '👨‍👩‍👧', Musical: '🎵', Music: '🎸', Documentary: '🎥',
  };
  return icons[genre] ?? '🎬';
}

// ============================================================
// CARD HTML
// ============================================================
function cardHTML(movie, idx, trail) {
  // ── Genre attribution ──────────────────────────────────────────────────
  // rowGenre: the genre for THIS ROW — used for skips and as the primary genre.
  //   trail = "genre:Horror"  →  rowGenre = "Horror"
  //   trail = "rfy"           →  rowGenre = movie.genres[0]
  //
  // allGenres: ALL genres to attribute on PLAY events.
  //   Genre row  →  [rowGenre]          (single-element; row intent only)
  //   RFY row    →  movie.genres        (full list; Comedy also gets credit)
  //
  // Skip attribution is always single-genre (rowGenre) regardless of row type.
  // This ensures per-genre HUD counts and row visibility logic stay accurate.
  const isGenreRow = trail?.startsWith('genre:');
  const rowGenre   = isGenreRow
    ? trail.slice(6)              // "genre:Horror" → "Horror"
    : (movie.genres?.[0] ?? ''); // RFY row: use movie's primary genre

  // For play events: attribute all genres on RFY, only row genre on genre rows.
  const allGenres = isGenreRow
    ? [rowGenre]
    : (movie.genres ?? [rowGenre]);

  const safeId      = escHtml(movie.id);
  const safeTitle   = escHtml(movie.title ?? 'Unknown');
  const safeGenre   = escHtml(rowGenre);
  const safeTrail   = escHtml(trail ?? '');
  const safePoster  = escHtml(movie.poster ?? '');
  // Encode allGenres as a JSON string safe for an HTML attribute / JS string literal.
  // We pass it as a base64-encoded string to avoid any quoting/escaping issues.
  const allGenresB64 = btoa(JSON.stringify(allGenres));
  const year  = movie.year ? `${movie.year} · ` : '';
  const score = movie.vector_score ? movie.vector_score.toFixed(3) : '';
  const boostedClass = movie.is_boosted ? ' boosted' : '';
  const boostBadge   = movie.is_boosted
    ? `<span class="boost-badge ${(movie.boost_label ?? 'hot').toLowerCase()}">${escHtml(movie.boost_label ?? 'Hot')}</span>`
    : '';

  return `
<div class="movie-card${boostedClass}" id="card-${safeId}" data-id="${safeId}" data-genre="${safeGenre}" data-trail="${safeTrail}">
  ${boostBadge}
  <img src="${safePoster}" alt="${safeTitle}"
       loading="${idx < 6 ? 'eager' : 'lazy'}"
       onerror="window.__posterErr(this,'${safeTitle.replace(/'/g,"\\'")}')">
  <div class="card-overlay">
    <div style="margin-bottom:6px;">
      <div style="font-size:12px;font-weight:700;color:#fff;line-height:1.3;margin-bottom:3px;">${safeTitle}</div>
      <div style="font-size:10px;color:#b3b3b3;margin-bottom:4px;">${year}${genreTagsHTML(movie.genres)}</div>
      ${score ? `<span class="score-badge">● ${score}</span>` : ''}
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;">
      <button class="btn-play" onclick="window.__play('${safeId}','${safeGenre}','${safeTrail}','${allGenresB64}')">▶ Play</button>
      <button class="btn-skip" onclick="window.__skip('${safeId}','${safeGenre}','${safeTrail}','${allGenresB64}')">⏭ Skip</button>
    </div>
  </div>
  <div class="card-info">
    <div class="card-title">${safeTitle}</div>
    <div class="card-meta">${year}${(movie.genres ?? []).slice(0,2).join(' · ')}</div>
  </div>
</div>`;
}

function genreTagsHTML(genres) {
  if (!genres?.length) return '';
  return genres.slice(0,3).map(g => `<span class="genre-pill">${escHtml(g)}</span>`).join('');
}

function escHtml(s) {
  return String(s ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ============================================================
// Highlight flash
// ============================================================
function highlightCard(movieId, type) {
  const card = document.getElementById(`card-${movieId}`);
  if (!card) return;
  const o = document.createElement('div');
  o.style.cssText = `position:absolute;inset:0;border-radius:6px;pointer-events:none;
    background:${type==='play'?'rgba(16,185,129,0.22)':'rgba(229,9,20,0.22)'};
    z-index:20;transition:opacity 600ms ease;`;
  card.appendChild(o);
  setTimeout(() => { o.style.opacity='0'; }, 50);
  setTimeout(() => { o.remove(); }, 700);
}

// ============================================================
// Skeleton / loading
// ============================================================
function showSkeleton(show) {
  if (D.loadingSkeleton) D.loadingSkeleton.style.display = show ? 'block' : 'none';
  if (D.trailsContainer) D.trailsContainer.style.opacity = show ? '0.3' : '1';
}
function showTrailError(msg) {
  if (D.trailsContainer) {
    D.trailsContainer.innerHTML =
      `<div class="empty-state"><div class="icon">⚠️</div><h3 style="color:#ff6b6b">Error</h3>
      <p style="font-size:13px;color:#737373;">${escHtml(msg)}</p></div>`;
  }
}

// ============================================================
// RESET
// ============================================================
function _clearSessionState() {
  STATE.negativeCount = 0; STATE.mitigationActive = false;
  STATE.penalizedGenre = null; STATE.rocchioApplied = false;
  STATE.genreSkipCounts = {}; STATE.swappedGenres = [];
  STATE.rocchioGenre = null; STATE.toastShown = false;
  STATE.positiveCount = 0; STATE.positiveGenreCounts = {};
  STATE.emergingGenreCounts = {};
  STATE.querySource = 'user_vector'; STATE.profileGenres = [];
  dismissToast();
}

async function resetDemo() {
  // Reset ALL personas silently in parallel — each gets a fresh session
  // and restored user_vector so the demo always starts from a clean baseline.
  if (D.btnReset) { D.btnReset.textContent = 'Resetting…'; D.btnReset.disabled = true; }
  try {
    const allUsers = CFG.demoUsers.length ? CFG.demoUsers : [STATE.activeUser];
    await Promise.all(allUsers.map(uid =>
      apiFetch('/demo/reset', {
        method: 'POST',
        body: JSON.stringify({ user_id: uid }),
      }).catch(e => console.warn(`reset ${uid}:`, e))
    ));
    _clearSessionState();
    updateHUD();
    await fetchTrails();
  } catch (e) { console.error('reset:', e); }
  finally {
    if (D.btnReset) { D.btnReset.textContent = '↺ Reset Demo'; D.btnReset.disabled = false; }
  }
}

// ============================================================
// ADMIN BOOST MODAL
// ============================================================
let selectedMovieId = null;

function openAdmin() {
  const overlay = document.getElementById('admin-overlay');
  if (overlay) {
    overlay.classList.add('open');
    loadActiveBoosts();
  }
}
function closeAdmin() {
  const overlay = document.getElementById('admin-overlay');
  if (overlay) overlay.classList.remove('open');
}
// Expose globally so inline onclick= and mobile HUD button can call them
window.openAdmin  = openAdmin;
window.closeAdmin = closeAdmin;

async function loadActiveBoosts() {
  if (!D.activeBoostsTbl) return;
  try {
    const boosts = await apiFetch('/admin/boosts');
    if (!boosts.length) {
      D.activeBoostsTbl.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text-muted);padding:16px;">No active boosts</td></tr>`;
      return;
    }
    D.activeBoostsTbl.innerHTML = boosts.map(b => `
      <tr>
        <td style="color:var(--text-primary);font-weight:500;">${escHtml(b.title ?? b.movie_id)}</td>
        <td><span class="boost-badge ${(b.label||'hot').toLowerCase()}" style="position:static;animation:none;">${escHtml(b.label)}</span></td>
        <td>${b.boost_score.toFixed(1)}×</td>
        <td><button class="btn-remove-boost" onclick="removeBoost('${escHtml(b.movie_id)}')">Remove</button></td>
      </tr>`).join('');
  } catch (e) { console.warn('loadActiveBoosts:', e); }
}

let searchTimer = null;
async function onMovieSearch(e) {
  const q = e.target.value.trim();
  if (!q) { D.searchResults?.classList.remove('open'); return; }
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    try {
      const movies = await apiFetch(`/admin/movies/search?q=${encodeURIComponent(q)}&limit=8`);
      renderSearchResults(movies);
    } catch (err) { console.warn('search:', err); }
  }, 280);
}

function renderSearchResults(movies) {
  if (!D.searchResults) return;
  if (!movies.length) { D.searchResults.classList.remove('open'); return; }
  D.searchResults.innerHTML = movies.map(m => `
    <div class="movie-search-item" onclick="selectMovie('${escHtml(m.id)}','${escHtml(m.title)}','${escHtml(m.poster??'')}')">
      <img src="${escHtml(m.poster??'')}" onerror="this.src=''" style="width:28px;height:42px;object-fit:cover;border-radius:2px;background:#2a2a2a;">
      <div>
        <div class="movie-search-title">${escHtml(m.title)}</div>
        <div class="movie-search-meta">${m.year ?? ''} · ${(m.genres??[]).slice(0,2).join(', ')}</div>
      </div>
    </div>`).join('');
  D.searchResults.classList.add('open');
}

function selectMovie(id, title, poster) {
  selectedMovieId = id;
  D.searchResults?.classList.remove('open');
  if (D.movieSearch) D.movieSearch.value = '';
  if (D.selectedMovie) {
    D.selectedMovie.classList.add('visible');
    const img = D.selectedMovie.querySelector('img');
    if (img) { img.src = poster; }
    if (D.selectedTitle) D.selectedTitle.textContent = title;
  }
}

function clearMovieSelection() {
  selectedMovieId = null;
  if (D.selectedMovie) D.selectedMovie.classList.remove('visible');
  if (D.movieSearch)   D.movieSearch.value = '';
}

async function submitBoost() {
  if (!selectedMovieId) { alert('Please search and select a movie first.'); return; }
  if (D.btnBoost) { D.btnBoost.disabled = true; D.btnBoost.textContent = 'Boosting…'; }
  try {
    await apiFetch('/admin/boost', {
      method: 'POST',
      body: JSON.stringify({
        movie_id:     selectedMovieId,
        boost_score:  parseFloat(D.boostScore?.value ?? 1.5),
        label:        D.boostLabel?.value ?? 'Hot',
        expires_hours: D.boostExpiry?.value ? parseInt(D.boostExpiry.value) : null,
      }),
    });
    clearMovieSelection();
    await loadActiveBoosts();
    await fetchTrails();  // refresh to show boosted movie in top slots
  } catch (e) { alert(`Boost failed: ${e.message}`); }
  finally {
    if (D.btnBoost) { D.btnBoost.disabled = false; D.btnBoost.textContent = '🚀 Boost Movie'; }
  }
}

async function removeBoost(movieId) {
  try {
    await apiFetch(`/admin/boost/${movieId}`, { method: 'DELETE' });
    await loadActiveBoosts();
    await fetchTrails();
  } catch (e) { alert(`Remove failed: ${e.message}`); }
}
window.removeBoost = removeBoost;
window.selectMovie = selectMovie;
window.clearMovieSelection = clearMovieSelection;

// ============================================================
// Global inline onclick handlers
// ============================================================
window.__play = (id, genre, trail, allGenresB64) => {
  // Decode the base64-encoded JSON array of genres passed from cardHTML().
  // Falls back to [genre] if absent or malformed (e.g. hero button, legacy calls).
  let allGenres = [genre];
  if (allGenresB64) {
    try { allGenres = JSON.parse(atob(allGenresB64)); } catch (_) {}
  }
  handlePlay(id, genre, trail, allGenres);
};
window.__skip = (id, genre, trail, allGenresB64) => {
  // Decode base64-encoded JSON genre array — same pattern as window.__play.
  // For RFY skips this carries all movie genres so every genre's skip count
  // is incremented (fixes genre row removal when skipping from RFY).
  let allGenres = [genre];
  if (allGenresB64) {
    try { allGenres = JSON.parse(atob(allGenresB64)); } catch (_) {}
  }
  handleSkip(id, genre, trail, allGenres);
};
window.__posterErr = (img, title) => {
  img.onerror = null;
  const initials = title.split(' ').slice(0,2).map(w => w[0]??'').join('').toUpperCase();
  img.src = `data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='300'><rect width='200' height='300' fill='%23232323'/><text x='100' y='160' font-size='50' font-weight='bold' fill='%23555' text-anchor='middle' dominant-baseline='middle' font-family='sans-serif'>${encodeURIComponent(initials)}</text></svg>`;
};

// ============================================================
// Nav scroll
// ============================================================
function initNavScroll() {
  const nav = document.querySelector('.nav-bar');
  if (!nav) return;
  const fn = () => nav.classList.toggle('scrolled', window.scrollY > 40);
  window.addEventListener('scroll', fn, { passive: true });
  fn();
}

// ============================================================
// SSE — real-time profile update notifications
// ============================================================
function initProfileStream() {
  // Close any existing SSE connection (e.g. after persona switch)
  if (STATE.sseSource) {
    STATE.sseSource.close();
    STATE.sseSource = null;
  }

  const url = `${CFG.apiBase}/profile/stream/${STATE.activeUser}`;
  const source = new EventSource(url);
  STATE.sseSource = source;

  source.addEventListener('profile_updated', (e) => {
    let reason = 'updated';
    try { reason = JSON.parse(e.data).reason ?? 'updated'; } catch (_) {}
    console.log(`[SSE] profile_updated reason=${reason} user=${STATE.activeUser}`);
    if (STATE.refetchTimer) { clearTimeout(STATE.refetchTimer); STATE.refetchTimer = null; }
    fetchTrails();
  });

  source.addEventListener('heartbeat', () => { /* keep-alive */ });

  source.onerror = () => {
    console.debug('[SSE] connection lost — EventSource will reconnect automatically');
  };

  console.log(`[SSE] stream opened for user=${STATE.activeUser}`);
}

// ============================================================
// PERSONA SWITCHER — dropdown
// ============================================================

/**
 * Fetch persona metadata and build a compact dropdown in #persona-switcher.
 *
 * Structure rendered:
 *   <div class="persona-switcher">          ← #persona-switcher (from HTML)
 *     <button class="persona-trigger">      ← shows active persona name
 *     <div class="persona-dropdown">        ← hidden until .open on wrapper
 *       <div class="persona-option" …>      ← one per persona
 */
async function initPersonaSwitcher() {
  const container = document.getElementById('persona-switcher');
  if (!container) return;

  let personas = [];
  try {
    personas = await apiFetch('/demo/personas');
    STATE.personas = personas;
  } catch (_) {
    personas = CFG.demoUsers.map(uid => ({
      user_id:      uid,
      display_name: uid.replace('demo_', '').replace(/_/g, ' ')
                       .replace(/\b\w/g, c => c.toUpperCase()),
      genres: [],
    }));
    STATE.personas = personas;
  }

  _buildPersonaDropdown(container, personas);

  // Close dropdown when clicking outside
  document.addEventListener('click', (e) => {
    if (!container.contains(e.target)) {
      const p = container.querySelector('.persona-dropdown');
      if (p) p.style.display = 'none';
      const t = container.querySelector('[aria-expanded]');
      if (t) t.setAttribute('aria-expanded', 'false');
    }
  });
}

function _activePersona() {
  return STATE.personas.find(p => p.user_id === STATE.activeUser) ?? null;
}

function _personaInitial(displayName) {
  return (displayName ?? '?')[0].toUpperCase();
}

// SVG icons matching the Tailwind UI sample exactly
const _CHEVRON_SVG = `<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" class="size-4 self-center text-gray-400">
  <path d="M5.22 10.22a.75.75 0 0 1 1.06 0L8 11.94l1.72-1.72a.75.75 0 1 1 1.06 1.06l-2.25 2.25a.75.75 0 0 1-1.06 0l-2.25-2.25a.75.75 0 0 1 0-1.06ZM10.78 5.78a.75.75 0 0 1-1.06 0L8 4.06 6.28 5.78a.75.75 0 0 1-1.06-1.06l2.25-2.25a.75.75 0 0 1 1.06 0l2.25 2.25a.75.75 0 0 1 0 1.06Z" clip-rule="evenodd" fill-rule="evenodd"/>
</svg>`;
const _CHECK_SVG = `<svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" class="size-5">
  <path d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z" clip-rule="evenodd" fill-rule="evenodd"/>
</svg>`;

function _buildPersonaDropdown(container, personas) {
  container.innerHTML = '';

  const active = _activePersona() ?? personas[0];

  // ── Trigger — styled like the sample's button, dark theme ──
  const trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.setAttribute('aria-haspopup', 'listbox');
  trigger.setAttribute('aria-expanded', 'false');
  // grid layout mirrors the sample: selected content left, chevron right
  trigger.className = [
    'grid', 'grid-cols-1', 'cursor-default', 'rounded-md',
    'bg-[#1a1a1a]', 'py-1.5', 'pr-2', 'pl-3',
    'text-left', 'text-white',
    'border', 'border-white/10',
    'hover:bg-[#252525]', 'hover:border-white/20',
    'focus-visible:outline-none', 'focus-visible:ring-2', 'focus-visible:ring-[#E50914]',
    'transition-colors', 'text-sm', 'min-w-[150px]',
  ].join(' ');

  // Selected content area (avatar + name) — col-start-1 row-start-1
  const selectedContent = document.createElement('span');
  selectedContent.className = 'col-start-1 row-start-1 flex items-center gap-2 pr-6';

  const trigAvatar = document.createElement('span');
  trigAvatar.className = 'inline-flex size-5 shrink-0 items-center justify-center rounded bg-[#E50914] text-[10px] font-bold text-white';
  trigAvatar.textContent = _personaInitial(active?.display_name);

  const trigName = document.createElement('span');
  trigName.className = 'block truncate font-semibold';
  trigName.textContent = active?.display_name ?? 'Persona';

  selectedContent.appendChild(trigAvatar);
  selectedContent.appendChild(trigName);

  // Chevron — col-start-1 row-start-1 justify-self-end (overlays the content area)
  const chevronWrap = document.createElement('span');
  chevronWrap.className = 'col-start-1 row-start-1 flex justify-end pr-1 pointer-events-none';
  chevronWrap.innerHTML = _CHEVRON_SVG;

  trigger.appendChild(selectedContent);
  trigger.appendChild(chevronWrap);

  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    const isOpen = panel.style.display === 'none';
    panel.style.display = isOpen ? 'block' : 'none';
    trigger.setAttribute('aria-expanded', String(isOpen));
  });

  // ── Popover panel — anchored directly below trigger ─────────
  const panel = document.createElement('div');
  panel.className = [
    'persona-dropdown',
    'absolute', 'top-full', 'right-0', 'z-[9999]',
    'mt-1', 'min-w-[220px]', 'max-h-60', 'overflow-auto',
    'rounded-md', 'bg-[#1f1f1f]',
    'py-1', 'text-base', 'text-white',
    'shadow-[0_8px_32px_rgba(0,0,0,0.7)]',
    'border', 'border-white/10',
    'sm:text-sm',
  ].join(' ');
  // Hide via inline style — beats any Tailwind utility specificity conflicts.
  // The trigger click handler toggles this directly instead of toggling a
  // parent CSS class, so there is no dependency on cascade ordering.
  panel.style.display = 'none';
  panel.setAttribute('role', 'listbox');

  for (const p of personas) {
    const isActive = p.user_id === STATE.activeUser;

    // Option row — mirrors sample's group/option pattern
    const option = document.createElement('div');
    option.className = [
      'persona-option',
      'relative', 'block', 'cursor-default', 'select-none',
      'py-2', 'pr-9', 'pl-3',
      isActive ? 'bg-[#E50914]/10 text-[#ff8080]' : 'text-gray-300',
      'hover:bg-white/5',
    ].join(' ');
    option.dataset.userId = p.user_id;
    option.setAttribute('role', 'option');
    option.setAttribute('aria-selected', String(isActive));

    // Inner row: avatar + text body
    const inner = document.createElement('div');
    inner.className = 'flex items-start gap-2.5';

    const optAvatar = document.createElement('span');
    optAvatar.className = 'mt-0.5 inline-flex size-5 shrink-0 items-center justify-center rounded bg-[#E50914] text-[10px] font-bold text-white';
    optAvatar.textContent = _personaInitial(p.display_name);

    const body = document.createElement('div');
    body.className = 'flex flex-col gap-1 overflow-hidden';

    const optName = document.createElement('span');
    optName.className = [
      'block', 'truncate',
      isActive ? 'font-semibold' : 'font-normal',
    ].join(' ');
    optName.textContent = p.display_name;

    // Genre tags
    const tagsEl = document.createElement('div');
    tagsEl.className = 'flex flex-wrap gap-1';
    for (const g of (p.genres ?? []).slice(0, 4)) {
      const tag = document.createElement('span');
      tag.className = [
        'inline-block', 'rounded', 'px-1.5', 'py-0',
        'text-[9px]', 'font-semibold', 'leading-4',
        isActive ? 'bg-[#E50914]/20 text-[#ffaaaa]' : 'bg-white/[0.08] text-gray-500',
      ].join(' ');
      tag.textContent = g;
      tagsEl.appendChild(tag);
    }

    body.appendChild(optName);
    body.appendChild(tagsEl);
    inner.appendChild(optAvatar);
    inner.appendChild(body);
    option.appendChild(inner);

    // Checkmark — absolute right, visible only when active
    const checkWrap = document.createElement('span');
    checkWrap.className = [
      'absolute', 'inset-y-0', 'right-0',
      'flex', 'items-center', 'pr-4',
      'text-[#E50914]',
      isActive ? '' : 'hidden',
    ].join(' ');
    checkWrap.innerHTML = _CHECK_SVG;
    option.appendChild(checkWrap);

    option.addEventListener('click', (e) => {
      e.stopPropagation();
      panel.style.display = 'none';
      trigger.setAttribute('aria-expanded', 'false');
      switchPersona(p.user_id);
    });

    panel.appendChild(option);
  }

  container.appendChild(trigger);
  container.appendChild(panel);
}

/** Re-render the dropdown after a persona switch to update active states */
function _updatePersonaDropdownActive() {
  const container = document.getElementById('persona-switcher');
  if (!container || !STATE.personas.length) return;
  // Re-render from scratch — the new panel starts with display:none so it is
  // always closed after a switch regardless of prior state.
  _buildPersonaDropdown(container, STATE.personas);
}

/**
 * Switch to a different demo persona:
 *  1. Marks the target option as switching (visual feedback)
 *  2. Resets that persona's session + restores their user_vector
 *  3. Clears all local session STATE
 *  4. Updates active user + dropdown trigger label
 *  5. Reopens SSE stream for the new user
 *  6. Re-fetches trails
 */
async function switchPersona(userId) {
  if (userId === STATE.activeUser) return;

  // Dim target option while switching
  const targetOpt = document.querySelector(`.persona-option[data-user-id="${userId}"]`);
  if (targetOpt) targetOpt.classList.add('switching');

  try {
    await apiFetch('/demo/reset', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId }),
    });

    STATE.activeUser = userId;
    _clearSessionState();
    _updatePersonaDropdownActive();
    updateHUD();
    initProfileStream();
    await fetchTrails();
  } catch (e) {
    console.error('switchPersona:', e);
    if (targetOpt) targetOpt.classList.remove('switching');
  }
}

// ============================================================
// Boot
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
  initDOM();
  initNavScroll();

  // Nav wiring
  D.btnReset?.addEventListener('click', resetDemo);
  D.btnAdmin?.addEventListener('click', window.openAdmin);
  D.toastClose?.addEventListener('click', dismissToast);
  D.adminClose?.addEventListener('click', window.closeAdmin);
  D.adminOverlay?.addEventListener('click', e => { if (e.target === D.adminOverlay) window.closeAdmin(); });

  // Alt+A keyboard shortcut for admin
  window.addEventListener('keydown', e => {
    if (e.altKey && e.key.toLowerCase() === 'a') window.openAdmin();
  });

  // Boost form wiring
  D.movieSearch?.addEventListener('input', onMovieSearch);
  D.clearSelection?.addEventListener('click', clearMovieSelection);
  D.btnBoost?.addEventListener('click', submitBoost);
  if (D.boostScore && D.boostScoreVal) {
    D.boostScore.addEventListener('input', () => {
      D.boostScoreVal.textContent = parseFloat(D.boostScore.value).toFixed(1) + '×';
    });
  }

  // Persona switcher — fetch metadata and render pills
  await initPersonaSwitcher();

  // Open SSE stream for real-time profile update notifications
  initProfileStream();

  await fetchTrails();
});
