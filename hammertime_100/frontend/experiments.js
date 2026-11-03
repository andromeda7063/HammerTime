(() => {
  const registry = document.querySelector('#registry');
  const familyFilter = document.querySelector('#family-filter');
  const statusFilter = document.querySelector('#status-filter');
  const groupFilter = document.querySelector('#group-filter');
  const sortFilter = document.querySelector('#sort-filter');
  const searchFilter = document.querySelector('#search-filter');
  const resultCount = document.querySelector('#result-count');
  const filterMessage = document.querySelector('#filter-message');
  const resetButton = document.querySelector('#reset-filters');
  const downloadButton = document.querySelector('#download-csv');
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (v, n = 5) => Number.isFinite(v) ? v.toFixed(n) : '—';
  const normalize = value => String(value ?? '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim().toLocaleLowerCase();
  const slug = value => String(value ?? '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  let groups = [];

  function recordId(group, run) {
    const key = `${group.id}\u001f${run.model}\u001f${run.variant}`;
    let a = 2166136261, b = 0x9e3779b9;
    for (let i = 0; i < key.length; i++) {
      const code = key.charCodeAt(i);
      a = Math.imul(a ^ code, 16777619);
      b = Math.imul(b ^ code, 0x5bd1e995);
    }
    const hash = `${(a >>> 0).toString(16).padStart(8, '0')}${(b >>> 0).toString(16).padStart(8, '0')}`;
    const modelSlug = slug(run.model).slice(0, 24).replace(/-+$/g, '') || 'model';
    return `run-${slug(group.id)}-${modelSlug}-${hash}`;
  }

  const circuitModelLabels = {
    legacy_input_legacy_graph: 'Legacy inputs + legacy graph',
    circuit_context_legacy_graph: 'Circuit context + legacy graph',
    circuit_context_physical_scalar: 'Circuit context + physical scalar',
    circuit_context_physical_v1: 'Circuit context + physical graph v1',
    circuit_context_self_loop_only: 'Circuit context + self-loop only',
  };

  function displayModel(group, run) {
    return group.id === 'openf1-hungary-circuit-screen' ? (circuitModelLabels[run.model] || run.model) : run.model;
  }

  function validateData(data) {
    const errors = [];
    const evidences = new Set(['held-out test', 'validation', 'pilot', 'failed', 'report-only']);
    if (data?.schema_version !== 1 || !Array.isArray(data.groups) || !data.groups.length) {
      throw new Error('Unsupported or empty registry data schema.');
    }
    const ids = new Set();
    const runIds = new Set();
    for (const g of data.groups) {
      if (!g || typeof g.id !== 'string' || !g.id || ids.has(g.id)) errors.push('group id is missing or duplicated');
      ids.add(g?.id);
      if (!evidences.has(g?.evidence)) errors.push(`${g?.id || 'unknown group'} has an unknown evidence class`);
      if (!Array.isArray(g?.runs) || !g.runs.length) errors.push(`${g?.id || 'unknown group'} has no run records`);
      if (!Array.isArray(g?.sources) || !g.sources.length || g.sources.some(s => typeof s.path !== 'string' || !/^[a-f0-9]{64}$/.test(s.sha256 || ''))) {
        errors.push(`${g?.id || 'unknown group'} has incomplete source provenance`);
      }
      for (const r of g?.runs || []) {
        if (![r.model, r.family, r.variant, r.status].every(v => typeof v === 'string' && v.length)) errors.push(`${g.id} has an incomplete run label`);
        const id = recordId(g, r);
        if (runIds.has(id)) errors.push(`${g.id}/${r.model} has a duplicate stable record id`);
        runIds.add(id);
        for (const key of ['mae_sec', 'rmse_sec', 'r2_score']) {
          if (r[key] !== null && r[key] !== undefined && (typeof r[key] !== 'number' || !Number.isFinite(r[key]))) errors.push(`${g.id}/${r.model} has a non-finite ${key}`);
        }
        if (g.evidence === 'failed') {
          if (['mae_sec', 'rmse_sec', 'r2_score', 'sample_count'].some(k => r[k] !== null && r[k] !== undefined)) errors.push(`${g.id}/${r.model} must not contain a score or sample count`);
        } else {
          if (!Number.isFinite(r.mae_sec) || !Number.isFinite(r.r2_score)) errors.push(`${g.id}/${r.model} is missing a finite MAE or R²`);
          if (r.sample_count !== null && r.sample_count !== undefined && (!Number.isInteger(r.sample_count) || r.sample_count < 0)) errors.push(`${g.id}/${r.model} has an invalid sample count`);
        }
        if (r.rmse_sec !== null && r.rmse_sec !== undefined && r.rmse_sec < 0) errors.push(`${g.id}/${r.model} has a negative RMSE`);
      }
    }
    if (errors.length) throw new Error(`Registry data validation failed: ${errors.slice(0, 3).join('; ')}.`);
  }

  function renderedRuns(g) {
    const family = familyFilter.value;
    const needle = normalize(searchFilter.value);
    const filtered = g.runs.filter(r => (family === 'all' || r.family === family) && (!needle || normalize([g.title, g.provider, g.evaluation, g.protocol, g.note, r.model, displayModel(g, r), r.variant, r.family].join(' ')).includes(needle)));
    if (sortFilter.value === 'source') return filtered;
    const key = sortFilter.value === 'mae-asc' ? 'mae_sec' : 'r2_score';
    const direction = sortFilter.value === 'mae-asc' ? 1 : -1;
    return filtered.map((run, index) => ({run, index})).sort((a, b) => {
      const av = a.run[key], bv = b.run[key];
      const aMissing = !Number.isFinite(av), bMissing = !Number.isFinite(bv);
      if (aMissing !== bMissing) return aMissing ? 1 : -1;
      if (aMissing) return a.index - b.index;
      const delta = (av - bv) * direction;
      return delta || a.index - b.index;
    }).map(item => item.run);
  }

  function renderGroup(g) {
    if ((groupFilter.value !== 'all' && groupFilter.value !== g.id) || (statusFilter.value !== 'all' && statusFilter.value !== g.evidence)) return '';
    const runs = renderedRuns(g);
    if (!runs.length) return '';
    const rows = runs.map(r => {
      const id = recordId(g, r);
      const href = `?group=${encodeURIComponent(g.id)}#${id}`;
      const label = displayModel(g, r);
      const isReadableLabel = label !== r.model;
      const sourceDetail = isReadableLabel ? `Source key: ${r.model} · ${r.variant}` : r.variant;
      const ariaLabel = `Open direct link to ${g.title}: ${label}; source key ${r.model}; ${r.variant}`;
      return `<tr id="${id}" class="ex-run-row" tabindex="-1">
      <td><a class="ex-model ex-record-link" data-source-key="${esc(r.model)}" href="${href}" aria-label="${esc(ariaLabel)}" title="Open this result in its evidence group">${esc(label)}</a><span class="ex-variant">${esc(sourceDetail)}</span></td>
      <td>${fmt(r.mae_sec)}</td><td>${fmt(r.rmse_sec)}</td><td>${fmt(r.r2_score)}</td>
      <td>${Number.isInteger(r.sample_count) ? r.sample_count.toLocaleString('en-US') : '—'}<span class="ex-variant">${esc(r.sample_unit)}</span></td>
      <td class="ex-status"><span class="ex-status-text" data-status="${esc(r.status)}">${esc(r.status)}</span>${r.status_note ? `<span class="ex-variant">${esc(r.status_note)}</span>` : ''}</td>
    </tr>`;
    }).join('');
    const sources = g.sources.map(s => `<code>${esc(s.path)} · SHA-256 ${esc(s.sha256)}</code>`).join('');
    const titleId = `group-${g.id}`;
    const hintId = `table-hint-${g.id}`;
    return `<article class="ex-group" aria-labelledby="${esc(titleId)}">
      <header class="ex-group-head"><h2 id="${esc(titleId)}">${esc(g.title)}</h2><div class="ex-meta"><span class="ex-badge" data-evidence="${esc(g.evidence)}">${esc(g.evidence)}</span> · ${esc(g.evaluation)}</div>
      <p class="ex-protocol">${esc(g.protocol)}</p><p class="ex-note">${esc(g.note)}</p></header>
      <p class="ex-table-hint" id="${esc(hintId)}">On small screens, scroll horizontally to inspect all columns. Focus the table region and use arrow keys.</p>
      <div class="ex-table-wrap" role="region" tabindex="0" aria-label="${esc(g.title)} result table" aria-describedby="${esc(hintId)}">
      <table class="ex-table"><caption>${esc(g.title)} model metrics. Results are comparable only within this evaluation group.</caption><thead><tr><th scope="col">Model / variant</th><th scope="col">MAE · s</th><th scope="col">RMSE · s</th><th scope="col">R²</th><th scope="col">Samples</th><th scope="col">Record status</th></tr></thead><tbody>${rows}</tbody></table></div>
      <details class="ex-sources"><summary>Source provenance (${g.sources.length} files)</summary>${sources}</details>
    </article>`;
  }

  function stateFromUrl() {
    filterMessage.hidden = true;
    const params = new URLSearchParams(location.search);
    const allowed = {
      family: new Set(['all', ...[...document.querySelector('#family-filter').options].map(o => o.value)]),
      evidence: new Set(['all', ...[...statusFilter.options].map(o => o.value)]),
      group: new Set(['all', ...groups.map(g => g.id)]),
      sort: new Set([...sortFilter.options].map(o => o.value)),
    };
    const bad = [];
    for (const [key, control, param] of [['family', familyFilter, 'family'], ['evidence', statusFilter, 'evidence'], ['group', groupFilter, 'group'], ['sort', sortFilter, 'sort']]) {
      const value = params.get(param);
      const fallback = param === 'sort' ? 'source' : 'all';
      if (value === null) control.value = fallback;
      else if (allowed[key].has(value)) control.value = value;
      else { control.value = fallback; bad.push(param); }
    }
    searchFilter.value = params.get('q') || '';
    if (bad.length) {
      filterMessage.hidden = false;
      filterMessage.textContent = `Unknown URL filter ${bad.join(', ')}; cleared it. Available choices are loaded from this registry.`;
      updateUrl(false);
    }
  }

  function updateUrl(push = true) {
    const url = new URL(location.href);
    for (const [key, value] of [['family', familyFilter.value], ['evidence', statusFilter.value], ['group', groupFilter.value], ['sort', sortFilter.value], ['q', searchFilter.value.trim()]]) {
      if (value && value !== 'all' && !(key === 'sort' && value === 'source')) url.searchParams.set(key, value); else url.searchParams.delete(key);
    }
    history[push ? 'pushState' : 'replaceState']({}, '', url);
  }

  function csvCell(value) {
    const s = value === null || value === undefined ? '' : String(value);
    return `"${s.replaceAll('"', '""')}"`;
  }

  function downloadCsv() {
    const cols = ['sort_within_group', 'evidence', 'group_id', 'group_title', 'provider', 'evaluation', 'protocol', 'model', 'family', 'variant', 'status', 'status_note', 'mae_sec', 'rmse_sec', 'r2_score', 'sample_count', 'sample_unit', 'source_paths', 'source_sha256'];
    const records = groups.flatMap(g => (groupFilter.value === 'all' || groupFilter.value === g.id) && (statusFilter.value === 'all' || statusFilter.value === g.evidence)
      ? renderedRuns(g).map(r => ({sort_within_group:sortFilter.value, evidence:g.evidence, group_id:g.id, group_title:g.title, provider:g.provider, evaluation:g.evaluation, protocol:g.protocol, ...r, source_paths:g.sources.map(s=>s.path).join(' | '), source_sha256:g.sources.map(s=>s.sha256).join(' | ')}))
      : []);
    const csv = [cols.join(','), ...records.map(r => cols.map(c => csvCell(r[c])).join(','))].join('\r\n');
    const href = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'}));
    const a = document.createElement('a'); a.href = href; a.download = 'hammertime-filtered-evidence.csv'; a.click();
    setTimeout(() => URL.revokeObjectURL(href), 60000);
  }

  function draw() {
    const html = groups.map(renderGroup).join('');
    const visibleGroups = (html.match(/class="ex-group"/g) || []).length;
    const visibleRows = groups.reduce((n, g) => n + ((groupFilter.value === 'all' || groupFilter.value === g.id) && (statusFilter.value === 'all' || statusFilter.value === g.evidence) ? renderedRuns(g).length : 0), 0);
    registry.innerHTML = html || '<p class="ex-empty">No records match these filters.</p>';
    markHashTarget();
    for (const region of registry.querySelectorAll('.ex-table-wrap')) {
      region.addEventListener('keydown', event => {
        if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
          event.preventDefault();
          const delta = event.key === 'ArrowRight' ? 80 : -80;
          region.scrollLeft = Math.max(0, Math.min(region.scrollWidth - region.clientWidth, region.scrollLeft + delta));
        }
      });
    }
    resultCount.textContent = `${visibleRows} records · ${visibleGroups} evidence groups`;
    downloadButton.disabled = visibleRows === 0;
  }

  function markHashTarget() {
    registry.querySelectorAll('.ex-run-row.is-deep-link-target').forEach(row => row.classList.remove('is-deep-link-target'));
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); }
    catch { return null; }
    if (!id) return null;
    const target = document.getElementById(id);
    if (!target || !target.matches('tr.ex-run-row') || !registry.contains(target)) return null;
    target.classList.add('is-deep-link-target');
    return target;
  }

  function focusHashRecord() {
    const target = markHashTarget();
    if (!target) return;
    target.scrollIntoView({block: 'center', inline: 'nearest'});
    target.focus({preventScroll: true});
  }

  function filterChanged(push = true) {
    filterMessage.hidden = true;
    updateUrl(push);
    draw();
  }

  fetch('data/experiments.json', {cache: 'no-store'})
    .then(response => { if (!response.ok) throw new Error(`Registry request failed (${response.status})`); return response.json(); })
    .then(data => {
      validateData(data);
      groups = data.groups;
      const families = [...new Set(groups.flatMap(g => g.runs.map(r => r.family)))].sort();
      for (const family of families) familyFilter.add(new Option(family, family));
      for (const evidence of [...new Set(groups.map(g => g.evidence))]) statusFilter.add(new Option(evidence, evidence));
      for (const g of groups) groupFilter.add(new Option(g.title, g.id));
      stateFromUrl();
      draw();
      focusHashRecord();
      for (const control of [familyFilter, statusFilter, groupFilter, sortFilter]) control.addEventListener('change', filterChanged);
      searchFilter.addEventListener('input', () => filterChanged(false));
      resetButton.addEventListener('click', () => {
        familyFilter.value = 'all';
        statusFilter.value = 'all';
        groupFilter.value = 'all';
        sortFilter.value = 'source';
        searchFilter.value = '';
        if (location.hash) {
          const url = new URL(location.href);
          url.hash = '';
          history.replaceState({}, '', url);
        }
        filterChanged();
      });
      downloadButton.addEventListener('click', downloadCsv);
      addEventListener('popstate', () => { stateFromUrl(); draw(); focusHashRecord(); });
      addEventListener('hashchange', focusHashRecord);
    })
    .catch(error => {
      registry.innerHTML = `<p class="ex-error" role="alert">Experiment registry unavailable or invalid. ${esc(error.message)} No fallback or demo scores are shown.</p>`;
      resultCount.textContent = 'Registry unavailable';
      downloadButton.disabled = true;
    });
})();
