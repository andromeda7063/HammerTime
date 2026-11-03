/* Optional team-logo decoration for the archived Marina Bay replay card. */
(() => {
  const scriptUrl = document.currentScript?.src || new URL('driver_assets.js', location.href).href;
  const manifestUrl = new URL('data/marina_bay_driver_assets.json', scriptUrl);
  let manifestPromise;

  async function load() {
    if (!manifestPromise) {
      manifestPromise = fetch(manifestUrl, { cache: 'no-store' }).then(async (response) => {
        if (!response.ok) throw new Error(`Driver asset manifest request failed (HTTP ${response.status})`);
        const manifest = await response.json();
        if (manifest.schema_version !== 1 || manifest.event_race_id !== '2023-Singapore-R') {
          throw new Error('Driver asset manifest has an unexpected schema or event.');
        }
        const entries = new Map((manifest.drivers || []).map((driver) => [String(driver.number), driver]));
        entries.teamColors = manifest.team_colors_2023 || {};
        return entries;
      }).catch((error) => {
        manifestPromise = null;
        throw error;
      });
    }
    return manifestPromise;
  }

  function renderSummary(target, driver, label, manifest) {
    if (!(target instanceof HTMLElement)) return;
    const driverKey = driver ? String(driver.number) : 'all';
    const entry = driver && manifest ? manifest.get(driverKey) : null;
    const shouldShowLogo = Boolean(entry?.replay_available && entry.team_logo_asset && entry.team);
    if (target.dataset.driverAssetKey === driverKey && target.dataset.driverAssetLabel === label
        && (!shouldShowLogo || target.querySelector('img'))) return;
    target.replaceChildren();
    if (shouldShowLogo) {
      const logo = document.createElement('img');
      logo.src = new URL(entry.team_logo_asset, scriptUrl).href;
      logo.alt = `${entry.team} team logo`;
      logo.decoding = 'async';
      logo.style.cssText = 'height:1.7em;max-width:3.1em;object-fit:contain;vertical-align:middle;margin-inline-end:.5em;border-radius:2px';
      target.append(logo);
    }
    const text = document.createElement('span');
    text.textContent = label;
    target.append(text);
    target.dataset.driverAssetKey = driverKey;
    target.dataset.driverAssetLabel = label;
  }

  function officialTeamColorFor(driverOrTeam, manifest) {
    if (!manifest) return null;
    let entry = null;
    let team = null;
    if (driverOrTeam && typeof driverOrTeam === 'object') {
      team = driverOrTeam.team || null;
      if (driverOrTeam.number != null) entry = manifest.get(String(driverOrTeam.number)) || null;
    } else if (driverOrTeam != null) {
      const key = String(driverOrTeam);
      entry = manifest.get(key) || null;
      if (!entry) team = key;
    }
    team ||= entry?.team || null;
    return (team && manifest.teamColors?.[team]?.official_hex)
      || entry?.official_team_color_hex
      || null;
  }

  window.HammerTimeDriverAssets = {
    load,
    renderSummary,
    officialTeamColorFor,
    teamColorFor: officialTeamColorFor,
  };
})();
