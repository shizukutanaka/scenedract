// SceneDract desktop app (Electron shell)
// Loads the self-contained SceneDract.html in its own window.
// All app data stays in this app's own local storage on the user's PC.
//
// Updates: the shell checks GitHub Releases (public API, no token) at most
// once a day, shows an in-app banner when a newer version exists, and can
// download the matching package for this OS/arch with a progress bar.
// Applying the update is deliberately manual and the same on every OS:
// when the download finishes, the shell reveals the downloaded zip and
// the banner guides the user to unzip it and replace the app. There is
// no self-replacement helper — an unverified swap mechanism would be a
// large failure surface for very little user value.
const path = require('path');
const fs = require('fs');
const https = require('https');

const IS_ELECTRON = !!(process.versions && process.versions.electron);
const electron = IS_ELECTRON ? require('electron') : null;
const { app, BrowserWindow, shell, Menu, dialog } = electron || {};

/* ---------------- Pure helpers (also used by the unit tests) ---------------- */

const UPDATE_REPO = 'shizukutanaka/scenedract';
const UPDATE_API_URL = 'https://api.github.com/repos/' + UPDATE_REPO + '/releases/latest';
const UPDATE_CHANNEL_HOST = 'update.scenedract.invalid';
const CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;

function parseVersion(v) {
  if (typeof v !== 'string') return null;
  const core = v.trim().replace(/^v/i, '').split('-')[0].split('+')[0];
  const parts = core.split('.');
  if (!parts.length || parts.some((p) => !/^\d+$/.test(p))) return null;
  return parts.map((p) => parseInt(p, 10));
}

// Returns 1 when a > b, -1 when a < b, 0 when equal, null when unparseable.
function compareVersions(a, b) {
  const pa = Array.isArray(a) ? a : parseVersion(a);
  const pb = Array.isArray(b) ? b : parseVersion(b);
  if (!pa || !pb) return null;
  const n = Math.max(pa.length, pb.length);
  for (let i = 0; i < n; i++) {
    const d = (pa[i] || 0) - (pb[i] || 0);
    if (d !== 0) return d > 0 ? 1 : -1;
  }
  return 0;
}

function platformTag(platform, arch) {
  const p = { win32: 'windows', darwin: 'macos', linux: 'linux' }[platform];
  if (!p) return null;
  if (arch !== 'x64' && arch !== 'arm64') return null;
  return p + '-' + arch;
}

// Pick the release asset matching this OS/arch, e.g.
// SceneDract-v1.2.0-windows-x64.zip. Exact name wins; suffix match is a
// fallback so a different version naming still resolves.
function pickAsset(assets, version, platform, arch) {
  const tag = platformTag(platform, arch);
  if (!tag || !Array.isArray(assets)) return null;
  const exact = 'SceneDract-v' + version + '-' + tag + '.zip';
  let fallback = null;
  for (const a of assets) {
    if (!a || typeof a.name !== 'string') continue;
    if (a.name === exact) return a;
    if (a.name.endsWith('-' + tag + '.zip')) fallback = a;
  }
  return fallback;
}

function makeNotesExcerpt(body, maxLen) {
  if (typeof body !== 'string') return '';
  const limit = maxLen || 260;
  const text = body
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[*_`>]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  return text.length > limit ? text.slice(0, limit) + '…' : text;
}

module.exports = {
  parseVersion,
  compareVersions,
  platformTag,
  pickAsset,
  makeNotesExcerpt,
  UPDATE_API_URL,
  UPDATE_CHANNEL_HOST,
  CHECK_INTERVAL_MS,
};

/* ---------------- Shell + updater (Electron only) ---------------- */

if (IS_ELECTRON) {

let mainWindow = null;
let pageReady = false;
let startupCheckDone = false;
let latestRelease = null;   // { version, notes, releaseUrl, asset }
let downloadedFile = null;  // path of the downloaded zip
let downloadBusy = false;

const L = (ja, en) => ((app.getLocale() || '').toLowerCase().startsWith('ja') ? ja : en);

function stateFilePath() { return path.join(app.getPath('userData'), 'update-state.json'); }

function readUpdateState() {
  try { return JSON.parse(fs.readFileSync(stateFilePath(), 'utf8')) || {}; }
  catch (e) { return {}; }
}

function writeUpdateState(patch) {
  try {
    const next = Object.assign(readUpdateState(), patch);
    fs.mkdirSync(path.dirname(stateFilePath()), { recursive: true });
    fs.writeFileSync(stateFilePath(), JSON.stringify(next));
  } catch (e) { /* state is an optimization only; ignore failures */ }
}

function sendToPage(method, payload) {
  if (!mainWindow || mainWindow.isDestroyed() || !pageReady) return;
  const code = 'window.__sdUpdate && window.__sdUpdate.' + method +
    '(' + JSON.stringify(payload === undefined ? null : payload) + ')';
  mainWindow.webContents.executeJavaScript(code).catch(() => {});
}

function fetchJson(url, redirects) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: {
        'User-Agent': 'SceneDract/' + app.getVersion(),
        'Accept': 'application/vnd.github+json',
      },
    }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        if ((redirects || 0) >= 5) return reject(new Error('Too many redirects'));
        return resolve(fetchJson(res.headers.location, (redirects || 0) + 1));
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error('HTTP ' + res.statusCode));
      }
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
        catch (e) { reject(e); }
      });
      res.on('error', reject);
    });
    req.setTimeout(20000, () => req.destroy(new Error('Request timed out')));
    req.on('error', reject);
  });
}

function checkForUpdates(manual) {
  const state = readUpdateState();
  if (!manual && state.lastCheckAt && Date.now() - state.lastCheckAt < CHECK_INTERVAL_MS) return;
  writeUpdateState({ lastCheckAt: Date.now() });
  fetchJson(UPDATE_API_URL).then((rel) => {
    const version = String((rel && rel.tag_name) || '').replace(/^v/i, '');
    if (!version || compareVersions(version, app.getVersion()) === null) {
      if (manual) sendToPage('toast', 'checkfailed');
      return;
    }
    if (compareVersions(version, app.getVersion()) <= 0) {
      latestRelease = null;
      if (manual) sendToPage('toast', 'uptodate');
      return;
    }
    const asset = pickAsset(rel.assets, version, process.platform, process.arch);
    latestRelease = {
      version,
      notes: makeNotesExcerpt(rel.body),
      releaseUrl: rel.html_url || ('https://github.com/' + UPDATE_REPO + '/releases'),
      asset: asset ? { name: asset.name, url: asset.browser_download_url } : null,
    };
    if (!manual && state.dismissedVersion === version) return;
    sendToPage('show', {
      version: latestRelease.version,
      notes: latestRelease.notes,
      releaseUrl: latestRelease.releaseUrl,
      canDownload: !!latestRelease.asset,
      platform: process.platform,
    });
  }).catch(() => {
    // Offline, rate-limited, or no published release yet: stay quiet.
    if (manual) sendToPage('toast', 'checkfailed');
  });
}

function downloadFile(url, destPath, onProgress) {
  return new Promise((resolve, reject) => {
    const follow = (u, redirects) => {
      const req = https.get(u, { headers: { 'User-Agent': 'SceneDract/' + app.getVersion() } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          if (redirects >= 5) return reject(new Error('Too many redirects'));
          return follow(res.headers.location, redirects + 1);
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error('HTTP ' + res.statusCode));
        }
        const total = parseInt(res.headers['content-length'] || '0', 10);
        let received = 0;
        let lastPct = -1;
        const out = fs.createWriteStream(destPath);
        res.on('data', (c) => {
          received += c.length;
          if (total > 0) {
            const pct = Math.floor((received / total) * 100);
            if (pct !== lastPct) { lastPct = pct; onProgress(pct); }
          }
        });
        res.pipe(out);
        out.on('finish', () => out.close(() => resolve(received)));
        out.on('error', reject);
        res.on('error', reject);
      });
      req.setTimeout(60000, () => req.destroy(new Error('Download timed out')));
      req.on('error', reject);
    };
    follow(url, 0);
  });
}

function startDownload() {
  if (downloadBusy || !latestRelease || !latestRelease.asset) return;
  downloadBusy = true;
  const updatesDir = path.join(app.getPath('userData'), 'updates');
  const finalPath = path.join(updatesDir, latestRelease.asset.name);
  const partPath = finalPath + '.part';
  sendToPage('progress', 0);
  fs.mkdirSync(updatesDir, { recursive: true });
  downloadFile(latestRelease.asset.url, partPath, (pct) => sendToPage('progress', pct))
    .then(() => {
      fs.renameSync(partPath, finalPath);
      downloadedFile = finalPath;
      // Same on every OS: reveal the zip; the banner explains that the
      // user unzips it and replaces the current app folder by hand.
      sendToPage('state', 'downloaded');
    })
    .catch(() => {
      try { fs.unlinkSync(partPath); } catch (e) { /* nothing to clean */ }
      sendToPage('state', 'error');
    })
    .then(() => { downloadBusy = false; });
}

function handleUpdateAction(action) {
  if (action === 'download') startDownload();
  else if (action === 'later') {
    if (latestRelease) writeUpdateState({ dismissedVersion: latestRelease.version });
  } else if (action === 'reveal') {
    if (downloadedFile) shell.showItemInFolder(downloadedFile);
  }
}

function buildMenu() {
  const checkItem = {
    label: L('更新を確認', 'Check for Updates'),
    click: () => checkForUpdates(true),
  };
  const aboutItem = {
    label: L('SceneDract について', 'About SceneDract'),
    click: () => {
      dialog.showMessageBox({
        type: 'info',
        title: 'SceneDract',
        message: 'SceneDract ' + app.getVersion(),
        detail: L(
          '文章で頼んだシーンを、検証つきの設計図（Scene IR）と Blender / Unity 向けコードに変えるアプリです。',
          'Describe a scene in words; get a verified blueprint (Scene IR) and code for Blender and Unity.'),
      }).catch(() => {});
    },
  };
  const template = [];
  if (process.platform === 'darwin') {
    template.push({
      label: 'SceneDract',
      submenu: [aboutItem, checkItem, { type: 'separator' }, { role: 'quit' }],
    });
  }
  template.push({
    label: L('ファイル', 'File'),
    submenu: [process.platform === 'darwin' ? { role: 'close' } : { role: 'quit' }],
  });
  template.push({
    label: L('編集', 'Edit'),
    submenu: [
      { role: 'undo' }, { role: 'redo' }, { type: 'separator' },
      { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' },
    ],
  });
  template.push({
    label: L('表示', 'View'),
    submenu: [
      { role: 'reload' }, { type: 'separator' },
      { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
      { type: 'separator' }, { role: 'togglefullscreen' },
    ],
  });
  template.push({
    label: L('ヘルプ', 'Help'),
    submenu: process.platform === 'darwin' ? [checkItem] : [checkItem, aboutItem],
  });
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 360,
    minHeight: 560,
    title: 'SceneDract',
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow = win;
  pageReady = false;
  win.webContents.on('did-finish-load', () => {
    pageReady = true;
    if (!startupCheckDone) {
      startupCheckDone = true;
      setTimeout(() => checkForUpdates(false), 4000);
    }
  });
  // Open external links in the user's browser, not inside the app.
  // Banner buttons signal the shell through a reserved host name instead.
  win.webContents.setWindowOpenHandler(({ url }) => {
    try {
      const u = new URL(url);
      if (u.hostname === UPDATE_CHANNEL_HOST) {
        handleUpdateAction(u.pathname.replace(/^\//, ''));
        return { action: 'deny' };
      }
    } catch (e) { /* not a URL we manage */ }
    if (url.startsWith('http://') || url.startsWith('https://')) {
      shell.openExternal(url);
      return { action: 'deny' };
    }
    return { action: 'allow' };
  });
  win.loadFile(path.join(__dirname, 'SceneDract.html'));
}

app.whenReady().then(() => {
  buildMenu();
  createWindow();
});
app.on('window-all-closed', () => app.quit());
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

}
