const form = document.querySelector('#upload-form');
const input = document.querySelector('#file-input');
const zone = document.querySelector('#dropzone');
const selectedFile = document.querySelector('#selected-file');
const nameNode = document.querySelector('#file-name');
const preview = document.querySelector('#preview');
const analyzeButton = document.querySelector('#analyze-button');
const resultsSection = document.querySelector('#results-section');
const resultContent = document.querySelector('#result-content');
const jobMessage = document.querySelector('#job-message');
const downloadReport = document.querySelector('#download-report');
const uploadHelp = document.querySelector('#upload-help');
let currentFile = null;
let previewUrl = null;
let hiveConfigured = false;
let maxUploadMb = 100;
let deletedAfterAnalysis = true;

const iconFor = (type) => ({ image: '▧', video: '▷', audio: '♫' }[type] || '•');
const modalityName = (type) => ({ image: 'Photo check', video: 'Video check', audio: 'Voice check' }[type] || 'Media check');
function displayVerdict(value) {
  return ({
    likely_synthetic: 'AI-like signs were found',
    likely_authentic: 'Fewer AI-like signs were found',
    inconclusive: 'Inconclusive',
    unavailable: 'No result is available',
  }[value] || 'Inconclusive');
}
const percent = (value) => Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—';
const precisePercent = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '—';
const supportedExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']);

function setUploadHelp(message, invalid = false) {
  uploadHelp.textContent = message;
  uploadHelp.classList.toggle('invalid', invalid);
}

function validateFile(file) {
  const extension = `.${file.name.split('.').pop().toLowerCase()}`;
  if (!supportedExtensions.has(extension)) {
    return 'That file type is not supported. Choose an image, video, or audio file listed above.';
  }
  if (file.size > maxUploadMb * 1024 * 1024) {
    return `This file is larger than the ${maxUploadMb} MB limit. Choose a smaller file.`;
  }
  return '';
}

function setFile(file) {
  if (!file) return;
  const validationMessage = validateFile(file);
  if (validationMessage) {
    clearFile();
    setUploadHelp(validationMessage, true);
    return;
  }
  setUploadHelp(`Up to ${maxUploadMb} MB. Your report may remain in recent activity after the uploaded file is removed.`);
  currentFile = file;
  resultsSection.hidden = true;
  resultContent.hidden = true;
  jobMessage.hidden = true;
  document.querySelector('#hive-consent').checked = false;
  updateHiveConsentVisibility();
  document.querySelector('#drop-title').textContent = 'Your file is ready';
  document.querySelector('#drop-subtitle').textContent = 'Choose another file to replace it';
  nameNode.textContent = `${file.name} · ${formatBytes(file.size)}`;
  selectedFile.hidden = false;
  updateAnalyzeButton();
  preview.replaceChildren();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  if (file.type.startsWith('image/')) {
    const image = document.createElement('img'); image.src = previewUrl; image.alt = 'Preview of the selected photo'; preview.append(image); preview.hidden = false;
  } else if (file.type.startsWith('video/')) {
    const video = document.createElement('video'); video.src = previewUrl; video.controls = true; video.preload = 'metadata'; video.setAttribute('aria-label', 'Preview of the selected video'); preview.append(video); preview.hidden = false;
  } else if (file.type.startsWith('audio/')) {
    const audio = document.createElement('audio'); audio.src = previewUrl; audio.controls = true; audio.setAttribute('aria-label', 'Listen to the selected audio'); preview.append(audio); preview.hidden = false;
  } else preview.hidden = true;
}

function clearFile() {
  currentFile = null;
  input.value = '';
  selectedFile.hidden = true;
  preview.hidden = true;
  preview.replaceChildren();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  updateAnalyzeButton();
  updateHiveConsentVisibility();
  document.querySelector('#drop-title').textContent = 'Drop a file here';
  document.querySelector('#drop-subtitle').textContent = 'or choose one from your device';
}

function updateAnalyzeButton() {
  analyzeButton.disabled = !currentFile;
}

function updateHiveConsentVisibility() {
  const fileName = currentFile?.name.toLowerCase() || '';
  const hiveSupported = /\.(jpg|jpeg|png|webp|gif|mp4|mov|avi|mkv|webm)$/.test(fileName);
  const wrap = document.querySelector('#hive-consent-wrap');
  wrap.hidden = !hiveConfigured || !hiveSupported;
  if (wrap.hidden) document.querySelector('#hive-consent').checked = false;
}

async function loadAppSettings() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) return;
    const health = await response.json();
    hiveConfigured = Boolean(health.models?.hive?.configured);
    maxUploadMb = Number.isFinite(health.max_upload_mb) ? health.max_upload_mb : 100;
    deletedAfterAnalysis = health.upload_policy?.deleted_after_analysis !== false;
    document.querySelector('#privacy-copy').textContent = deletedAfterAnalysis
      ? 'Uploaded files are removed after the check'
      : 'Uploaded files are kept after the check';
    setUploadHelp(deletedAfterAnalysis
      ? `Choose one file up to ${maxUploadMb} MB. The uploaded file is removed after analysis; its report can remain in recent activity.`
      : `Choose one file up to ${maxUploadMb} MB. Uploaded files are retained by this server’s current settings.`);
    updateHiveConsentVisibility();
    updateAnalyzeButton();
  } catch (_) {
    // Upload validation still uses the documented default if the health route is unavailable.
  }
}

loadAppSettings();

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

input.addEventListener('change', () => setFile(input.files[0]));
document.querySelector('#clear-file').addEventListener('click', clearFile);
['dragenter', 'dragover'].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove('dragging'); }));
zone.addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!currentFile) return;
  resultContent.hidden = true;
  downloadReport.hidden = true;
  showMessage('Sending your file for a check…');
  resultsSection.hidden = false;
  analyzeButton.disabled = true;
  analyzeButton.querySelector('span:first-child').textContent = 'Checking…';
  try {
    const body = new FormData();
    body.append('upload', currentFile);
    body.append('external_provider_consent', document.querySelector('#hive-consent').checked ? 'true' : 'false');
    const response = await fetch('/api/jobs', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'The upload could not be started.');
    await pollJob(payload.id);
  } catch (error) {
    showMessage(error.message || 'We couldn’t submit this file. Try again or choose a supported file.', true);
  } finally {
    analyzeButton.disabled = !currentFile;
    analyzeButton.querySelector('span:first-child').textContent = 'Check this file';
    refreshHistory();
  }
});

async function pollJob(jobId) {
  const started = Date.now();
  while (Date.now() - started < 10 * 60 * 1000) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || 'We couldn’t read the check status.');
    if (job.status === 'completed') {
      renderResult(job.result, job.id);
      return;
    }
    if (job.status === 'failed') throw new Error('We couldn’t finish the check. Try another supported file or review the server window for details.');
    const message = job.status === 'processing'
      ? 'The checks are running. The first file can take longer while model files load.'
      : 'Your file is waiting for the checks to start.';
    showMessage(message);
    await new Promise((resolve) => setTimeout(resolve, 1400));
  }
  throw new Error('This check is taking longer than expected. Look in Recent checks for its status.');
}

function showMessage(message, isError = false) {
  jobMessage.textContent = message;
  jobMessage.classList.toggle('error', isError);
  jobMessage.hidden = false;
}

function expectedChecks(mediaType) {
  return { image: 1, video: 2, audio: 1 }[mediaType] || 1;
}

function provenanceText(provenance) {
  const descriptions = {
    no_credentials: ['No signed origin details found', 'No signed origin details were found in this file. That is common and does not tell us whether the media is AI-made or camera-captured.'],
    trusted: ['Signed details checked', 'The file contains signed origin details that passed the current check. This supports the recorded history but does not prove everything shown in the media.'],
    valid: ['Signed details found', 'The file contains signed origin details that passed a validity check. The signer’s identity may not be confirmed.'],
    invalid: ['A problem with signed details', 'The file contains signed origin details, but the check found a problem. This does not by itself mean the file is AI-made.'],
    unverified: ['Signed details not confirmed', 'Signed origin details were found, but TruthLen could not confirm their status.'],
    sdk_missing: ['Origin check not installed', 'The optional signed-origin checker is not installed. This does not change the media checks.'],
    unavailable: ['Origin details could not be checked', 'TruthLen could not read signed origin details from this file. This does not mean the file is AI-made.'],
  };
  return descriptions[provenance?.status] || ['Origin details not checked', provenance?.summary || 'No signed origin information was returned.'];
}

function makeDetails(label, content) {
  const details = document.createElement('details');
  details.className = 'card-details';
  const summary = document.createElement('summary');
  summary.textContent = label;
  const body = document.createElement('p');
  body.textContent = content;
  details.append(summary, body);
  return details;
}

function shortClassLabel(label, modality) {
  const normalized = String(label || '').toLowerCase().replaceAll('_', ' ').replaceAll('-', ' ');
  if (normalized.includes('deepfake') || normalized.includes('face swap')) return 'Deepfake';
  if (normalized.includes('real') || normalized.includes('authentic') || normalized.includes('genuine')) return 'Real';
  if (normalized.includes('ai') || normalized.includes('generated') || normalized.includes('synthetic')) return modality === 'audio' ? 'AI voice' : 'AI';
  if (normalized.includes('fake')) return modality === 'audio' ? 'Fake voice' : 'Fake';
  return label;
}

function renderHeroScores(result) {
  const host = document.querySelector('#hero-class-scores');
  const title = document.querySelector('#result-score-title');
  const modalities = (result.modalities || []).filter((item) => item.available);
  const imageOrAudio = result.media_type === 'image' || result.media_type === 'audio';
  title.textContent = result.media_type === 'image' ? 'AI · Real · Deepfake'
    : result.media_type === 'audio' ? 'AI voice · Real voice'
      : 'Video and sound scores';
  host.replaceChildren();

  const appendScore = (label, score, context = '') => {
    const item = document.createElement('div');
    item.className = 'hero-score-item';
    const name = document.createElement('span'); name.className = 'hero-score-name'; name.textContent = label;
    const value = document.createElement('strong'); value.className = 'hero-score-value'; value.textContent = precisePercent(score);
    const bar = document.createElement('span'); bar.className = 'hero-score-bar';
    const fill = document.createElement('i'); fill.style.width = percent(Math.max(0, Math.min(1, score))); bar.append(fill);
    item.append(name, value, bar);
    if (context) { const note = document.createElement('small'); note.textContent = context; item.append(note); }
    host.append(item);
  };

  if (imageOrAudio) {
    const modality = modalities.find((item) => item.modality === result.media_type);
    if (modality?.class_scores?.length) {
      const entries = modality.class_scores.map((entry) => ({ label: shortClassLabel(entry.label, result.media_type), score: entry.score }));
      if (result.media_type === 'image') {
        const order = ['AI', 'Real', 'Deepfake'];
        entries.sort((a, b) => (order.indexOf(a.label) < 0 ? 99 : order.indexOf(a.label)) - (order.indexOf(b.label) < 0 ? 99 : order.indexOf(b.label)));
      }
      entries.forEach((entry) => appendScore(entry.label, entry.score));
    } else if (modality && Number.isFinite(modality.synthetic_score)) {
      appendScore(result.media_type === 'image' ? 'AI' : 'AI voice', modality.synthetic_score);
      appendScore(result.media_type === 'image' ? 'Real' : 'Real voice', 1 - modality.synthetic_score);
    } else {
      const message = document.createElement('p'); message.className = 'hero-score-empty'; message.textContent = 'No model class scores were returned for this file.'; host.append(message);
    }
    return;
  }

  modalities.forEach((modality) => {
    const group = document.createElement('section'); group.className = 'hero-score-group';
    const groupTitle = document.createElement('h4');
    groupTitle.textContent = modality.modality === 'video' ? 'Video frames' : 'Sound';
    group.append(groupTitle);
    if (modality.class_scores?.length) {
      modality.class_scores.forEach((entry) => {
        const row = document.createElement('div'); row.className = 'hero-score-mini';
        const label = document.createElement('span'); label.textContent = shortClassLabel(entry.label, modality.modality);
        const value = document.createElement('strong'); value.textContent = precisePercent(entry.score);
        row.append(label, value); group.append(row);
      });
    } else if (Number.isFinite(modality.synthetic_score)) {
      const label = document.createElement('span'); label.textContent = modality.modality === 'video' ? 'AI-like frames' : 'AI voice';
      const value = document.createElement('strong'); value.textContent = precisePercent(modality.synthetic_score);
      const row = document.createElement('div'); row.className = 'hero-score-mini'; row.append(label, value); group.append(row);
      if (Number.isFinite(modality.manipulation_score)) {
        const edit = document.createElement('div'); edit.className = 'hero-score-mini';
        const editLabel = document.createElement('span'); editLabel.textContent = 'Deepfake';
        const editValue = document.createElement('strong'); editValue.textContent = precisePercent(modality.manipulation_score);
        edit.append(editLabel, editValue); group.append(edit);
      }
    }
    host.append(group);
  });
  if (!modalities.length) {
    const message = document.createElement('p'); message.className = 'hero-score-empty'; message.textContent = 'No model scores were returned for this file.'; host.append(message);
  }
}

function finalCaution(result, usable) {
  const comparisons = (result.comparisons || []).filter((item) => item.available);
  if (result.media_type === 'video') {
    const expected = expectedChecks(result.media_type);
    return usable.length < expected
      ? `Only ${usable.length} of ${expected} video checks returned a score. Results can be wrong; don’t rely on them alone.`
      : 'The video was checked using sampled frames and sound. These checks can be wrong and do not analyze motion. Don’t rely on this result alone.';
  }
  if (usable.length === 1 && comparisons.length) {
    return 'Only one main model is used for the overall result. A separate comparison may appear above, but it does not change the result. Model scores can be wrong; don’t rely on this result alone.';
  }
  if (usable.length === 1) {
    return `We ran only one model on this ${result.media_type === 'image' ? 'photo' : 'audio file'}. Its score can be wrong, including on real camera photos. Don’t rely on this result alone.`;
  }
  return 'Automated model scores can be wrong. Don’t rely on this result alone.';
}

function renderScoreCard(item, comparison = false) {
  const card = document.createElement('article');
  card.className = 'modality-card';
  const top = document.createElement('div');
  top.className = 'card-top';
  const title = document.createElement('h4');
  title.textContent = comparison ? (item.model || 'Additional model') : modalityName(item.modality);
  const tag = document.createElement('span');
  tag.className = `tag${item.available ? '' : ' unavailable'}`;
  tag.textContent = item.available ? 'CHECKED' : 'NOT AVAILABLE';
  top.append(title, tag);
  const metric = document.createElement('div');
  metric.className = 'metric';
  metric.textContent = item.available ? precisePercent(item.synthetic_score) : '—';
  const metricLabel = document.createElement('p');
  metricLabel.className = 'subtext';
  metricLabel.textContent = item.available ? 'Class scores · not verified probabilities' : 'This check could not return a score.';
  card.append(top, metric, metricLabel);
  if (item.available) {
    if (item.class_scores?.length) {
      const breakdown = document.createElement('div');
      breakdown.className = 'class-breakdown';
      breakdown.setAttribute('aria-label', 'Model class scores');
      item.class_scores.forEach((entry) => {
        const chip = document.createElement('span');
        chip.className = 'class-chip';
        const label = document.createElement('strong'); label.textContent = shortClassLabel(entry.label, item.modality);
        const value = document.createElement('span'); value.textContent = precisePercent(entry.score);
        chip.append(label, value); breakdown.append(chip);
      });
      card.append(breakdown);
    }
    const scores = (item.class_scores || []).map((entry) => `${entry.label}: ${precisePercent(entry.score)}`).join(' · ');
    const modelDetails = [scores, item.model ? `Model: ${item.model}` : '', item.provider ? `Provider: ${item.provider}` : ''].filter(Boolean).join('\n');
    if (modelDetails) card.append(makeDetails('Show model details', modelDetails));
    if (!comparison && Number.isFinite(item.manipulation_score)) {
      card.append(makeDetails('About the separate editing score', `Separate face-editing/deepfake class: ${precisePercent(item.manipulation_score)}. This is not added to the AI-generated score. Treat it cautiously because real camera files have triggered high scores in manual checks.`));
    }
  } else if (item.reason) {
    card.append(makeDetails('Why this check was unavailable', item.reason));
  }
  return card;
}

function renderResult(result, jobId) {
  jobMessage.hidden = true;
  resultContent.hidden = false;
  const usable = (result.modalities || []).filter((item) => item.available && Number.isFinite(item.synthetic_score));
  renderHeroScores(result);

  const cards = document.querySelector('#modality-grid');
  cards.replaceChildren(...(result.modalities || []).map((item) => renderScoreCard(item)));

  const comparisons = result.comparisons || [];
  const comparisonPanel = document.querySelector('#comparison-panel');
  const comparisonGrid = document.querySelector('#comparison-grid');
  comparisonGrid.replaceChildren(...comparisons.map((item) => renderScoreCard(item, true)));
  comparisonPanel.hidden = comparisons.length === 0;
  comparisonPanel.open = false;

  const provenance = result.provenance || {};
  const [provenanceLabel, provenanceSummary] = provenanceText(provenance);
  const provenanceNode = document.querySelector('#provenance-label');
  provenanceNode.textContent = provenanceLabel;
  provenanceNode.className = `tag${['trusted', 'valid'].includes(provenance.status) ? '' : ' unavailable'}`;
  document.querySelector('#provenance-summary').textContent = provenanceSummary;

  const explanationList = document.querySelector('#explanations');
  explanationList.replaceChildren();
  (result.explanations || []).forEach((explanation) => {
    const li = document.createElement('li');
    li.textContent = explanation;
    explanationList.append(li);
  });
  const timestamps = document.querySelector('#timestamps');
  timestamps.replaceChildren();
  const video = (result.modalities || []).find((item) => item.modality === 'video');
  const generatedFrames = video?.details?.highest_ai_generated_timestamps || video?.details?.suspicious_timestamps || [];
  const manipulationFrames = video?.details?.highest_manipulation_timestamps || [];
  if (generatedFrames.length || manipulationFrames.length) {
    if (generatedFrames.length) {
      const label = document.createElement('span'); label.className = 'timestamp-chip'; label.textContent = 'Frames with the highest AI-like scores'; timestamps.append(label);
      generatedFrames.forEach((frame) => { const chip = document.createElement('span'); chip.className = 'timestamp-chip'; chip.textContent = `${formatTime(frame.timestamp_seconds)} · ${percent(frame.synthetic_score)}`; timestamps.append(chip); });
    }
    if (manipulationFrames.length) {
      const label = document.createElement('span'); label.className = 'timestamp-chip'; label.textContent = 'Frames with the highest separate editing scores'; timestamps.append(label);
      manipulationFrames.forEach((frame) => { const chip = document.createElement('span'); chip.className = 'timestamp-chip'; chip.textContent = `${formatTime(frame.timestamp_seconds)} · ${percent(frame.manipulation_score)}`; timestamps.append(chip); });
    }
    timestamps.hidden = false;
  } else timestamps.hidden = true;
  document.querySelector('#result-notice').textContent = result.notice || 'Automated screening only. Model scores can be wrong and are not proof of where media came from.';
  document.querySelector('#final-caution').textContent = finalCaution(result, usable);
  downloadReport.href = `/api/jobs/${jobId}/report`;
  downloadReport.hidden = false;
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function formatTime(seconds = 0) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

async function refreshHistory() {
  const list = document.querySelector('#history-list');
  try {
    const response = await fetch('/api/jobs');
    if (!response.ok) throw new Error();
    const jobs = await response.json();
    list.replaceChildren();
    if (!jobs.length) {
      const empty = document.createElement('p'); empty.className = 'empty-state'; empty.textContent = 'Your recent checks will appear here.'; list.append(empty); return;
    }
    jobs.forEach((job) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = `history-item${job.status === 'failed' ? ' failed' : ''}`;
      const icon = document.createElement('span'); icon.className = 'history-icon'; icon.setAttribute('aria-hidden', 'true'); icon.textContent = iconFor(job.media_type);
      const filename = document.createElement('span'); filename.className = 'history-name'; filename.textContent = job.filename;
      const meta = document.createElement('span'); meta.className = 'history-meta'; meta.textContent = new Date(job.created_at).toLocaleString();
      const historyResult = job.result;
      const outcome = document.createElement('span'); outcome.className = 'history-verdict'; outcome.textContent = job.status === 'completed' ? displayVerdict(historyResult?.fusion?.verdict) : ({ failed: 'Could not finish', processing: 'Checking…', queued: 'Waiting' }[job.status] || 'In progress');
      row.append(icon, filename, meta, outcome);
      row.addEventListener('click', () => {
        resultsSection.hidden = false;
        if (job.result) renderResult(job.result, job.id);
        else { resultContent.hidden = true; showMessage(job.status === 'failed' ? 'This check could not finish. Choose the row again after refreshing if you need its latest status.' : 'This check has not finished yet.', job.status === 'failed'); }
      });
      list.append(row);
    });
  } catch {
    const empty = document.createElement('p'); empty.className = 'empty-state'; empty.textContent = 'Recent checks could not be loaded. Make sure the app is running.'; list.replaceChildren(empty);
  }
}

document.querySelector('#refresh-history').addEventListener('click', refreshHistory);
refreshHistory();
