'use strict';

const scenarios = {
  pause: {state:'Collecting oak logs · task running',progress:'6 / 16 logs',environment:'Daylight · forest',dialogue:'“Gather sixteen oak logs for the cabin.”',utterance:'“Wait here for a moment.”',decision:'pause',skill:null,reason:'A temporary stop preserves the current task so it can be resumed later.'},
  continue: {state:'Collecting oak logs · task running',progress:'6 / 16 logs',environment:'Daylight · forest',dialogue:'“When I say stop, I mean stop counting aloud. Keep collecting.”',utterance:'“Stop.”',decision:'continue',skill:null,reason:'The recent dialogue makes “stop” about the spoken count. The collection task stays active.'},
  cancel: {state:'Collecting oak logs · task running',progress:'6 / 16 logs',environment:'Daylight · forest',dialogue:'“We were collecting wood for a cabin, but the plan has changed.”',utterance:'“Forget the cabin. Cancel that task.”',decision:'cancel',skill:null,reason:'The player abandons the goal. Cancel ends the current task instead of preserving it for resumption.'},
  replace: {state:'Collecting oak logs · task running',progress:'6 / 16 logs',environment:'Dusk · forest',dialogue:'“The cabin can wait. I’m heading back to our base.”',utterance:'“Come with me instead.”',decision:'replace_task',skill:'follow_player',reason:'A new actionable goal replaces the old one. The executor receives the follow_player skill.'}
};

const scenarioTabs = [...document.querySelectorAll('[data-scenario]')];
function selectScenario(tab, moveFocus = false) {
  const scenario = scenarios[tab.dataset.scenario];
  if (!scenario) return;
  scenarioTabs.forEach(item => {const selected = item === tab;item.setAttribute('aria-selected', String(selected));item.tabIndex = selected ? 0 : -1;});
  document.getElementById('scenario-panel').setAttribute('aria-labelledby', tab.id);
  document.getElementById('scenario-state').textContent = scenario.state;
  const details = document.querySelectorAll('#state-details b');
  details[0].textContent = scenario.progress; details[1].textContent = scenario.environment;
  document.getElementById('recent-dialogue').textContent = scenario.dialogue;
  document.getElementById('scenario-utterance').textContent = scenario.utterance;
  document.getElementById('scenario-reason').textContent = scenario.reason;
  const code = document.getElementById('scenario-decision');
  code.replaceChildren(document.createTextNode('{\n  '));
  function token(value, className) {const span = document.createElement('span');span.className = className;span.textContent = value;code.append(span);}
  token('"decision"','json-key');code.append(': ');token(JSON.stringify(scenario.decision),'json-value');code.append(',\n  ');
  token('"skill"','json-key');code.append(': ');token(JSON.stringify(scenario.skill), scenario.skill === null ? 'json-null' : 'json-value');code.append('\n}');
  if (moveFocus) tab.focus();
}
scenarioTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectScenario(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % scenarioTabs.length;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index + scenarioTabs.length - 1) % scenarioTabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = scenarioTabs.length - 1;
    if (next !== undefined) {event.preventDefault();selectScenario(scenarioTabs[next], true);}
  });
});

const modelNames = {qwen3_1_7b:'Qwen3-1.7B', qwen2_5_7b:'Qwen2.5-7B-Instruct'};
const metrics = [
  ['decision_accuracy','Strict decision accuracy','Invalid schema counts as incorrect'],
  ['joint_accuracy','Joint accuracy','Decision and skill both correct'],
  ['macro_f1','Macro F1','Balance across decision classes'],
  ['json_validity','Schema validity','Valid decision / skill contract'],
  ['skill_accuracy','Skill accuracy','All 60 gold replacement examples']
];
const statusLabels = {pending:'Pending run',running:'Run in progress',completed:'Run completed',failed:'Run failed'};
let resultData = {schema_version:1,models:{}};
let selectedModel = 'qwen3_1_7b';
const modelButtons = [...document.querySelectorAll('[data-result-model]')];
const validMetric = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
const formatMetric = value => validMetric(value) ? `${(value * 100).toFixed(2)}%` : 'Pending';
function safeUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return null;
  try {const url = new URL(value, window.location.href);return ['http:', 'https:'].includes(url.protocol) ? url.href : null;} catch {return null;}
}
function applyLink(element, value) {const url = safeUrl(value);element.hidden = !url;if(url) element.href = url;else element.removeAttribute('href');return Boolean(url);}
function renderKeywordBaseline() {
  const baseline = resultData.keyword_baseline || {};
  document.getElementById('keyword-decision').textContent = formatMetric(baseline.decision_accuracy);
  document.getElementById('keyword-joint').textContent = formatMetric(baseline.joint_accuracy);
  document.getElementById('keyword-f1').textContent = validMetric(baseline.macro_f1) ? baseline.macro_f1.toFixed(3) : 'Pending';
  const count = baseline.sample_count;
  [['keyword-decision-count', baseline.decision_correct], ['keyword-joint-count', baseline.joint_correct]].forEach(([id, correct]) => {
    document.getElementById(id).textContent = Number.isInteger(count) && count > 0 && Number.isInteger(correct) && correct >= 0 && correct <= count ? `${correct} / ${count} correct` : 'Awaiting artifact';
  });
}
function renderCards() {
  document.querySelectorAll('[data-model]').forEach(card => {
    const model = resultData.models?.[card.dataset.model] || {};
    const status = Object.hasOwn(statusLabels, model.status) ? model.status : 'pending';
    const badge = card.querySelector('[data-status]');badge.textContent = statusLabels[status];badge.dataset.state = status;
    const hasMetrics = metrics.some(([key]) => validMetric(model.baseline?.[key]) || validMetric(model.adapter?.[key]));
    card.querySelector('[data-evidence]').textContent = hasMetrics ? 'Evaluation data available' : (status === 'failed' ? 'Failure recorded' : 'Evaluation pending');
    const adapterAvailable = applyLink(card.querySelector('[data-adapter]'), model.adapter_url);
    card.querySelector('[data-download-pending]').hidden = adapterAvailable;
    applyLink(card.querySelector('[data-report]'), model.report_url);
    const error = card.querySelector('[data-error]');error.hidden = status !== 'failed' || typeof model.error !== 'string' || !model.error.trim();error.textContent = error.hidden ? '' : model.error;
  });
}
function renderDiagnostics(model) {
  document.getElementById('diagnostics-caption').textContent = `${modelNames[selectedModel]} exploratory baseline versus adapter diagnostics`;
  const body = document.getElementById('diagnostics-body');body.replaceChildren();
  [['decision_accuracy','Decision choice, ignoring skill-schema errors'], ['json_syntax_validity','JSON syntax validity']].forEach(([key, label]) => {
    const row = document.createElement('tr');const heading = document.createElement('th');heading.scope = 'row';heading.textContent = label;row.append(heading);
    ['baseline', 'adapter'].forEach(variant => {
      const value = model.diagnostics?.[variant]?.[key];const cell = document.createElement('td');cell.textContent = formatMetric(value);if(validMetric(value)) cell.className = 'measured';row.append(cell);
    });
    body.append(row);
  });
}
function renderResults() {
  const model = resultData.models?.[selectedModel] || {};
  modelButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.resultModel === selectedModel)));
  document.getElementById('results-caption').textContent = `${modelNames[selectedModel]} baseline versus adapter evaluation`;
  const sampleCount = model.sample_count;
  document.getElementById('sample-count').textContent = `Held-out samples: ${Number.isInteger(sampleCount) && sampleCount > 0 ? sampleCount.toLocaleString() : 'pending'}`;
  const body = document.getElementById('results-body');body.replaceChildren();
  metrics.forEach(([key, label, description]) => {
    const row = document.createElement('tr');const heading = document.createElement('th');heading.scope = 'row';heading.textContent = label;
    const detail = document.createElement('small');detail.textContent = description;heading.append(detail);row.append(heading);
    const baseline = model.baseline?.[key];const adapter = model.adapter?.[key];
    [baseline, adapter].forEach(value => {const cell = document.createElement('td');cell.textContent = formatMetric(value);if(validMetric(value)) cell.className = 'measured';row.append(cell);});
    const change = document.createElement('td');
    if(validMetric(baseline) && validMetric(adapter)) {const difference = (adapter - baseline) * 100;change.textContent = `${difference > 0 ? '+' : ''}${difference.toFixed(2)} pp`;change.className = difference > 0 ? 'delta-positive' : difference < 0 ? 'delta-negative' : '';}
    else change.textContent = '—';
    row.append(change);body.append(row);
  });
  renderDiagnostics(model);
}
modelButtons.forEach(button => button.addEventListener('click', () => {selectedModel = button.dataset.resultModel;renderResults();}));
async function loadResults() {
  const message = document.getElementById('results-load-state');
  try {
    const response = await fetch('results.json', {cache:'no-store'});
    if(!response.ok) throw new Error('Result data unavailable');
    const payload = await response.json();
    if(payload.schema_version !== 1 || !payload.models || typeof payload.models !== 'object' || Array.isArray(payload.models)) throw new Error('Unsupported result data');
    resultData = payload;
    const models = Object.values(payload.models);
    const hasMetrics = models.some(model => metrics.some(([key]) => validMetric(model?.baseline?.[key]) || validMetric(model?.adapter?.[key])));
    const hasFailure = models.some(model => model?.status === 'failed');
    message.textContent = hasMetrics ? 'Evaluation artifacts loaded. Missing measurements remain pending.' : hasFailure ? 'A run failure is recorded. No missing score is inferred.' : 'No model evaluation measurements have been published yet.';
  } catch {
    message.textContent = 'Result data is unavailable. Measurements remain pending.';
  }
  renderKeywordBaseline();renderCards();renderResults();
}
loadResults();
