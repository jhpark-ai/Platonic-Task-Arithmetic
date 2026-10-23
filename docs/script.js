'use strict';
const video = document.getElementById('explainer');
const motionButton = document.getElementById('motion-toggle');
const restartButton = document.getElementById('restart-video');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let inView = false;
let waitingForAnchor = Boolean(location.hash && location.hash !== "#");
let userPaused = false;
let allowMotion = !reducedMotion.matches;
video.muted = true;
video.controls = false;
motionButton.hidden = false;
restartButton.hidden = false;
function updateMotionButton() {
  motionButton.textContent = video.paused ? 'Play animation' : 'Pause animation';
  motionButton.setAttribute('aria-pressed', String(video.paused));
}
async function playVideo() { try { await video.play(); } catch { updateMotionButton(); } }
function syncPlayback() {
  if (inView && !waitingForAnchor && !document.hidden && allowMotion && !userPaused) playVideo();
  else video.pause();
}
video.addEventListener('play', updateMotionButton);
video.addEventListener('pause', updateMotionButton);
video.addEventListener('error', () => {
  video.controls = true;
  motionButton.hidden = true;
  restartButton.hidden = true;
});
motionButton.addEventListener('click', () => {
  if (video.paused) { userPaused = false; allowMotion = true; playVideo(); }
  else { userPaused = true; video.pause(); }
});
restartButton.addEventListener('click', () => {
  video.currentTime = 0;
  userPaused = false;
  allowMotion = true;
  playVideo();
});
reducedMotion.addEventListener('change', event => {
  allowMotion = !event.matches;
  syncPlayback();
});
document.addEventListener('visibilitychange', syncPlayback);
if ('IntersectionObserver' in window) {
  const playbackObserver = new IntersectionObserver(entries => {
    inView = entries[0].intersectionRatio >= 0.6;
    if (waitingForAnchor) {
      if (!inView) waitingForAnchor = false;
      video.pause();
      return;
    }
    syncPlayback();
  }, {threshold:[0,0.6,1]});
  playbackObserver.observe(video);
} else {
  // Preserve the opening scene until the visitor chooses to play.
  video.controls = true;
}
updateMotionButton();

function setTabs(group, selected) {
  group.forEach(tab => {
    const active = tab === selected;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  });
}
function wireTabs(group, activate) {
  group.forEach((tab, index) => {
    tab.addEventListener('click', () => { setTabs(group, tab); activate(tab); });
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % group.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + group.length) % group.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = group.length - 1;
      if (next !== undefined) { event.preventDefault(); setTabs(group, group[next]); activate(group[next]); group[next].focus(); }
    });
  });
}
const operatorTabs = [...document.querySelectorAll('[data-operation]')];
wireTabs(operatorTabs, tab => {
  const add = tab.dataset.operation === 'add';
  document.getElementById('operator-display').setAttribute('aria-labelledby', tab.id);
  document.getElementById('operator-equation').innerHTML = add ? '<span class="red">A<sub>task A</sub></span><span>+</span><span class="blue">A<sub>task B</sub></span>' : '<span>−λ</span><span class="red">A<sub>task A</sub></span>';
  document.getElementById('operator-explanation').textContent = add ? 'Add operators from two sources, then fold their sum into one target. The paper uses α = 0.5 for closed-form addition.' : 'Negate a source task’s operator and fold it into the target. The paper uses λ = 1 for the reported negation tables.';
});
const experiments = {
  addition: {
    title:'Two sources. Two tasks. One target.',
    description:'Composing operators from two distinct sources improves both tasks in one target model.',
    metrics:['Task A ↑','Task B ↑','CIFAR-100 control','ImageNet control'],
    methods:[
      {name:'Zero-shot',short:'Zero-shot',color:'#aab4c2',values:[60.98,60.98,75.82,72.33]},
      {name:'Single-A',short:'Single-A',color:'#dfae88',values:[80.56,59.86,75.04,71.01]},
      {name:'Single-B',short:'Single-B',color:'#92afd4',values:[59.45,73.85,74.88,70.51]},
      {name:'PTA, closed form',short:'PTA',color:'#c94b49',values:[79.08,72.60,74.01,69.09]}
    ],
    population:'Table 1 · Cross-model, any pair of families. 3,360 cells. Two distinct sources; target differs from both. Closed-form addition at α = 0.5. Controls are reported alongside the edited tasks.',
    takeaway:'<strong>Both tasks improve:</strong> +18.10 pp on Task A and +11.62 pp on Task B, from the rounded table values. Relative to the target’s own-descriptor reference, the paper reports 80% and 74% of the task gains retained.'
  },
  negation: {
    title:'Remove a task through another model.',
    description:'A descriptor from another model selectively removes the targeted task, with a substantially smaller change on held-out controls.',
    metrics:['Removed task ↓','CIFAR-100 control','ImageNet control'],
    methods:[
      {name:'Zero-shot',short:'Zero-shot',color:'#aab4c2',values:[60.98,75.82,72.33]},
      {name:'PTA, closed form',short:'PTA',color:'#c94b49',values:[25.76,71.27,65.67]}
    ],
    population:'Table 2 · Cross-model, any pair of families. 240 cells. Closed-form negation at λ = 1. Lower removed-task accuracy means stronger removal; higher control accuracy means better preservation.',
    takeaway:'<strong>−35.22 pp on the removed task,</strong> versus −4.55 pp on CIFAR-100 and −6.66 pp on ImageNet. The task-specific edit is obtained by scaling one stored operator.'
  },
  adapter: {
    title:'What does a richer edit buy?',
    description:'On the same 168 cross-model addition cells, the adapter improves task accuracy and preserves ImageNet better. Each edit requires its own optimization.',
    metrics:['Task A ↑','Task B ↑','CIFAR-100 control','ImageNet control'],
    methods:[
      {name:'PTA, closed form',short:'Closed form',color:'#c94b49',values:[79.05,72.65,73.89,68.93]},
      {name:'PTA, adapter',short:'Adapter',color:'#4c75bb',values:[79.50,76.70,74.00,72.15]}
    ],
    population:'Table 1 · Matched 168-cell subset selected for adapter evaluation. Closed form at α = 0.5; adapter at full descriptor strength.',
    takeaway:'<strong>+4.05 pp on Task B and +3.22 pp on ImageNet</strong> relative to the closed form on the matched subset. The closed form offers a reusable bank; the adapter trades a separate optimization for a richer, jointly fitted edit.'
  }
};
function renderExperiment(key) {
  const data = experiments[key];
  document.getElementById('experiment-title').textContent = data.title;
  document.getElementById('experiment-description').textContent = data.description;
  document.getElementById('result-population').textContent = data.population;
  document.getElementById('result-takeaway').innerHTML = data.takeaway;
  document.getElementById('result-chart').innerHTML = data.metrics.map((metric, index) => '<div class="metric-chart"><h4>' + metric + '</h4>' + data.methods.map(method => '<div class="bar-row"><span>' + method.short + '</span><div class="bar-track"><i style="--value:' + method.values[index] + '%;--bar:' + method.color + '"></i></div><b>' + method.values[index].toFixed(2) + '</b></div>').join('') + '<div class="chart-ticks" aria-hidden="true"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div></div>').join('');
  document.getElementById('chart-legend').innerHTML = data.methods.map(method => '<span><i style="--bar:' + method.color + '"></i>' + method.name + '</span>').join('');
  document.getElementById('result-table').innerHTML = '<table><caption class="small-note">Accuracy (%) · ' + (key === 'adapter' ? 'Matched 168-cell subset' : key === 'addition' ? '3,360 cross-model cells' : '240 cross-model cells') + '</caption><thead><tr><th scope="col">Method</th>' + data.metrics.map(metric => '<th scope="col">' + metric + '</th>').join('') + '</tr></thead><tbody>' + data.methods.map(method => '<tr><th scope="row">' + method.name + '</th>' + method.values.map(value => '<td>' + value.toFixed(2) + '</td>').join('') + '</tr>').join('') + '</tbody></table>';
}
const resultTabs = [...document.querySelectorAll('[data-result]')];
wireTabs(resultTabs, tab => {
  document.getElementById('experiment-panel').setAttribute('aria-labelledby', tab.id);
  renderExperiment(tab.dataset.result);
});
renderExperiment('addition');
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  const target = document.getElementById(button.dataset.copy);
  const old = button.textContent;
  try {
    await navigator.clipboard.writeText(target.textContent);
    button.textContent = 'Copied!';
    document.getElementById('copy-status').textContent = button.dataset.copy === 'bibtex' ? 'BibTeX copied to clipboard.' : 'Commands copied to clipboard.';
    setTimeout(() => { button.textContent = old; }, 2000);
  } catch {
    const range = document.createRange(); range.selectNodeContents(target);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    button.textContent = 'Selected — press ⌘C / Ctrl+C';
  }
}));
const chapters = [...document.querySelectorAll('.chapter-nav a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) chapters.forEach(link => link.classList.toggle('active', link.getAttribute('href') === '#' + entry.target.id));
    });
  }, {rootMargin:'-15% 0px -60% 0px'});
  chapters.forEach(link => observer.observe(document.querySelector(link.getAttribute('href'))));
}
