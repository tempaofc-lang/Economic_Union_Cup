/* ============================================================================
   手语视界 · 拼音解译调试台 —— 前端逻辑（纯原生 JS，无构建步骤）

   主流程：
     输入拼音串 → POST /api/recognize（模拟 YOLO）
                → 按 0.75 阈值过滤低置信度 token（与 Gradio 版一致）
                → POST /api/agent/stream（SSE 流式）
                → 边收边渲染大字翻译，最后用 final 事件补齐卡片/反问/明细

   两种解译策略：
     llm   —— LLM 直接读原始拼音解码（默认）
     local —— 先用封闭词表过滤，再交给 LLM 组句
   ========================================================================== */

const CONF_THRESHOLD = 0.75;      // 与 app.py 的默认业务阈值保持一致

const state = {
  sessionId: null,
  vocab: [],
  vocabByHanzi: new Map(),
  lastTokens: [],
  session: [],
  strategy: 'llm',
  rawBuffer: '',
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ 工具函数 */
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function setStatus(text, kind = '') {
  const el = $('status');
  el.textContent = text;
  el.className = 'status' + (kind ? ' ' + kind : '');
}

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* 忽略 */ }
    throw new Error(detail);
  }
  return res.json();
}

/* -------------------------------------------------- 从流式 JSON 里抠出字段 */
// 模型输出的是 JSON 文本，边流边用正则抠出 translation / answer 两个字段的**当前值**，
// 即使字符串还没闭合也能先渲染出来，形成打字机效果。final 事件到达后以结构化数据为准。
const RE_TRANSLATION = /"translation"\s*:\s*"((?:[^"\\]|\\.)*)/;
const RE_ANSWER = /"answer"\s*:\s*"((?:[^"\\]|\\.)*)/;

function unescapePartial(s) {
  try {
    return JSON.parse('"' + s + '"');
  } catch (_) {
    return s.replace(/\\n/g, '\n').replace(/\\"/g, '"').replace(/\\\\/g, '\\');
  }
}

function extractField(text, re) {
  const m = text.match(re);
  return m ? unescapePartial(m[1]) : '';
}

/* ------------------------------------------------------------ SSE 流式读取 */
async function streamAgent(words, handlers) {
  const res = await fetch('/api/agent/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ words, session_id: state.sessionId, strategy: state.strategy }),
  });
  if (!res.ok || !res.body) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* 忽略 */ }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      const line = frame.split('\n').find((l) => l.startsWith('data: '));
      if (!line) continue;
      const payload = line.slice(6).trim();
      if (payload === '[DONE]') return;

      let event;
      try { event = JSON.parse(payload); } catch (_) { continue; }
      handlers(event);
    }
  }
}

/* ------------------------------------------------------------------ 初始化 */
async function boot() {
  try {
    const health = await api('/api/health');
    $('health').innerHTML = [
      `<span class="chip chip-${health.llm_available ? 'ok' : 'bad'}">${health.llm_available ? 'LLM 已接入' : 'LLM 不可用'}</span>`,
      `<span class="chip chip-muted">${esc(health.model)}</span>`,
      `<span class="chip chip-muted">思考模式 ${health.thinking ? '开' : '关'}</span>`,
      `<span class="chip chip-muted">知识库 ${health.kb_items} 条</span>`,
    ].join('');
    if (health.default_strategy) {
      document.querySelectorAll('input[name="strategy"]').forEach((el) => {
        el.checked = (el.value === health.default_strategy);
      });
      state.strategy = health.default_strategy;
      updateStrategyNote();
    }
  } catch (err) {
    $('health').innerHTML = `<span class="chip chip-bad">后端未连接：${esc(err.message)}</span>`;
  }

  try {
    const vocab = await api('/api/vocab');
    state.vocab = vocab.classes;
    vocab.classes.forEach((c) => state.vocabByHanzi.set(c.hanzi, c.pinyin));
    renderVocab(vocab);
  } catch (err) {
    $('vocab').innerHTML = `<span class="muted">词表加载失败：${esc(err.message)}</span>`;
  }
}

function renderVocab(vocab) {
  const chips = vocab.classes
    .filter((c) => c.hanzi !== vocab.confirm_hanzi)
    .map((c) => `<span class="vocab-chip" data-pinyin="${esc(c.pinyin)}">${esc(c.hanzi)}<small>${esc(c.pinyin)}</small></span>`);
  chips.push(`<span class="vocab-chip confirm" data-pinyin="${esc(vocab.confirm_pinyin)}">✋ ${esc(vocab.confirm_hanzi)}（控制信号）</span>`);
  $('vocab').innerHTML = chips.join('');

  $('vocab').querySelectorAll('.vocab-chip').forEach((el) => {
    el.addEventListener('click', () => {
      const input = $('pinyin-input');
      input.value = (input.value.trim() + ' ' + el.dataset.pinyin).trim();
      input.focus();
    });
  });
}

function updateStrategyNote() {
  $('strategy-note').textContent = state.strategy === 'llm'
    ? 'LLM 直接读原始拼音解码：可纠正错拼、按上下文消歧、推断词表外的常用词；本地查表仅作交叉校验。'
    : '先用封闭词表（17 类）硬过滤，再交给 LLM 组句：结果更稳但会丢弃词表外的拼音。用它来对比两种策略的差异。';
}

/* ------------------------------------------------------------ 渲染识别结果 */
function renderTokens(tokens, warning) {
  state.lastTokens = tokens;
  $('mock-badge').hidden = false;

  if (!tokens.length) {
    $('tokens').innerHTML = '<div class="empty">没有识别到任何 token</div>';
  } else {
    $('tokens').innerHTML = tokens.map((t) => {
      const pct = Math.round(t.conf * 100);
      const low = t.conf < CONF_THRESHOLD;
      return `
        <div class="token">
          <span class="token-word">${esc(t.word)}</span>
          <span class="token-bar ${low ? 'low' : ''}"><i style="width:${pct}%"></i></span>
          <span class="token-conf">${(t.conf).toFixed(2)}</span>
          <span class="token-box">[${t.bbox.join(', ')}]</span>
          ${low ? '<span class="token-flag">低于阈值·将被过滤</span>' : ''}
        </div>`;
    }).join('');
  }

  const warn = $('mock-warning');
  if (warning) { warn.hidden = false; warn.textContent = '⚠️ ' + warning; }
  else { warn.hidden = true; }
}

/* ---------------------------------------------------------- 流式骨架与增量 */
function renderStreamSkeleton() {
  $('result-meta').innerHTML = `<span class="pill pill-live">生成中…</span>`;
  $('result-body').innerHTML = `
    <div class="block-title">🗣️ 手语原意中文翻译（第一人称）</div>
    <div class="translation typing" id="live-tr"></div>
    <div class="block-title">🤖 政务导办回答</div>
    <div class="answer" id="live-an"></div>`;
  $('card-mapping').hidden = true;
  $('card-tools').hidden = true;
  $('card-raw').hidden = false;
  $('raw-body').textContent = '';
  $('stream-dot').hidden = false;
}

function updateStreamLive() {
  const tr = extractField(state.rawBuffer, RE_TRANSLATION);
  const an = extractField(state.rawBuffer, RE_ANSWER);
  const trEl = $('live-tr');
  const anEl = $('live-an');
  if (trEl) trEl.textContent = tr;
  if (anEl) anEl.textContent = an;
}

function endStreaming() {
  $('stream-dot').hidden = true;
  const trEl = $('live-tr');
  if (trEl) trEl.classList.remove('typing');
}

/* --------------------------------------------------------------- 渲染结果 */
const SRC_LABEL = { vocab: '词表命中', inferred: 'LLM 推断', ignored: '判为噪声', local: '本地查表' };

function renderResult(r) {
  const inferred = (r.mapping || []).filter((m) => m.source === 'inferred').length;
  const ignored = (r.mapping || []).filter((m) => m.source === 'ignored').length;

  const pills = [
    `<span class="pill ${r.mode === 'llm' ? 'pill-llm' : 'pill-rule'}">${r.mode === 'llm' ? 'LLM 解译' : '规则兜底'}</span>`,
    `<span class="pill">策略 ${r.strategy === 'llm' ? 'LLM 主导' : '本地查表'}</span>`,
    `<span class="pill">置信度 ${(r.confidence ?? 0).toFixed(2)}</span>`,
    `<span class="pill">耗时 ${r.elapsed_ms} ms</span>`,
    `<span class="pill">${esc(r.session_id)}</span>`,
  ];
  if (inferred) pills.push(`<span class="pill" style="background:#FFFBEB;color:#B45309;border-color:#FDE68A">${inferred} 个 LLM 推断</span>`);
  if (ignored) pills.push(`<span class="pill" style="background:#F1F5F9;color:#64748B">${ignored} 个判为噪声</span>`);
  $('result-meta').innerHTML = pills.join('');

  const parts = [];
  if (r.translation) {
    parts.push(`<div class="block-title">🗣️ 手语原意中文翻译（第一人称）</div>
                <div class="translation">${esc(r.translation)}</div>`);
  }
  if (r.answer) {
    parts.push(`<div class="block-title">🤖 政务导办回答</div>
                <div class="answer">${esc(r.answer)}</div>`);
  }

  if (r.clarify) {
    const opts = (r.clarify.options || []).map(
      (o) => `<button class="clarify-opt" data-opt="${esc(o)}">${esc(o)}</button>`
    ).join('');
    parts.push(`<div class="block-title">🙋 Agent 主动反问（没有硬猜）</div>
      <div class="clarify">
        <div class="clarify-q">${esc(r.clarify.question)}</div>
        <div class="clarify-opts">${opts}</div>
      </div>`);
  }

  if (r.confirm_card) {
    const c = r.confirm_card;
    const materials = (c.materials || []).length
      ? `<ul>${c.materials.map((m) => `<li>${esc(m)}</li>`).join('')}</ul>`
      : '<span class="muted">未登记</span>';
    parts.push(`<div class="block-title">📋 办事卡片（数据来自本地政务知识库）</div>
      <div class="confirm-card">
        <h3>${esc(c.title)}</h3>
        <div class="kv"><span class="kv-k">窗口</span><span class="kv-v">${esc(c.window || '未登记')}</span></div>
        <div class="kv"><span class="kv-k">材料</span><span class="kv-v">${materials}</span></div>
        <div class="kv"><span class="kv-k">费用</span><span class="kv-v">${esc(c.fee || '未登记')}</span></div>
        ${c.duration ? `<div class="kv"><span class="kv-k">时长</span><span class="kv-v">${esc(c.duration)}</span></div>` : ''}
      </div>`);
  }

  if (r.error) {
    parts.push(`<div class="warn-note" style="margin-top:12px">⚠️ 内部提示：${esc(r.error)}</div>`);
  }

  $('result-body').innerHTML = parts.length
    ? parts.join('')
    : '<div class="empty big-empty">这次没有产出内容。</div>';

  $('result-body').querySelectorAll('.clarify-opt').forEach((btn) => {
    btn.addEventListener('click', () => {
      const hanzi = btn.dataset.opt;
      const pinyin = state.vocabByHanzi.get(hanzi) || hanzi;
      $('pinyin-input').value = pinyin;
      runAgentOnly();
    });
  });

  renderMapping(r.mapping || []);
  renderTools(r.tool_calls || []);
}

function renderMapping(rows) {
  const card = $('card-mapping');
  if (!rows.length) { card.hidden = true; return; }
  card.hidden = false;

  const inferred = rows.filter((m) => m.source === 'inferred').length;
  const conflicts = rows.filter((m) => m.agree === false).length;
  $('mapping-summary').innerHTML = [
    `<span class="src src-vocab">词表命中 ${rows.filter((m) => m.source === 'vocab').length}</span>`,
    inferred ? `<span class="src src-inferred">LLM 推断 ${inferred}</span>` : '',
    rows.some((m) => m.source === 'ignored') ? `<span class="src src-ignored">噪声 ${rows.filter((m) => m.source === 'ignored').length}</span>` : '',
    conflicts ? `<span class="src src-inferred">与本地不一致 ${conflicts}</span>` : '',
  ].join(' ');

  $('mapping-body').innerHTML = rows.map((m, i) => {
    const word = m.word || '<span class="muted">—</span>';
    const local = m.local_word || '<span class="muted">—</span>';
    const agree = m.agree === true ? '<span class="agree-yes">✓</span>'
      : m.agree === false ? '<span class="agree-no">✗ 不一致</span>'
        : '<span class="agree-na">—</span>';
    const note = m.note ? `<div class="note-cell">${esc(m.note)}</div>` : '';
    return `
      <tr>
        <td>${i + 1}</td>
        <td class="mono">${esc(m.raw)}</td>
        <td><strong>${word}</strong>${note}</td>
        <td><span class="src src-${esc(m.source)}">${esc(SRC_LABEL[m.source] || m.source)}</span></td>
        <td class="mono">${local}</td>
        <td>${agree}</td>
      </tr>`;
  }).join('');
}

function renderTools(calls) {
  const card = $('card-tools');
  if (!calls.length) { card.hidden = true; return; }
  card.hidden = false;
  $('tools-body').innerHTML = calls.map((t) => `
    <div class="tool">
      <span class="tool-name">${esc(t.name)}</span>
      <span class="${t.ok ? 'tool-ok' : 'tool-fail'}">${t.ok ? '✓ 命中' : '✗ 未命中'}</span>
      <span>${esc(t.result || '')}</span>
      <span class="tool-ms">${t.ms} ms</span>
    </div>`).join('');
}

/* --------------------------------------------------------------- 会话记录 */
function pushSession(r) {
  state.session.unshift(r);
  if (state.session.length > 20) state.session.pop();

  $('session-body').innerHTML = state.session.map((s) => `
    <div class="session-item">
      <div class="s-head">
        <span>${esc((s.raw_pinyin || []).join(' '))}</span>
        <span>${s.strategy === 'llm' ? 'LLM主导' : '本地查表'} · ${s.mode === 'llm' ? 'LLM' : '规则'} · ${s.elapsed_ms}ms</span>
      </div>
      <div class="s-text">${esc(s.translation || '（无译文）')}</div>
    </div>`).join('');
}

/* ------------------------------------------------------------------ 主流程 */
async function runFullFlow() {
  const text = $('pinyin-input').value.trim();
  if (!text) { setStatus('请先输入拼音串', 'err'); return; }

  const buttons = [$('btn-full'), $('btn-agent-only')];
  buttons.forEach((b) => (b.disabled = true));

  try {
    setStatus('① 调用 /api/recognize（模拟 YOLO 识别）…', 'running');
    const rec = await api('/api/recognize', {
      method: 'POST',
      body: JSON.stringify({ text, session_id: state.sessionId }),
    });
    state.sessionId = rec.session_id;
    renderTokens(rec.tokens, rec.warning);

    const kept = rec.tokens.filter((t) => t.conf >= CONF_THRESHOLD);
    const dropped = rec.tokens.length - kept.length;

    if (!kept.length) {
      setStatus(`② 全部 ${rec.tokens.length} 个 token 均低于阈值 ${CONF_THRESHOLD}，按设计不调用 Agent`, 'err');
      $('result-body').innerHTML = `<div class="empty big-empty">
        未检出达到置信度阈值的有效手语词。<br>模拟真实的"识别失败"分支：<strong>不猜测、不调用大模型</strong>。</div>`;
      $('result-meta').innerHTML = `<span class="pill pill-rule">已拦截</span>`;
      $('card-mapping').hidden = true;
      $('card-tools').hidden = true;
      $('card-raw').hidden = true;
      return;
    }

    setStatus(`② 保留 ${kept.length} 个、过滤 ${dropped} 个，正在流式解译…`, 'running');
    await streamAgent(kept.map((t) => t.word), handleEvents(kept.length, dropped));
  } catch (err) {
    endStreaming();
    setStatus('流程失败：' + err.message, 'err');
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}

async function runAgentOnly() {
  const text = $('pinyin-input').value.trim();
  if (!text) { setStatus('请先输入拼音串', 'err'); return; }

  $('btn-agent-only').disabled = true;
  try {
    setStatus('直接流式解译（跳过识别）…', 'running');
    const words = text.replace(/[,，、]/g, ' ').split(/\s+/).filter(Boolean);
    $('tokens').innerHTML = '<div class="empty">本次跳过了识别步骤，没有 token 数据</div>';
    $('mock-badge').hidden = true;
    $('mock-warning').hidden = true;
    await streamAgent(words, handleEvents(words.length, 0));
  } catch (err) {
    endStreaming();
    setStatus('解译失败：' + err.message, 'err');
  } finally {
    $('btn-agent-only').disabled = false;
  }
}

/* --------------------------------------------------------- SSE 事件处理器 */
function handleEvents(keptCount, dropped) {
  let finalData = null;

  return (event) => {
    if (event.type === 'meta') {
      state.sessionId = event.session_id;
      state.rawBuffer = '';
      renderStreamSkeleton();
      if (event.tool_calls && event.tool_calls.length) renderTools(event.tool_calls);
      setStatus(`③ 已接入 Agent，正在流式生成…（保留 ${keptCount} 个词，过滤 ${dropped} 个）`, 'running');
      return;
    }

    if (event.type === 'delta') {
      state.rawBuffer += event.text || '';
      $('raw-body').textContent = state.rawBuffer;
      updateStreamLive();
      return;
    }

    if (event.type === 'error') {
      setStatus('模型报错：' + event.message, 'err');
      return;
    }

    if (event.type === 'final') {
      finalData = event.data;
      endStreaming();
      renderResult(finalData);
      pushSession(finalData);
      setStatus(
        `④ 解译完成（${finalData.strategy === 'llm' ? 'LLM 主导' : '本地查表'} · `
        + `${finalData.mode === 'llm' ? 'LLM' : '规则兜底'} · ${finalData.elapsed_ms} ms）`,
        'ok'
      );
    }
  };
}

/* ------------------------------------------------------------------ 事件绑定 */
document.addEventListener('DOMContentLoaded', () => {
  boot();

  $('btn-full').addEventListener('click', runFullFlow);
  $('btn-agent-only').addEventListener('click', runAgentOnly);

  document.querySelectorAll('input[name="strategy"]').forEach((el) => {
    el.addEventListener('change', () => {
      state.strategy = el.value;
      updateStrategyNote();
      setStatus(`已切换解译策略：${state.strategy === 'llm' ? 'LLM 主导' : '本地查表过滤'}`, 'ok');
    });
  });

  $('btn-clear').addEventListener('click', () => {
    $('pinyin-input').value = '';
    $('tokens').innerHTML = '<div class="empty">点上面的按钮后，这里显示 YOLO 风格的 token（词 / 置信度 / 检测框）</div>';
    $('result-body').innerHTML = '<div class="empty big-empty">等待解译。左侧输入拼音后点「走完整流程」。</div>';
    $('result-meta').innerHTML = '';
    $('card-mapping').hidden = true;
    $('card-tools').hidden = true;
    $('card-raw').hidden = true;
    $('mock-badge').hidden = true;
    $('mock-warning').hidden = true;
    $('stream-dot').hidden = true;
    state.rawBuffer = '';
    setStatus('待命');
  });

  $('btn-reset-session').addEventListener('click', async () => {
    if (state.sessionId) {
      try { await api(`/api/session/${state.sessionId}/reset`, { method: 'POST' }); } catch (_) { /* 忽略 */ }
    }
    state.session = [];
    $('session-body').innerHTML = '<div class="empty">暂无记录</div>';
    setStatus('会话已清空', 'ok');
  });

  document.querySelectorAll('[data-example]').forEach((btn) => {
    btn.addEventListener('click', () => {
      $('pinyin-input').value = btn.dataset.example;
      $('pinyin-input').focus();
    });
  });

  $('pinyin-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      runFullFlow();
    }
  });
});
