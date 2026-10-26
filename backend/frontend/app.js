/* ============================================================================
   手语视界 · 拼音解译调试台 —— 前端逻辑（纯原生 JS，无构建步骤）

   流程：
     输入拼音串 → POST /api/recognize（模拟 YOLO）
                → 按 0.75 阈值过滤低置信度 token（与 Gradio 版一致）
                → POST /api/agent
                → 渲染译文 / 导办回答 / 办事卡片 / 反问 / 解码明细 / 工具轨迹
   ========================================================================== */

const CONF_THRESHOLD = 0.75;   // 与 app.py 的默认业务阈值保持一致

const state = {
  sessionId: null,
  vocab: [],
  vocabByHanzi: new Map(),
  lastTokens: [],
  session: [],
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
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* 忽略 */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ------------------------------------------------------------------ 初始化 */
async function boot() {
  try {
    const health = await api('/api/health');
    $('health').innerHTML = [
      `<span class="chip chip-${health.llm_available ? 'ok' : 'bad'}">${health.llm_available ? 'LLM 已接入' : 'LLM 不可用（规则兜底）'}</span>`,
      `<span class="chip chip-muted">${esc(health.model)}</span>`,
      `<span class="chip chip-muted">知识库 ${health.kb_items} 条</span>`,
      `<span class="chip chip-muted">v${esc(health.version)}</span>`,
    ].join('');
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
  chips.push(`<span class="vocab-chip confirm" data-pinyin="${esc(vocab.confirm_pinyin)}">✋ ${esc(vocab.confirm_hanzi)}（确认手势）</span>`);
  $('vocab').innerHTML = chips.join('');

  $('vocab').querySelectorAll('.vocab-chip').forEach((el) => {
    el.addEventListener('click', () => {
      const input = $('pinyin-input');
      input.value = (input.value.trim() + ' ' + el.dataset.pinyin).trim();
      input.focus();
    });
  });
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
  if (warning) {
    warn.hidden = false;
    warn.textContent = '⚠️ ' + warning;
  } else {
    warn.hidden = true;
  }
}

/* --------------------------------------------------------------- 渲染结果 */
function renderResult(r) {
  const pills = [
    `<span class="pill ${r.mode === 'llm' ? 'pill-llm' : 'pill-rule'}">${r.mode === 'llm' ? 'LLM 解译' : '规则兜底'}</span>`,
    `<span class="pill">置信度 ${(r.confidence ?? 0).toFixed(2)}</span>`,
    `<span class="pill">耗时 ${r.elapsed_ms} ms${r.cached ? '（缓存命中）' : ''}</span>`,
    `<span class="pill">${esc(r.session_id)}</span>`,
  ];
  if (r.unknown && r.unknown.length) {
    pills.push(`<span class="pill" style="background:#FFF1F2;color:#BE123C;border-color:#FECDD3">忽略 ${r.unknown.length} 个词表外拼音</span>`);
  }
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
  if (r.staff_reply) {
    parts.push(`<div class="block-title">💬 窗口工作人员回复</div>
                <div class="answer" style="background:#ECFDF5;border-color:#A7F3D0;color:#065F46">“${esc(r.staff_reply)}”</div>`);
  }

  $('result-body').innerHTML = parts.length
    ? parts.join('')
    : '<div class="empty big-empty">这次没有产出内容。</div>';

  // 反问选项 → 自动填入并重跑
  $('result-body').querySelectorAll('.clarify-opt').forEach((btn) => {
    btn.addEventListener('click', () => {
      const hanzi = btn.dataset.opt;
      const pinyin = state.vocabByHanzi.get(hanzi);
      $('pinyin-input').value = pinyin || hanzi;
      runAgentOnly();
    });
  });

  renderDecode(r.decode || []);
  renderTools(r.tool_calls || []);
}

const HIT_LABEL = { exact: '精确匹配', alias: '别名匹配', fuzzy: '模糊匹配', cjk: '已是中文', unknown: '词表外' };

function renderDecode(items) {
  const card = $('card-decode');
  if (!items.length) { card.hidden = true; return; }
  card.hidden = false;
  $('decode-body').innerHTML = items.map((d, i) => `
    <tr>
      <td>${i + 1}</td>
      <td class="mono">${esc(d.raw)}</td>
      <td>→</td>
      <td><strong>${esc(d.hanzi || '—')}</strong></td>
      <td class="hit-${esc(d.hit)}">${esc(HIT_LABEL[d.hit] || d.hit)}</td>
      <td class="mono">${(d.score ?? 0).toFixed(2)}</td>
    </tr>`).join('');
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
        <span>${s.mode === 'llm' ? 'LLM' : '规则'} · ${s.elapsed_ms}ms</span>
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
      setStatus(`② 全部 ${rec.tokens.length} 个 token 均低于阈值 ${CONF_THRESHOLD}，按设计不调用 Agent（对应 Gradio 版的"识别失败四联动"）`, 'err');
      $('result-body').innerHTML = `<div class="empty big-empty">
        未检出达到置信度阈值的有效手语词。<br>模拟真实的"识别失败"分支：<strong>不猜测、不调用大模型</strong>。</div>`;
      $('result-meta').innerHTML = `<span class="pill pill-rule">已拦截</span>`;
      $('card-decode').hidden = true;
      $('card-tools').hidden = true;
      return;
    }

    setStatus(`② 阈值过滤完成：保留 ${kept.length} 个、过滤 ${dropped} 个，正在调用 /api/agent…`, 'running');
    const result = await api('/api/agent', {
      method: 'POST',
      body: JSON.stringify({
        words: kept.map((t) => t.word),
        session_id: state.sessionId,
      }),
    });

    renderResult(result);
    pushSession(result);
    setStatus(`③ 解译完成（${result.mode === 'llm' ? 'LLM' : '规则兜底'}，${result.elapsed_ms} ms${result.cached ? '，缓存命中' : ''}）`, 'ok');
  } catch (err) {
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
    setStatus('直接调用 /api/agent（跳过识别）…', 'running');
    const words = text.replace(/[,，、]/g, ' ').split(/\s+/).filter(Boolean);
    const result = await api('/api/agent', {
      method: 'POST',
      body: JSON.stringify({ words, session_id: state.sessionId }),
    });
    state.sessionId = result.session_id;
    $('tokens').innerHTML = '<div class="empty">本次跳过了识别步骤，没有 token 数据</div>';
    $('mock-badge').hidden = true;
    $('mock-warning').hidden = true;
    renderResult(result);
    pushSession(result);
    setStatus(`解译完成（${result.mode === 'llm' ? 'LLM' : '规则兜底'}，${result.elapsed_ms} ms${result.cached ? '，缓存命中' : ''}）`, 'ok');
  } catch (err) {
    setStatus('解译失败：' + err.message, 'err');
  } finally {
    $('btn-agent-only').disabled = false;
  }
}

/* ------------------------------------------------------------------ 事件绑定 */
document.addEventListener('DOMContentLoaded', () => {
  boot();

  $('btn-full').addEventListener('click', runFullFlow);
  $('btn-agent-only').addEventListener('click', runAgentOnly);

  $('btn-clear').addEventListener('click', () => {
    $('pinyin-input').value = '';
    $('tokens').innerHTML = '<div class="empty">点上面的按钮后，这里显示 YOLO 风格的 token（词 / 置信度 / 检测框）</div>';
    $('result-body').innerHTML = '<div class="empty big-empty">等待解译。左侧输入拼音后点「走完整流程」。</div>';
    $('result-meta').innerHTML = '';
    $('card-decode').hidden = true;
    $('card-tools').hidden = true;
    $('mock-badge').hidden = true;
    $('mock-warning').hidden = true;
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
