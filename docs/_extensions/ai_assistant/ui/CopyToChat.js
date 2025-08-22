(function () {
  function ready(fn) {
    if (document.readyState !== 'loading') { fn(); } else { document.addEventListener('DOMContentLoaded', fn); }
  }

  function getLanguage(el) {
    const c = (el && el.getAttribute && (el.getAttribute('class') || '')) + ' ' + ((el && el.closest && el.closest('.highlight') && el.closest('.highlight').getAttribute('class')) || '');
    let m = c && c.match(/language-([\w-]+)/i); if (m) return m[1];
    m = c && c.match(/highlight-([\w-]+)/i); if (m) return m[1];
    return '';
  }

  function getSelectionWithin(el) {
    const sel = window.getSelection && window.getSelection();
    if (!sel || sel.rangeCount === 0) return '';
    const range = sel.getRangeAt(0);
    if (!el || !el.contains || !el.contains(range.commonAncestorContainer)) return '';
    return sel.toString() || '';
  }

  function truncate(text, max = 8000) {
    if (!text) return '';
    if (text.length <= max) return text;
    return text.slice(0, max - 20) + "\n...";
  }

  function buildPrompt(lang, code) {
    const lead = 'Explain the following code snippet clearly and concisely. Include purpose, key steps, and potential pitfalls. If relevant, suggest improvements.';
    const fence = '```' + (lang || '') + '\n' + code + '\n```';
    return lead + '\n\n' + fence;
  }

  function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(text);
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } finally { document.body.removeChild(ta); }
    return Promise.resolve();
  }

  function openChat() {
    const url = 'https://chat.openai.com/';
    window.open(url, '_blank', 'noopener');
  }

  function buildToolbar(container, codeEl) {
    // Install toolbar CSS once
    if (!document.getElementById('ai-toolbar-style')) {
      const style = document.createElement('style');
      style.id = 'ai-toolbar-style';
      style.textContent = `
        .ai-toolbar{position:absolute;top:6px;right:6px;display:flex;gap:10px;z-index:2}
        .ai-btn{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;font-size:12px;border:1px solid #d1d5db;border-radius:8px;background:#ffffffcc;color:#111827;cursor:pointer;box-shadow:0 1px 2px rgba(17,24,39,.06)}
        .ai-btn:hover{box-shadow:0 4px 12px rgba(17,24,39,.10)}
        .ai-btn:focus{outline:2px solid rgba(111,176,0,.35);outline-offset:2px}
        .ai-btn__icon{font-size:14px;line-height:1}
        .ai-btn--primary{background:var(--primary,#6FB000);border-color:transparent;color:#fff;box-shadow:0 6px 16px rgba(17,24,39,.12)}
        .ai-btn--primary:hover{filter:brightness(1.05)}
        @media (max-width:480px){.ai-btn__label{display:none}.ai-btn{padding:6px}}
      `;
      document.head.appendChild(style);
    }

    const toolbar = document.createElement('div');
    toolbar.setAttribute('role', 'toolbar');
    toolbar.className = 'ai-toolbar';

    function makeBtn(iconEmoji, label, title, primary) {
      const b = document.createElement('button');
      b.type = 'button'; b.setAttribute('aria-label', label);
      if (title) b.title = title;
      b.className = 'ai-btn' + (primary ? ' ai-btn--primary' : '');
      const icon = document.createElement('span');
      icon.className = 'ai-btn__icon';
      icon.textContent = iconEmoji;
      const text = document.createElement('span');
      text.className = 'ai-btn__label';
      text.textContent = label;
      b.appendChild(icon);
      b.appendChild(text);
      return b;
    }

    // Copy button (raw code or selection)
    const copyBtn = makeBtn('📋', 'Copy', 'Copy code');
    copyBtn.onclick = () => {
      const raw = (codeEl && codeEl.textContent) || '';
      const selected = getSelectionWithin(codeEl);
      const text = (selected && selected.trim()) ? selected : raw;
      copyToClipboard(text);
    };

    // Ask AI button (prompt + code)
    const askBtn = makeBtn('💬', 'Ask AI', 'Copy prompt + code and open chat', true);
    askBtn.onclick = () => {
      const lang = getLanguage(codeEl);
      const raw = (codeEl && codeEl.textContent) || '';
      const selected = getSelectionWithin(codeEl);
      const snippet = truncate((selected && selected.trim()) ? selected : raw);
      const prompt = buildPrompt(lang, snippet);
      copyToClipboard(prompt).then(openChat);
    };

    toolbar.appendChild(copyBtn);
    toolbar.appendChild(askBtn);
    container.appendChild(toolbar);
  }

  function installToolbar() {
    // Wrap each code block and inject toolbar
    const highlights = Array.from(document.querySelectorAll('div.highlight'));
    const preOnly = Array.from(document.querySelectorAll('pre:not(.literal-block)')).filter(p => !p.closest('div.highlight'));
    const targets = highlights.concat(preOnly);

    targets.forEach(block => {
      if (block.getAttribute('data-ai-toolbar')) return;
      block.setAttribute('data-ai-toolbar', 'true');

      // Hide default sphinx copy button if present to avoid overlap
      const defaultCopy = block.querySelector('button.copybtn, .copybtn');
      if (defaultCopy) defaultCopy.style.display = 'none';

      // Ensure a wrapper with position: relative
      const wrapper = document.createElement('div');
      wrapper.style.position = 'relative';
      wrapper.style.width = '100%';
      block.parentNode.insertBefore(wrapper, block);
      wrapper.appendChild(block);

      // Add top padding so toolbar never overlaps code
      const pre = block.querySelector('pre') || block;
      if (pre && !pre.style.paddingTop) pre.style.paddingTop = '28px';

      // Determine code element
      const codeEl = block.querySelector('code') || pre;
      buildToolbar(wrapper, codeEl);
    });
  }

  ready(function () { installToolbar(); });
})();


