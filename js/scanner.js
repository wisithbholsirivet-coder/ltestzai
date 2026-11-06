/* =====================================================
   ChaeCheck . js/scanner.js  v3
   Deep-learning OCR (ocrWorker.js) + Ingredient Scan
   ===================================================== */

const Scanner = (() => {
  let _lastResults = [];
  let _lastRawText = '';

  // =====================================================
  //  OCR via local Ollama glm-ocr Vision Model
  //  Calls http://localhost:11434/api/generate
  //  Falls back to Tesseract.js Worker if Ollama down
  // =====================================================
  const OLLAMA_URL = 'http://localhost:11434/api/generate';
  const OLLAMA_MODEL = 'glm-ocr';
  const OLLAMA_TIMEOUT = 60000; // 60s — vision models need time

  // Tesseract fallback worker (lazy loaded only if Ollama fails)
  let _ocrWorker = null;
  let _ocrBusy = false;
  let _ocrScanId = 0;

  function _getOCRWorker() {
    if (_ocrWorker) return _ocrWorker;
    try { _ocrWorker = new Worker('js/ocrWorker.js'); } catch (_) { _ocrWorker = null; }
    return _ocrWorker;
  }

  // ── File → downscaled JPEG base64 ──
  // Max 1024px on longest side, JPEG 82% quality.
  // Reduces payload 5-10x for 12MP phone photos.
  const MAX_OCR_DIM = 1024;
  const JPEG_QUALITY = 0.82;

  async function _fileToBase64(file) {
    const bitmap = await createImageBitmap(file);
    let { width, height } = bitmap;

    // Downscale if larger than MAX_OCR_DIM
    if (width > MAX_OCR_DIM || height > MAX_OCR_DIM) {
      const scale = MAX_OCR_DIM / Math.max(width, height);
      width  = Math.round(width * scale);
      height = Math.round(height * scale);
    }

    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();

    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: JPEG_QUALITY });
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  // ═══════════════════════════════════════════════════
  // STAGE 1: Hardened Ollama API — Few-Shot Prompt
  // ═══════════════════════════════════════════════════
  const OLLAMA_SYSTEM_PROMPT = `You are a strict INCI cosmetic ingredient OCR tool. Extract the text from the image and output ONLY a comma-separated list of valid chemical names. Fix spelling errors to correct INCI names. Drop all marketing claims, symbols, Thai characters, and non-ingredient text. Output each ingredient exactly once. After the last ingredient, write END_OF_LIST and stop.`;

  const OLLAMA_USER_PROMPT = `EXAMPLE INPUT: Aqua ( Water ), Octocryrene, Iso hexadecane ， Silica， Dimethione, Gliceryll Stearate. U mineral oil that vegetable oi, markdown
EXAMPLE OUTPUT: Aqua (Water), Octocrylene, Isohexadecane, Silica, Dimethicone, Glyceryl Stearate END_OF_LIST
Now process the provided image and output ONLY the corrected comma-separated list of INCI ingredient names:`;

  // ── Main OCR function ──
  async function _runMLOCR(imageFile) {
    if (_ocrBusy) {
      console.warn('[OCR] Scan already in progress, skipping.');
      return '';
    }
    _ocrBusy = true;
    const thisScanId = ++_ocrScanId;

    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const ocrProgress  = document.getElementById('ocr-progress');

    if (ocrProgress) ocrProgress.hidden = false;
    if (progressFill) progressFill.style.width = '1%';
    if (progressText) progressText.textContent = 'Preparing image...';

    try {
      // Try Ollama first
      const text = await _runOllamaOCR(imageFile, progressFill, progressText);

      // Discard stale
      if (thisScanId !== _ocrScanId) { _ocrBusy = false; return ''; }

      if (progressFill) progressFill.style.width = '95%';
      if (progressText) progressText.textContent = 'Correcting spellings...';

      // Stage 2: Regex sanitization
      const sanitized = _fixOCRText(text);
      // Stage 3: Fuzzy match against DB to fix misspellings
      const corrected = DB.fuzzyCorrect(sanitized);

      if (progressFill) progressFill.style.width = '100%';
      if (progressText) progressText.textContent =
        I18n.t('scanner.ocr.done', 'Text extraction complete!');
      setTimeout(() => { if (ocrProgress) ocrProgress.hidden = true; }, 1500);

      _ocrBusy = false;
      return corrected;
    } catch (ollamaErr) {
      console.warn('[Ollama OCR] Failed, falling back to Tesseract:', ollamaErr.message);

      // Fallback: Tesseract via Web Worker
      try {
        if (progressText) progressText.textContent = 'Ollama unavailable, using Tesseract...';
        const text = await _runTesseractFallback(imageFile, thisScanId, progressFill, progressText, ocrProgress);
        _ocrBusy = false;
        return DB.fuzzyCorrect(text); // Stage 3 on Tesseract output too
      } catch (tessErr) {
        _ocrBusy = false;
        throw tessErr;
      }
    }
  }

  // ── Ollama API call ──
  async function _runOllamaOCR(imageFile, progressFill, progressText) {
    if (progressText) progressText.textContent = 'Connecting to local AI model...';
    if (progressFill) progressFill.style.width = '10%';

    const base64Image = await _fileToBase64(imageFile);
    if (progressFill) progressFill.style.width = '20%';
    if (progressText) progressText.textContent = 'Sending to AI model...';

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), OLLAMA_TIMEOUT);

    let response;
    try {
      response = await fetch(OLLAMA_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          model: OLLAMA_MODEL,
          system: OLLAMA_SYSTEM_PROMPT,
          prompt: OLLAMA_USER_PROMPT,
          images: [base64Image],
          stream: false,
          keep_alive: -1,
          options: {
            temperature: 0.0,          // absolute determinism
            num_predict: 512,          // hard kill-switch for runaway generation
            top_p: 1.0,
            top_k: 40,                 // restrict vocabulary breadth
            repeat_penalty: 1.5,       // break autoregressive loops
            repeat_last_n: 256,
            penalize_newline: true,
          }
        })
      });
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        throw new Error('Local OCR server timed out (60s). Is Ollama running?');
      }
      throw new Error('Local OCR server is currently unreachable. Start Ollama with: ollama serve');
    }

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errText = await response.text().catch(() => '');
      throw new Error(`Ollama returned ${response.status}: ${errText.slice(0, 200)}`);
    }

    if (progressFill) progressFill.style.width = '50%';
    if (progressText) progressText.textContent = 'AI model analyzing label...';

    const data = await response.json();
    const rawText = (data.response || '').trim();

    if (progressFill) progressFill.style.width = '90%';

    if (!rawText || rawText === 'NO_INGREDIENTS_FOUND') {
      return '';
    }

    return rawText;
  }

  // ── Tesseract fallback (original Worker-based path) ──
  async function _runTesseractFallback(imageFile, thisScanId, progressFill, progressText, ocrProgress) {
    const worker = _getOCRWorker();
    if (!worker) throw new Error('OCR worker unavailable');

    // Pre-warm if needed
    worker.postMessage({ type: 'load' });

    const bitmap = await createImageBitmap(imageFile);
    const { width, height } = bitmap;
    const offscreen = new OffscreenCanvas(width, height);
    offscreen.getContext('2d').drawImage(bitmap, 0, 0);
    bitmap.close();
    const imageData = offscreen.getContext('2d').getImageData(0, 0, width, height);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(
        () => { reject(new Error('OCR timed out (180s)')); }, 180000
      );

      worker.onmessage = ({ data: msg }) => {
        if (msg.type === 'progress') {
          if (progressFill) progressFill.style.width = `${msg.pct}%`;
          if (progressText) progressText.textContent  = msg.label || 'Processing...';
          return;
        }
        if (msg.type === 'ready') return;
        clearTimeout(timeout);

        if (thisScanId !== _ocrScanId) { resolve(''); return; }

        if (msg.type === 'done') {
          if (progressFill) progressFill.style.width = '100%';
          if (progressText) progressText.textContent =
            I18n.t('scanner.ocr.done', 'Text extraction complete!');
          setTimeout(() => { if (ocrProgress) ocrProgress.hidden = true; }, 1500);
          resolve(_fixOCRText(msg.text || ''));
        } else if (msg.type === 'error') {
          reject(new Error(msg.message || 'Worker error'));
        }
      };

      worker.onerror = (e) => { clearTimeout(timeout); reject(e); };
      worker.postMessage(
        { type: 'ocr', imageData: imageData.data.buffer, width, height },
        [imageData.data.buffer]
      );
    });
  }

  // Pre-warm: check if Ollama is running on page load
  function _prewarmOCR() {
    fetch('http://localhost:11434/api/tags', { method: 'GET' })
      .then(r => r.json())
      .then(data => {
        const hasModel = (data.models || []).some(m => m.name.startsWith(OLLAMA_MODEL));
        console.log(`[OCR] Ollama ${hasModel ? 'ready with ' + OLLAMA_MODEL : 'running but model not found'}`);
        if (!hasModel) {
          console.warn(`[OCR] Run: ollama pull ${OLLAMA_MODEL}`);
        }
      })
      .catch(() => {
        console.warn('[OCR] Ollama not reachable. Will fallback to Tesseract.js.');
        // Pre-warm Tesseract instead
        const w = _getOCRWorker();
        if (w) w.postMessage({ type: 'load' });
      });
  }


  // ═══════════════════════════════════════════════════
  // STAGE 2: Programmatic Regex Sanitization
  // ═══════════════════════════════════════════════════
  function _fixOCRText(raw) {
    if (!raw) return '';
    let s = raw;

    // ── 2a. Truncate at END_OF_LIST / NO_INGREDIENTS_FOUND marker ──
    const endIdx = s.search(/END_OF_LIST|NO_INGREDIENTS_FOUND/i);
    if (endIdx !== -1) s = s.slice(0, endIdx);

    // ── 2b. Cut at repeated INGREDIENTS header (loop restart) ──
    const headerRe = /(?:ingredients?\s*[:\)\}\]]|ส่วนประกอบ\s*[:\)\}\]]?)/gi;
    const headers = [...s.matchAll(headerRe)];
    if (headers.length >= 1) {
      const firstEnd = headers[0].index + headers[0][0].length;
      s = headers.length >= 2
        ? s.slice(firstEnd, headers[1].index)
        : s.slice(firstEnd);
    }

    // ── 2c. Normalize ALL delimiters to standard comma ──
    s = s
      .replace(/\uFF0C/g, ',')           // full-width comma ，
      .replace(/\u3001/g, ',')           // Japanese comma 、
      .replace(/;/g, ',')                // semicolons
      .replace(/\.(?=\s*[A-Z])/g, ',')   // period before next ingredient
      .replace(/\.\s*$/g, '')            // trailing period
      .replace(/\r?\n/g, ', ')           // newlines
      .replace(/\uFB01/g, 'fi').replace(/\uFB02/g, 'fl')  // ligatures
      .replace(/\uFB00/g, 'ff').replace(/\uFB03/g, 'ffi').replace(/\uFB04/g, 'ffl');

    // ── 2d. Marketing & artifact purge ──
    s = s
      // Thai characters
      .replace(/[\u0E00-\u0E7F]+/g, ' ')
      // Marketing claim phrases
      .replace(/\b(mineral\s+oil\s+(free|that)|vegetable\s+oil\s*(free)?|paraben\s+free|dermatologically\s+tested|no\s+animal\s+testing|clinically\s+proven|for\s+sensitive\s+skin|fragrance\s+free|alcohol\s+free|hypoallergenic|non[- ]?comedogenic)\b/gi, ' ')
      // "markdown" artifact from LLM
      .replace(/\bmarkdown\b/gi, ' ')
      // Floating single characters (e.g., "U", "o", "i" from mangled text)
      .replace(/(?:^|,)\s*[A-Za-z]\s*(?=,|$)/g, ',')
      // Copyright, trademark, registered symbols
      .replace(/[\u00A9\u00AE\u2122]/g, '')
      // Noise characters
      .replace(/[*#^~`_=+<>{}\[\]\\@$%&!?|]/g, ' ');

    // ── 2e. Whitespace cleanup ──
    s = s
      .replace(/[ \t]{2,}/g, ' ')        // collapse multi-spaces
      .replace(/,\s*,+/g, ',')           // collapse multi-commas
      .replace(/^[,\s]+|[,\s]+$/g, '')   // trim leading/trailing commas+spaces
      .trim();

    // ── 2f. Tokenize, filter, and loop-killer dedup ──
    const tokens = s.split(/,/).map(t => t.trim()).filter(t => {
      if (t.length < 2) return false;
      // Must contain at least 2 consecutive letters
      if (!/[A-Za-z]{2,}/.test(t)) return false;
      // Skip digit-heavy tokens (batch codes, etc.)
      const digits  = (t.match(/\d/g)  || []).length;
      const letters = (t.match(/[A-Za-z]/g) || []).length;
      if (letters === 0 || digits / t.length > 0.5) return false;
      // Skip very short all-caps non-abbreviations
      if (t.length <= 2 && /^[A-Z]+$/.test(t)) return false;
      // Common noise words
      const noise = /^(the|and|for|with|see|back|receipt|offer|total|tel|cashier|refund|invoice|your|order|next|original|from|this|that|also|may|can|use|used|not|all|has|had|have|been|was|are|were|will|www|http|com|net|org|page|end|of|list|free|tested|proven|made|in|by|contains|without)$/i;
      if (noise.test(t)) return false;
      return true;
    });

    // Clean up individual tokens
    const cleaned = tokens.map(t =>
      t.replace(/^[-.:]+|[-.:]+$/g, '').trim()
    ).filter(t => t.length >= 2);

    // LOOP-KILLER DEDUP: truncate at first duplicate (not just remove it).
    // If "Water" appears at index 0 and again at index 15, the model
    // looped — everything from index 15 onward is hallucinated repetition.
    const seen = new Set();
    const unique = [];
    for (const item of cleaned) {
      const key = item.toLowerCase().replace(/\s+/g, ' ').trim();
      if (seen.has(key)) {
        // First duplicate = loop start. Truncate here.
        console.warn(`[OCR Sanitizer] Loop detected at "${item}", truncating.`);
        break;
      }
      seen.add(key);
      unique.push(item);
    }

    return unique.join(', ');
  }


  // =====================================================
  //  Helpers
  // =====================================================
  // showToast: delegate to global in app.js
  function showToast(msg, duration) {
    if (typeof window.showToast === 'function') {
      window.showToast(msg, duration);
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function _applyFilter(filter) {
    const cards = document.querySelectorAll('.ingredient-card');
    cards.forEach(card => {
      card.style.display =
        (!filter || filter === 'all' || card.dataset.risk === filter)
          ? '' : 'none';
    });
    document.querySelectorAll('.risk-filter').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.filter === filter);
    });
  }


  // =====================================================
  //  Render ingredient card
  // =====================================================
  function _renderCard(result) {
    const lang      = I18n.getLang();
    const riskLabel = { green: 'SAFE', yellow: 'CAUTION', red: 'DANGER', unknown: 'UNKNOWN' }[result.risk] || 'UNKNOWN';
    const riskEmoji = { green: 'circle', yellow: 'circle', red: 'circle', unknown: '?' }[result.risk] || '?';
    const card = document.createElement('div');
    card.className = `ingredient-card risk-${result.risk}`;
    card.setAttribute('role', 'listitem');
    card.setAttribute('tabindex', '0');
    card.dataset.risk = result.risk;

    const purpose = lang === 'th'
      ? (result.description_th || result.purpose || '')
      : (result.purpose || '');
    const isFuzzy = result._matchType === 'fuzzy';

    const fuzzyLabel = lang === 'th' ? '~ ใกล้เคียง' : '~ fuzzy match';
    const flagsHtml = [];
    if (result._isUserAllergen)  flagsHtml.push(`<span class="card-flag allergen-badge" title="${lang === 'th' ? '⚠ สารก่อภูมิแพ้ที่คุณเลือก' : '⚠ Your allergen'}">⚠ ${lang === 'th' ? 'แพ้' : 'Allergen'}</span>`);
    if (result.kidsUnsafe)       flagsHtml.push(`<span class="card-flag" title="${lang === 'th' ? 'ไม่เหมาะสำหรับเด็ก' : 'Not safe for children'}">&#x1F476;</span>`);
    if (result.pregnancyWarning) flagsHtml.push(`<span class="card-flag" title="${lang === 'th' ? 'ระวังในระหว่างตั้งครรภ์' : 'Caution during pregnancy'}">&#x1F931;</span>`);
    if (isFuzzy)                 flagsHtml.push(`<span class="card-flag" title="${fuzzyLabel}">${fuzzyLabel}</span>`);

    let suggestHtml = '';
    if (result.risk === 'unknown' && result._suggestions && result._suggestions.length > 0) {
      const chips = result._suggestions.slice(0, 3).map(s =>
        `<button class="suggest-chip" data-sug-id="${escapeHtml(s.id)}">${escapeHtml(s.names[0])}</button>`
      ).join('');
      suggestHtml = `<div class="card-suggest"><span class="suggest-label">${lang === 'th' ? 'หมายถึง?' : 'DID YOU MEAN?'}</span>${chips}</div>`;
    }

    if (result.risk === 'unknown') {
      card.innerHTML =
        `<div class="card-header">
          <div class="card-name">${escapeHtml(result.inputName)}</div>
          <span class="risk-badge badge-unknown">${riskLabel}</span>
        </div>
        <div class="card-unknown-msg">Not found in our database</div>
        ${suggestHtml}`;
    } else {
      const detailLink = `<span class="card-detail-link">${I18n.t('card.learn.more', 'Details &rarr;')}</span>`;
      card.innerHTML =
        `<div class="card-header">
          <div class="card-name">${escapeHtml(result.inputName)}</div>
          <span class="risk-badge badge-${result.risk}">${riskLabel}</span>
        </div>
        ${result.category ? `<div class="card-category">${escapeHtml(lang === 'th' ? (CATEGORY_TH[result.category] || result.category) : result.category)}</div>` : ''}
        <div class="card-purpose">${escapeHtml(purpose)}</div>
        <div class="card-footer">
          <div class="card-flags">${flagsHtml.join('')}</div>
          ${detailLink}
        </div>`;
    }

    if (result.risk !== 'unknown') {
      card.addEventListener('click', e => {
        if (!e.target.closest('.suggest-chip')) openModal(result);
      });
      card.addEventListener('keydown', e => {
        if ((e.key === 'Enter' || e.key === ' ') && !e.target.closest('.suggest-chip')) {
          e.preventDefault(); openModal(result);
        }
      });
    }

    card.querySelectorAll('.suggest-chip').forEach(btn => {
      btn.addEventListener('click', e => {
        e.stopPropagation();
        const sug = DB.getById(btn.dataset.sugId);
        if (sug) {
          openModal({
            inputName: result.inputName + ' -> ' + sug.names[0],
            ...sug,
            _matchType: 'suggested',
            _suggestions: [],
          });
        }
      });
    });

    return card;
  }


  // =====================================================
  //  Show results
  // =====================================================
  function showResults(results, rawText) {
    _lastResults = results;
    _lastRawText = rawText;

    const section   = document.getElementById('results-section');
    const list      = document.getElementById('ingredient-grid');
    const summary   = document.getElementById('summary-chips');
    const filterBar = document.getElementById('risk-filter-bar');

    if (!section || !list) return;

    list.innerHTML = '';
    section.hidden = false;

    const counts = { green: 0, yellow: 0, red: 0, unknown: 0 };
    results.forEach(r => { counts[r.risk] = (counts[r.risk] || 0) + 1; });

    if (summary) {
      const lang = I18n.getLang();
      summary.innerHTML = [
        counts.red    ? `<span class="sum-chip chip-red">&#x1F534; ${lang === 'th' ? 'อันตราย ' + counts.red   : counts.red   + ' Danger'}</span>`   : '',
        counts.yellow ? `<span class="sum-chip chip-yellow">&#x1F7E1; ${lang === 'th' ? 'ระวัง '    + counts.yellow : counts.yellow + ' Caution'}</span>` : '',
        counts.green  ? `<span class="sum-chip chip-green">&#x1F7E2; ${lang === 'th' ? 'ปลอดภัย ' + counts.green  : counts.green  + ' Safe'}</span>`    : '',
        counts.unknown? `<span class="sum-chip chip-unknown">&#x26AA; ${lang === 'th' ? 'ไม่รู้จัก ' + counts.unknown : counts.unknown + ' Unknown'}</span>` : '',
      ].filter(Boolean).join('');
    }

    results.forEach(r => list.appendChild(_renderCard(r)));

    if (filterBar) filterBar.hidden = results.length < 2;
    _applyFilter('all');

    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }


  // =====================================================
  //  Modal
  // =====================================================
  const CATEGORY_TH = {
    'Abrasive': 'สารขัดผิว', 'Abrasive/Absorbent': 'สารขัดผิว/ดูดซับ', 'Absorbent': 'สารดูดซับ', 'Active Ingredient': 'สารออกฤทธิ์',
    'Amino': 'กรดอะมิโน', 'Amino Acid': 'กรดอะมิโน', 'Anti-acne': 'รักษาสิว', 'Anti-aging': 'ต่อต้านริ้วรอย', 'Anti-inflammatory': 'ลดการอักเสบ',
    'Antifungal': 'ต้านเชื้อรา', 'Antifungal/Antibacterial': 'ต้านเชื้อรา/แบคทีเรีย', 'Antimicrobial': 'ต้านเชื้อจุลินทรีย์',
    'Antioxidant': 'สารต้านอนุมูลอิสระ', 'Antioxidant Preservative': 'สารต้านอนุมูลอิสระ/กันเสีย', 'Antioxidant/Chelating': 'ต้านอนุมูลอิสระ/จับโลหะ',
    'Antiperspirant': 'ระงับเหงื่อ', 'Antiseptic': 'ฆ่าเชื้อ', 'Astringent': 'กระชับรูขุมขน', 'BHA': 'BHA', 'Binder': 'สารยึดเกาะ',
    'Botanical': 'สารสกัดจากพืช', 'Botanical Extract': 'สารสกัดจากพืช', 'Chelating': 'สารจับโลหะ', 'Chelating Agent': 'สารจับโลหะ',
    'Cleansing': 'สารทำความสะอาด', 'Colorant': 'สารให้สี', 'Conditioning Agent': 'สารบำรุง', 'Contaminant': 'สารปนเปื้อน',
    'Cooling Agent': 'สารให้ความเย็น', 'Coupling Agent': 'สารประสาน', 'Deodorizing': 'ระงับกลิ่น', 'Depilatory': 'ขจัดขน',
    'Dispersant': 'สารกระจายตัว', 'Emollient': 'สารให้ความชุ่มชื้น', 'Emollient/Emulsifier': 'ให้ความชุ่มชื้น/ผสานน้ำมัน',
    'Emulsifier': 'ผสานน้ำกับน้ำมัน', 'Encapsulant': 'สารห่อหุ้ม', 'Enzyme': 'เอนไซม์', 'Essential Oil': 'น้ำมันหอมระเหย',
    'Exfoliant': 'ผลัดเซลล์ผิว', 'Exfoliant (AHA)': 'ผลัดเซลล์ผิว (AHA)', 'Exfoliant (PHA)': 'ผลัดเซลล์ผิว (PHA)',
    'Fatty': 'กรดไขมัน', 'Fatty Acid': 'กรดไขมัน', 'Ferment': 'สารสกัดจากการหมัก', 'Film Former': 'สารสร้างฟิล์ม',
    'Fragrance': 'น้ำหอม', 'Humectant': 'ดึงดูดน้ำให้ผิว', 'Humectant / Preservative Booster': 'ดึงดูดน้ำ/เสริมกันเสีย',
    'Humectant/Exfoliant': 'ดึงดูดน้ำ/ผลัดเซลล์ผิว', 'Humectant/Preservative Booster': 'ดึงดูดน้ำ/เสริมกันเสีย',
    'Humectant/Solvent': 'ดึงดูดน้ำ/ตัวทำละลาย', 'Lubricant': 'สารหล่อลื่น', 'Mineral': 'แร่ธาตุ', 'Neurotransmitter': 'สารสื่อประสาท',
    'Oxidizing Agent': 'สารออกซิไดซ์', 'Penetration Enhancer': 'สารช่วยนำพา', 'Peptide': 'เปปไทด์', 'Physical': 'ฟิสิคัล',
    'Preservative': 'สารกันเสีย', 'Preservative Booster': 'สารเสริมกันเสีย', 'Probiotic': 'โปรไบโอติก', 'Propellant': 'ก๊าซขับดัน',
    'Protein': 'โปรตีน', 'Restricted': 'สารควบคุม', 'Retinoid': 'เรตินอยด์', 'Sebum': 'ซีบัม', 'Self-Tanner': 'สารทำผิวแทน',
    'Silicone': 'ซิลิโคน', 'Skin': 'ผิวหนัง', 'Skin Barrier': 'เสริมเกราะป้องกันผิว', 'Skin Brightener': 'ปรับผิวกระจ่างใส',
    'Skin Conditioner': 'สารบำรุงผิว', 'Skin Conditioner / Fragrance': 'สารบำรุงผิว/น้ำหอม', 'Skin Conditioning': 'สารบำรุงผิว',
    'Skin Protectant': 'สารปกป้องผิว', 'Skin Soothing': 'สารปลอบประโลมผิว', 'Solvent': 'ตัวทำละลาย', 'Surfactant': 'สารทำความสะอาด',
    'Thickener': 'เพิ่มความหนืด', 'UV': 'กันแดด', 'UV Filter': 'สารกันแดด', 'Viscosity': 'ความหนืด', 'Vitamin': 'วิตามิน',
    'pH': 'ค่า pH', 'pH Adjuster': 'ปรับค่า pH'
  };

  const CONCERN_TH = {
    'Potential contact allergen': 'อาจก่อให้เกิดการแพ้สัมผัส',
    'May cause skin sensitization': 'อาจทำให้ผิวแพ้ง่ายขึ้น',
    'May cause allergic reactions': 'อาจทำให้เกิดอาการแพ้',
    'Skin irritant at high concentrations': '刺激ผิวได้ที่ความเข้มข้นสูง',
    'Skin irritant': 'ทำให้ผิวระคายเคือง',
    'Eye irritant': 'ระคายเคืองตา',
    'May cause respiratory irritation': 'อาจระคายเคืองระบบทางเดินหายใจ',
    'Possible endocrine disruptor': 'อาจรบกวนระบบฮอร์โมน',
    'Suspected carcinogen': 'ต้องสงสัยว่าเป็นสารก่อมะเร็ง',
    'Known carcinogen': 'เป็นสารก่อมะเร็ง',
    'Avoid in children under 3': 'หลีกเลี่ยงในเด็กอายุต่ำกว่า 3 ปี',
    'Avoid during pregnancy': 'หลีกเลี่ยงในระหว่างตั้งครรภ์',
    'Not recommended during pregnancy': 'ไม่แนะนำในระหว่างตั้งครรภ์',
    'Restricted in EU cosmetics': 'ถูกจำกัดการใช้ในสหภาพยุโรป',
    'Banned in EU': 'ถูกห้ามใช้ในสหภาพยุโรป',
    'May irritate sensitive skin': 'อาจระคายเคืองผิวบอบบาง',
    'Photosensitizing': 'ทำให้ผิวไวต่อแสงแดด',
    'May cause photosensitivity': 'อาจทำให้ผิวไวต่อแสง',
    'Accumulates in body fat': 'สะสมในเนื้อเยื่อไขมัน',
    'May be comedogenic': 'อาจอุดตันรูขุมขน',
    'May bleach fabrics': 'อาจทำให้ผ้าซีดจาง',
    'High concentration may cause irritation': 'ความเข้มข้นสูงอาจระคายเคือง',
    'Toxic if ingested': 'เป็นพิษหากกลืนกิน',
    'Linked to hair protein loss': 'เชื่อมโยงกับการสูญเสียโปรตีนเส้นผม',
  };

  function openModal(result) {
    const overlay     = document.getElementById('modal-overlay');
    const banner      = document.getElementById('modal-risk-banner');
    const nameEl      = document.getElementById('modal-ingredient-name');
    const metaEl      = document.getElementById('modal-meta');
    const purposeEl   = document.getElementById('modal-purpose');
    const descEl      = document.getElementById('modal-description');
    const concernsEl  = document.getElementById('modal-concerns');
    const concernsSec = document.getElementById('modal-concerns-section');
    const warningsRow = document.getElementById('modal-warnings-row');

    if (!overlay) return;

    const lang        = I18n.getLang();
    const riskLabel   = I18n.t('risk.label.' + result.risk, result.risk);
    const description = lang === 'th'
      ? (result.description_th || result.description_en || I18n.t('error.notfound'))
      : (result.description_en || I18n.t('error.notfound'));

    if (banner)  banner.className   = 'modal-risk-banner risk-' + result.risk;
    // Fallback chain: inputName → first alias → category → riskLabel
    const displayName = result.inputName
      || (result.names && result.names[0])
      || result.category
      || riskLabel
      || '?';
    if (nameEl)  nameEl.textContent = displayName;

    const purposeSection = document.getElementById('modal-purpose-section');
    if (purposeSection) purposeSection.style.display = '';
    
    if (purposeEl) {
      if (lang === 'th') {
        const catTh = result.category ? (CATEGORY_TH[result.category] || result.category) : '';
        // If we translated the category, use it. Otherwise fallback to english purpose.
        purposeEl.textContent = catTh || result.purpose || I18n.t('error.notfound');
      } else {
        purposeEl.textContent = result.purpose || I18n.t('error.notfound');
      }
    }

    if (descEl) descEl.textContent = description;

    if (metaEl) {
      const riskChipClass = { green: 'chip-risk-green', yellow: 'chip-risk-yellow', red: 'chip-risk-red' }[result.risk] || '';
      metaEl.innerHTML = [
        `<span class="modal-meta-chip ${riskChipClass}">${escapeHtml(riskLabel)}</span>`,
        result.category   ? `<span class="modal-meta-chip">${escapeHtml(lang === 'th' ? (CATEGORY_TH[result.category] || result.category) : result.category)}</span>` : '',
        result._matchType === 'fuzzy'     ? `<span class="modal-meta-chip">${lang === 'th' ? '~ ใกล้เคียง' : '~ fuzzy match'}</span>` : '',
        result._matchType === 'suggested' ? `<span class="modal-meta-chip">${lang === 'th' ? 'แนะนำ' : 'suggested'}</span>`          : '',
      ].filter(Boolean).join('');
    }

    if (concernsEl && concernsSec) {
      if (result.concerns && result.concerns.length > 0) {
        concernsEl.innerHTML = result.concerns
          .map(c => {
            const translated = lang === 'th' ? (CONCERN_TH[c] || c) : c;
            return `<li>${escapeHtml(translated)}</li>`;
          }).join('');
        concernsSec.hidden = false;
      } else {
        concernsSec.hidden = true;
      }
    }

    if (warningsRow) {
      warningsRow.innerHTML = [
        result.kidsUnsafe       ? `<span class="modal-warning-chip">⚠ ${I18n.t('warning.kids', 'Not safe for children')}</span>`       : '',
        result.pregnancyWarning ? `<span class="modal-warning-chip">⚠ ${I18n.t('warning.pregnancy', 'Caution during pregnancy')}</span>` : '',
        result._isUserAllergen  ? `<span class="modal-warning-chip allergen-badge">⚠ ${I18n.t('warning.allergen', 'Your allergen')}</span>` : '',
      ].filter(Boolean).join('');
    }

    // Show reference citation
    const refEl = document.getElementById('modal-reference');
    if (refEl) {
      if (result.reference) {
        refEl.textContent = `📋 ${lang === 'th' ? 'อ้างอิง' : 'Reference'}: ${result.reference}`;
        refEl.hidden = false;
      } else {
        refEl.hidden = true;
      }
    }

    // Apply colored top border class to modal element
    const modalEl = document.getElementById('ingredient-modal');
    if (modalEl) {
      modalEl.classList.remove('border-green', 'border-yellow', 'border-red', 'border-unknown');
      const borderMap = { green: 'border-green', yellow: 'border-yellow', red: 'border-red', unknown: 'border-unknown' };
      if (borderMap[result.risk]) modalEl.classList.add(borderMap[result.risk]);
    }

    overlay.classList.remove('modal-overlay--hidden');
    document.body.style.overflow = 'hidden';
    setTimeout(() => {
      const closeBtn = document.getElementById('modal-close');
      if (closeBtn) closeBtn.focus();
    }, 80);
  }

  function closeModal() {
    const overlay = document.getElementById('modal-overlay');
    if (overlay) overlay.classList.add('modal-overlay--hidden');
    document.body.style.overflow = '';
  }


  // =====================================================
  //  Restore from history
  // =====================================================
  function restoreResults(results, rawText) {
    const section = document.getElementById('results-section');
    if (section) section.hidden = false;
    showResults(results, rawText);
    if (rawText) {
      const textarea = document.getElementById('manual-input');
      if (textarea) textarea.value = rawText;
    }
  }


  // =====================================================
  //  Main scan
  // =====================================================
  // ── Allergen categories mapping ──
  const ALLERGEN_CATEGORY_MAP = {
    'fragrance':     ['Fragrance', 'Parfum', 'Aroma'],
    'essential-oil': ['Essential Oil'],
    'paraben':       ['Preservative'],
    'sulfate':       ['Surfactant'],
    'alcohol':       ['Solvent'],
    'preservative':  ['Preservative'],
  };

  function _getSelectedAllergens() {
    const checks = document.querySelectorAll('.allergy-checkbox:checked');
    const cats = new Set();
    checks.forEach(cb => {
      const keywords = ALLERGEN_CATEGORY_MAP[cb.value] || [];
      keywords.forEach(k => cats.add(k.toLowerCase()));
      // Also match by concern keywords
      cats.add(cb.value.toLowerCase());
    });
    return cats;
  }

  function _isAllergenMatch(result, allergenCats) {
    if (allergenCats.size === 0) return false;
    const cat = (result.category || '').toLowerCase();
    // Check category match
    for (const ac of allergenCats) {
      if (cat.includes(ac)) return true;
    }
    // Check concerns for allergen keywords
    if (result.concerns) {
      for (const concern of result.concerns) {
        const cl = concern.toLowerCase();
        if (cl.includes('allergen') || cl.includes('sensitiz') || cl.includes('fragrance')) {
          if (allergenCats.has('fragrance') || allergenCats.has('essential-oil')) return true;
        }
      }
    }
    // Check name for specific allergen matches
    const nameL = (result.names?.[0] || '').toLowerCase();
    if (allergenCats.has('paraben') && nameL.includes('paraben')) return true;
    if (allergenCats.has('sulfate') && (nameL.includes('sulfate') || nameL.includes('sulphate'))) return true;
    if (allergenCats.has('alcohol') && nameL.includes('alcohol') && !nameL.includes('cetyl') && !nameL.includes('cetearyl') && !nameL.includes('stearyl')) return true;
    return false;
  }

  async function scan(inputText) {
    if (!inputText || !inputText.trim()) return;

    if (!DB.isLoaded()) {
      showToast(I18n.t('db.loading', 'Database still loading, please wait...'));
      return;
    }

    const scanLoading = document.getElementById('scan-loading');
    const scanBtn     = document.getElementById('scan-btn');
    const section     = document.getElementById('results-section');

    if (scanLoading) scanLoading.hidden = false;
    if (section)     section.hidden     = true;
    if (scanBtn)     scanBtn.disabled   = true;

    try {
      await new Promise(r => setTimeout(r, 60));
      const results = DB.search(inputText);

      // Cross-reference with user allergen selections
      const allergenCats = _getSelectedAllergens();
      if (allergenCats.size > 0) {
        results.forEach(r => {
          r._isUserAllergen = _isAllergenMatch(r, allergenCats);
        });
      }

      if (results.length === 0) {
        showToast(I18n.t('scan.no.ingredients', 'No ingredients found. Try pasting the full ingredient list.'));
      } else {
        showResults(results, inputText);
      }
    } catch (err) {
      console.error('[Scanner] scan error:', err);
      showToast(I18n.t('error.scan.failed', 'Scan failed. Please try again.'));
    } finally {
      if (scanLoading) scanLoading.hidden = true;
      if (scanBtn)     scanBtn.disabled   = false;
    }
  }


  // =====================================================
  //  Init
  // =====================================================
  function init() {
    const uploadZone  = document.getElementById('upload-zone');
    const fileInput   = document.getElementById('file-input');
    const uploadClear = document.getElementById('upload-clear');
    const scanBtn     = document.getElementById('scan-btn');
    const manualInput = document.getElementById('manual-input');
    const saveScanBtn = document.getElementById('save-scan-btn');

    // Pre-warm ML model download on page open
    _prewarmOCR();

    if (uploadZone) {
      uploadZone.addEventListener('click', () => fileInput && fileInput.click());
      uploadZone.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput && fileInput.click(); }
      });
      uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
      uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
      uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) _handleImageFile(file);
      });
    }

    if (fileInput) {
      fileInput.addEventListener('change', () => {
        const file = fileInput.files && fileInput.files[0];
        if (file) _handleImageFile(file);
        fileInput.value = '';
      });
    }

    if (uploadClear) {
      uploadClear.addEventListener('click', e => {
        e.stopPropagation();
        const prev = document.getElementById('upload-preview');
        const ph   = document.getElementById('upload-placeholder');
        const prog = document.getElementById('ocr-progress');
        const card = document.getElementById('upload-card');
        const mini = document.getElementById('upload-mini');
        if (prev) prev.hidden = true;
        if (ph)   ph.hidden   = false;
        if (prog) prog.hidden = true;
        // Restore collapsed upload card
        if (card) card.classList.remove('upload-card--collapsed');
        if (mini) mini.hidden = true;
      });
    }

    // "Change" button on mini upload strip
    const changeBtn = document.getElementById('upload-change-btn');
    if (changeBtn) {
      changeBtn.addEventListener('click', (e) => {
        e.stopPropagation();  // prevent toggling <details>
        e.preventDefault();
        const card = document.getElementById('upload-card');
        const mini = document.getElementById('upload-mini');
        if (card) card.classList.remove('upload-card--collapsed');
        if (mini) mini.hidden = true;
        if (fileInput) fileInput.click();
      });
    }

    if (scanBtn) {
      scanBtn.addEventListener('click', () => {
        const text = (manualInput && manualInput.value || '').trim();
        if (!text) {
          showToast(I18n.t('scan.empty', 'Please paste ingredient text first'));
          if (manualInput) manualInput.focus();
          return;
        }
        scan(text);
      });
    }

    document.querySelectorAll('.risk-filter').forEach(btn => {
      btn.addEventListener('click', () => _applyFilter(btn.dataset.filter));
    });

    // Glare tip banner
    const glareTipToggle = document.getElementById('glare-tip-toggle');
    const glareTipBody   = document.getElementById('glare-tip-body');
    const glareTipBanner = document.getElementById('glare-tip-banner');
    if (glareTipToggle && glareTipBody && glareTipBanner) {
      let tipOpen = false;
      try { tipOpen = localStorage.getItem('chaecheck_tip_open') === '1'; } catch (_) {}
      if (tipOpen) {
        glareTipBody.hidden = false;
        glareTipBanner.classList.add('open');
        glareTipToggle.setAttribute('aria-expanded', 'true');
      }
      glareTipToggle.addEventListener('click', () => {
        tipOpen = !tipOpen;
        glareTipBody.hidden = !tipOpen;
        glareTipBanner.classList.toggle('open', tipOpen);
        glareTipToggle.setAttribute('aria-expanded', String(tipOpen));
        try { localStorage.setItem('chaecheck_tip_open', tipOpen ? '1' : '0'); } catch (_) {}
      });
    }

    const modalClose   = document.getElementById('modal-close');
    const modalOverlay = document.getElementById('modal-overlay');
    if (modalClose)   modalClose.addEventListener('click', closeModal);
    if (modalOverlay) {
      modalOverlay.addEventListener('click', e => {
        if (!e.target.closest('.modal')) closeModal();
      });
    }
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

    if (saveScanBtn) {
      saveScanBtn.addEventListener('click', () => {
        if (!_lastResults.length) {
          showToast(I18n.t('toast.no.results', 'No results to save'));
          return;
        }
        HistoryManager.add(_lastResults, _lastRawText);
        showToast(I18n.t('toast.saved', 'Scan saved to history'));
      });
    }
  }


  // =====================================================
  //  Handle image upload
  // =====================================================
  async function _handleImageFile(file) {
    const preview     = document.getElementById('upload-preview');
    const placeholder = document.getElementById('upload-placeholder');
    const previewImg  = document.getElementById('preview-img');
    const uploadCard  = document.getElementById('upload-card');
    const uploadMini  = document.getElementById('upload-mini');

    if (previewImg) previewImg.src = URL.createObjectURL(file);
    if (preview)     preview.hidden    = false;
    if (placeholder) placeholder.hidden = true;

    try {
      const text = await _runMLOCR(file);
      const manualInput = document.getElementById('manual-input');
      if (manualInput) {
        manualInput.value = text.trim();
        // Scroll to textarea so user can review
        manualInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        manualInput.focus();
      }

      // Collapse upload card, show mini preview accordion
      if (uploadCard) uploadCard.classList.add('upload-card--collapsed');
      if (uploadMini) {
        uploadMini.hidden = false;
        uploadMini.removeAttribute('open'); // start collapsed
        const miniImg = document.getElementById('upload-mini-img');
        const fullImg = document.getElementById('upload-mini-full');
        if (miniImg && previewImg) miniImg.src = previewImg.src;
        if (fullImg && previewImg) fullImg.src = previewImg.src;
      }

      // Pulse the analyze button to draw attention
      const scanBtn = document.getElementById('scan-btn');
      if (scanBtn) {
        scanBtn.classList.add('scan-btn-pulse');
        setTimeout(() => scanBtn.classList.remove('scan-btn-pulse'), 3000);
      }

      if (text.trim()) {
        // DO NOT auto-scan — let user review and edit first
        showToast(I18n.t('scanner.ocr.review', 'Text extracted! Review and click Analyze.'));
      } else {
        showToast(I18n.t('ocr.no.text', 'Could not extract text. Try pasting manually.'));
      }
    } catch (err) {
      console.error('[OCR Error]', err);
      showToast(I18n.t('ocr.failed', 'OCR failed. Please paste ingredients manually.'));
    }
  }

  return { init, scan, showResults, restoreResults, openModal, closeModal };
})();