/* ═══════════════════════════════════════════════════════════════════════
   Feed Ration Optimizer — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════ */

let appState = {
    ingredients: {},   // {key: {enabled, price, name, category}}
    result: null,      // last optimization result
    defaults: null,    // default ingredient data from API
};

// ── Initialization ───────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/api/defaults');
        const data = await res.json();
        appState.defaults = data.ingredients;
        renderIngredientGrid(data.ingredients);
    } catch (e) {
        console.error('Failed to load defaults:', e);
    }
});

function renderIngredientGrid(ingredients) {
    const grid = document.getElementById('ingredient-grid');
    grid.innerHTML = '';

    ingredients.forEach(ing => {
        appState.ingredients[ing.key] = {
            enabled: true,
            price: ing.price,
            name: ing.name,
            category: ing.category,
        };

        const card = document.createElement('div');
        card.className = 'ing-card active';
        card.dataset.key = ing.key;
        card.innerHTML = `
            <div class="ing-header">
                <span class="ing-name">${ing.name}</span>
                <div class="ing-toggle"></div>
            </div>
            <div class="ing-price">
                <input type="number" value="${ing.price}" step="10"
                       onchange="updateIngPrice('${ing.key}', this.value)">
                <span class="unit">$/ton</span>
            </div>
        `;

        // Toggle on click (but not on the price input)
        card.addEventListener('click', (e) => {
            if (e.target.tagName === 'INPUT') return;
            const state = appState.ingredients[ing.key];
            state.enabled = !state.enabled;
            card.classList.toggle('active', state.enabled);
        });

        grid.appendChild(card);
    });
}

function updateIngPrice(key, val) {
    appState.ingredients[key].price = parseFloat(val) || 0;
}


// ── Optimization ─────────────────────────────────────────────────────

async function runOptimize() {
    const btn = document.getElementById('optimize-btn');
    btn.disabled = true;
    btn.textContent = 'Optimizing...';

    // Collect inputs
    const payload = buildPayload();

    // Switch to results view
    document.getElementById('setup-view').style.display = 'none';
    document.getElementById('results-view').style.display = 'block';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results-content').style.display = 'none';

    try {
        const res = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json();
            alert(err.error || 'Optimization failed');
            showSetup();
            return;
        }

        appState.result = await res.json();

        // Must display block BEFORE renderResults so the canvas has a layout dimension!
        document.getElementById('loading').style.display = 'none';
        document.getElementById('results-content').style.display = 'block';

        renderResults(appState.result);

    } catch (e) {
        alert('Server error: ' + e.message);
        showSetup();
    } finally {
        btn.disabled = false;
        btn.textContent = 'Optimize My Ration';
    }
}

function buildPayload() {
    const ings = {};
    for (const [key, val] of Object.entries(appState.ingredients)) {
        ings[key] = { enabled: val.enabled, price: val.price };
    }

    return {
        ingredients: ings,
        cattle: {
            head_count: intVal('head-count', 100),
            start_weight: intVal('start-weight', 800),
            target_weight: intVal('target-weight', 1350),
            target_adg: 3.0,
        },
        economics: {
            purchase_cwt: floatVal('eco-purchase', 370),
            sale_cwt: floatVal('eco-sale', 240),
            yardage: floatVal('eco-yardage', 0.55),
            interest_rate: floatVal('eco-interest', 8),
            death_loss: floatVal('eco-death', 1.5),
            vet_cost: floatVal('eco-vet', 20),
            freight: 4.0,
            transit_shrink: 3.0,
            pencil_shrink: 4.0,
        },
    };
}

function showSetup() {
    document.getElementById('setup-view').style.display = 'block';
    document.getElementById('results-view').style.display = 'none';
}


// ── Render Results ───────────────────────────────────────────────────

function renderResults(r) {
    // KPIs
    const profit = r.economics.profit_per_head;
    const profitEl = document.getElementById('kpi-profit-val');
    profitEl.textContent = '$' + Math.abs(profit).toLocaleString();
    if (profit < 0) profitEl.textContent = '-' + profitEl.textContent;
    profitEl.className = 'kpi-value ' + (profit >= 0 ? 'profit-positive' : 'profit-negative');

    document.getElementById('kpi-feed').textContent = '$' + r.feed_cost_per_day.toFixed(2);

    const exitDay = r.optimal_exit.day;
    const sellDate = r.optimal_exit.sell_date || ('Day ' + exitDay);
    document.getElementById('kpi-sell-day').textContent = sellDate;
    document.getElementById('kpi-sell-sub').textContent =
        `Day ${exitDay} at ${r.optimal_exit.weight.toLocaleString()} lb`;

    // Traffic Light Sell Signal
    const mc = r.economics.current_mc || 0;
    const mr = r.economics.current_mr || 0;
    const signalEl = document.getElementById('kpi-signal');
    const signalSubEl = document.getElementById('kpi-signal-sub');

    if (mr >= mc) {
        signalEl.textContent = 'Keep Feeding';
        signalEl.className = 'kpi-value traffic-light-val signal-green';
        signalSubEl.textContent = `MR ($${mr.toFixed(2)}) ≥ MC ($${mc.toFixed(2)})`;
    } else {
        signalEl.textContent = 'Sell Now';
        signalEl.className = 'kpi-value traffic-light-val signal-red';
        signalSubEl.textContent = `MR ($${mr.toFixed(2)}) < MC ($${mc.toFixed(2)})`;
    }

    // Insight banner
    const insightArea = document.getElementById('results-content');
    const existingBanner = insightArea.querySelector('.insight-banner');
    if (existingBanner) existingBanner.remove();

    if (profit >= 0) {
        const banner = document.createElement('div');
        banner.className = 'insight-banner';
        banner.textContent = `At $${r.feed_cost_per_day.toFixed(2)}/day feed cost and $${floatVal('eco-sale', 240)}/cwt sale price, you're projected to make $${profit.toLocaleString()}/head over ${Math.round(r.economics.days_on_feed)} days. Breakeven: $${r.economics.breakeven_cwt}/cwt.`;
        insightArea.insertBefore(banner, insightArea.querySelector('.kpi-row'));
    } else {
        const banner = document.createElement('div');
        banner.className = 'insight-banner warning';
        banner.textContent = `At current prices, you'd lose $${Math.abs(profit).toLocaleString()}/head. You need $${r.economics.breakeven_cwt}/cwt to break even. Consider adjusting ADG or waiting for better cattle prices.`;
        insightArea.insertBefore(banner, insightArea.querySelector('.kpi-row'));
    }

    renderRecipe(r);
    renderPhases(r);
    renderExitChart(r);
    renderSensitivity(r);
    renderWhatIf(r);
}


// ── Recipe Bars ──────────────────────────────────────────────────────

function renderRecipe(r) {
    const container = document.getElementById('recipe-bars');
    container.innerHTML = '';

    const maxLb = Math.max(...r.ration.map(x => x.as_fed_lb));

    r.ration.forEach(item => {
        const pct = (item.as_fed_lb / maxLb) * 100;
        const row = document.createElement('div');
        row.className = 'recipe-row';
        row.innerHTML = `
            <span class="recipe-name">${item.name}</span>
            <div class="recipe-bar-wrap">
                <div class="recipe-bar ${item.category}" style="width:${pct}%"></div>
            </div>
            <span class="recipe-amount">${item.as_fed_lb} lb</span>
        `;
        container.appendChild(row);
    });

    document.getElementById('recipe-sub').textContent =
        `$${r.feed_cost_per_day.toFixed(2)}/head/day`;

    document.getElementById('recipe-total').innerHTML =
        `Total: <strong>${r.total_as_fed_lb} lb as-fed/head/day</strong>`;

    document.getElementById('recipe-batch').innerHTML =
        `Mixer batch (${r.head_count} head): <strong>${(r.total_as_fed_lb * r.head_count).toLocaleString()} lb as-fed</strong>`;
}


// ── Phase Timeline ───────────────────────────────────────────────────

function renderPhases(r) {
    const container = document.getElementById('phase-timeline');
    container.innerHTML = '';

    if (!r.phases || r.phases.length === 0) {
        container.innerHTML = '<p class="hint">No feeding program data available.</p>';
        return;
    }

    r.phases.forEach(phase => {
        // Aggregate by category
        const cats = { grain: 0, roughage: 0, protein: 0, supplement: 0 };
        (phase.ration || []).forEach(item => {
            const cat = item.category || 'grain';
            cats[cat] = (cats[cat] || 0) + item.pct;
        });

        const col = document.createElement('div');
        col.className = 'phase-col';

        let stackHTML = '';
        for (const [cat, pct] of Object.entries(cats)) {
            if (pct > 0) {
                const label = pct >= 8 ? `${cat.charAt(0).toUpperCase() + cat.slice(1)} ${Math.round(pct)}%` : '';
                stackHTML += `<div class="phase-segment ${cat}" style="flex:${pct}">${label}</div>`;
            }
        }

        const costStr = phase.cost_per_day ? `$${phase.cost_per_day.toFixed(2)}/d` : '--';

        col.innerHTML = `
            <div class="phase-stack">${stackHTML}</div>
            <div class="phase-label">${phase.name}</div>
            <div class="phase-days">Day ${phase.day_start}-${phase.day_end}</div>
            <div class="phase-cost">${costStr}</div>
        `;
        container.appendChild(col);
    });
}


// ── Exit Chart (Canvas) ──────────────────────────────────────────────

function renderExitChart(r) {
    const canvas = document.getElementById('exit-chart');
    const ctx = canvas.getContext('2d');

    // Handle retina
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = 260 * dpr;
    canvas.style.height = '260px';
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = 260;
    const pad = { top: 20, right: 20, bottom: 40, left: 55 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    const curve = r.exit_curve || [];
    const baseline = r.exit_baseline || [];
    if (curve.length === 0) return;

    // Find ranges
    const allProfits = [...curve.map(d => d.profit), ...baseline.map(d => d.profit)];
    const days = curve.map(d => d.day);
    const minDay = days[0];
    const maxDay = days[days.length - 1];
    const minProfit = Math.min(...allProfits);
    const maxProfit = Math.max(...allProfits);
    const profitRange = maxProfit - minProfit || 1;

    const xScale = d => pad.left + (d - minDay) / (maxDay - minDay) * plotW;
    const yScale = p => pad.top + (1 - (p - minProfit) / profitRange) * plotH;

    // Grid lines
    ctx.strokeStyle = '#E5E2DC';
    ctx.lineWidth = 0.5;
    const nTicks = 5;
    for (let i = 0; i <= nTicks; i++) {
        const y = pad.top + (i / nTicks) * plotH;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();

        const val = maxProfit - (i / nTicks) * profitRange;
        ctx.fillStyle = '#9B9B9B';
        ctx.font = '11px system-ui';
        ctx.textAlign = 'right';
        ctx.fillText('$' + Math.round(val).toLocaleString(), pad.left - 8, y + 4);
    }

    // X axis labels
    ctx.textAlign = 'center';
    const step = Math.ceil((maxDay - minDay) / 6);
    for (let d = minDay; d <= maxDay; d += step) {
        const x = xScale(d);
        ctx.fillStyle = '#9B9B9B';
        ctx.fillText('Day ' + d, x, H - 8);
    }

    // Zero line
    if (minProfit < 0 && maxProfit > 0) {
        const y0 = yScale(0);
        ctx.strokeStyle = '#CCC';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(pad.left, y0);
        ctx.lineTo(W - pad.right, y0);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#9B9B9B';
        ctx.textAlign = 'right';
        ctx.fillText('$0', pad.left - 8, y0 + 4);
    }

    // Baseline curve (dashed gray)
    if (r.has_price_forecast && baseline.length > 0) {
        ctx.strokeStyle = '#CCC';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        baseline.forEach((d, i) => {
            const x = xScale(d.day);
            const y = yScale(d.profit);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Main profit curve
    ctx.strokeStyle = '#2E7D32';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    curve.forEach((d, i) => {
        const x = xScale(d.day);
        const y = yScale(d.profit);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Optimal star
    const opt = r.optimal_exit;
    if (opt) {
        const ox = xScale(opt.day);
        const oy = yScale(opt.profit);
        drawStar(ctx, ox, oy, 8, '#D4A017', '#1A1A1A');

        ctx.fillStyle = '#1A1A1A';
        ctx.font = 'bold 12px system-ui';
        ctx.textAlign = 'left';
        ctx.fillText(`Day ${opt.day}: $${opt.profit.toLocaleString()}/hd`, ox + 12, oy - 4);
    }

    // Exit metrics
    const metricsEl = document.getElementById('exit-metrics');
    metricsEl.innerHTML = '';
    if (opt) {
        metricsEl.innerHTML = `
            <div class="exit-metric">Optimal exit: <strong>Day ${opt.day}</strong></div>
            <div class="exit-metric">At weight: <strong>${opt.weight.toLocaleString()} lb</strong></div>
            <div class="exit-metric">ADG at exit: <strong>${opt.adg_at_exit} lb/d</strong></div>
            <div class="exit-metric">Peak profit: <strong>$${opt.profit.toLocaleString()}/hd</strong></div>
        `;
    }

    // Insight
    const insightEl = document.getElementById('exit-insight');
    if (r.has_price_forecast) {
        insightEl.textContent = 'Includes cattle price forecast from futures';
    } else {
        insightEl.textContent = 'Constant sale price assumed';
    }

    // Legend
    const legendEl = document.getElementById('exit-legend');
    let legendHTML = `<div class="legend-item"><div class="legend-dot" style="background:#2E7D32"></div> Profit</div>`;
    if (r.has_price_forecast) {
        legendHTML += `<div class="legend-item"><div class="legend-dot" style="background:#CCC"></div> Without price forecast</div>`;
    }
    legendHTML += `<div class="legend-item"><div class="legend-dot" style="background:#D4A017;height:8px;width:8px;border-radius:50%"></div> Optimal exit</div>`;
    legendEl.innerHTML = legendHTML;
}

function drawStar(ctx, cx, cy, r, fill, stroke) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
        const radius = i % 2 === 0 ? r : r * 0.45;
        const angle = (i * Math.PI / 5) - Math.PI / 2;
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
}


// ── Sensitivity ──────────────────────────────────────────────────────

function renderSensitivity(r) {
    const container = document.getElementById('sensitivity-bars');
    container.innerHTML = '';

    if (!r.sensitivity || r.sensitivity.length === 0) return;

    const maxAbs = Math.max(...r.sensitivity.map(s => Math.abs(s.impact_per_head)));

    r.sensitivity.forEach(s => {
        const pct = (Math.abs(s.impact_per_head) / maxAbs) * 45; // max 45% of width
        const isPos = s.impact_per_head > 0;
        const sign = isPos ? '+' : '-';
        const colorClass = isPos ? 'positive' : 'negative';

        const row = document.createElement('div');
        row.className = 'sens-row';
        row.innerHTML = `
            <span class="sens-label">${s.parameter}</span>
            <div class="sens-bar-wrap">
                <div class="sens-bar-track"></div>
                <div class="sens-bar ${colorClass}" style="width:${pct}%"></div>
            </div>
            <span class="sens-value" style="color:${isPos ? 'var(--green)' : 'var(--red)'}">
                ${sign}$${Math.abs(s.impact_per_head).toLocaleString()}
            </span>
        `;
        container.appendChild(row);
    });
}


// ── What-If Sliders ──────────────────────────────────────────────────

let whatIfTimeout = null;

function renderWhatIf(r) {
    const container = document.getElementById('whatif-sliders');
    container.innerHTML = '';

    // Show sliders for active ingredients + ADG
    const activeIngs = r.ration.map(item => {
        const ing = appState.ingredients[item.key];
        return { key: item.key, name: item.name, price: ing ? ing.price : 0 };
    }).filter(x => x.price > 0);

    // ADG slider
    const adgGroup = document.createElement('div');
    adgGroup.className = 'whatif-slider-group';
    adgGroup.innerHTML = `
        <label>Target ADG <span class="price-val" id="adg-val">${r.target_adg.toFixed(1)} lb/d</span></label>
        <input type="range" min="1.5" max="4.5" step="0.1" value="${r.target_adg}"
               oninput="document.getElementById('adg-val').textContent=parseFloat(this.value).toFixed(1)+' lb/d'; debouncedWhatIf()">
    `;
    container.appendChild(adgGroup);

    // Ingredient price sliders
    activeIngs.forEach(ing => {
        const min = Math.round(ing.price * 0.5);
        const max = Math.round(ing.price * 1.8);
        const group = document.createElement('div');
        group.className = 'whatif-slider-group';
        group.dataset.key = ing.key;
        group.innerHTML = `
            <label>${ing.name} <span class="price-val" id="wi-${ing.key}-val">$${ing.price}/ton</span></label>
            <input type="range" min="${min}" max="${max}" step="5" value="${ing.price}"
                   oninput="document.getElementById('wi-${ing.key}-val').textContent='$'+this.value+'/ton'; debouncedWhatIf()">
        `;
        container.appendChild(group);
    });
}

function debouncedWhatIf() {
    clearTimeout(whatIfTimeout);
    whatIfTimeout = setTimeout(runWhatIf, 400);
}

async function runWhatIf() {
    const r = appState.result;
    if (!r) return;

    // Collect current slider values
    const sliderGroups = document.querySelectorAll('.whatif-slider-group');
    const modifiedIngs = {};
    let newAdg = r.target_adg;

    sliderGroups.forEach(group => {
        const slider = group.querySelector('input[type="range"]');
        const key = group.dataset.key;
        if (!key) {
            // ADG slider
            newAdg = parseFloat(slider.value);
        } else {
            modifiedIngs[key] = parseFloat(slider.value);
        }
    });

    // Build payload with modified prices
    const ings = {};
    for (const [key, val] of Object.entries(appState.ingredients)) {
        ings[key] = {
            enabled: val.enabled,
            price: modifiedIngs[key] !== undefined ? modifiedIngs[key] : val.price,
        };
    }

    const payload = {
        ingredients: ings,
        cattle: {
            start_weight: r.cattle.start_weight,
            target_weight: r.cattle.target_weight,
            target_adg: newAdg,
        },
        economics: buildPayload().economics,
    };

    try {
        const res = await fetch('/api/scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            document.getElementById('whatif-result').innerHTML =
                '<div class="whatif-placeholder">No feasible ration at these prices/ADG.</div>';
            return;
        }

        const scenario = await res.json();
        renderWhatIfResult(r, scenario, newAdg);

    } catch (e) {
        console.error('What-if failed:', e);
    }
}

function renderWhatIfResult(original, scenario, newAdg) {
    const container = document.getElementById('whatif-result');

    const profitDiff = scenario.economics.profit_per_head - original.economics.profit_per_head;
    const costDiff = scenario.feed_cost_per_day - original.feed_cost_per_day;
    const daysDiff = scenario.economics.days_on_feed - original.economics.days_on_feed;

    const profitSign = profitDiff >= 0 ? '+' : '';
    const profitClass = profitDiff >= 0 ? 'down' : 'up'; // green if more profit

    let html = `<div class="whatif-comparison">`;
    html += `<div class="comp-header">Scenario Impact</div>`;

    html += `<div class="comp-row">
        <span>Profit</span>
        <span class="comp-change ${profitClass}">$${scenario.economics.profit_per_head.toLocaleString()}/hd (${profitSign}$${Math.round(profitDiff).toLocaleString()})</span>
    </div>`;

    html += `<div class="comp-row">
        <span>Feed cost</span>
        <span>$${scenario.feed_cost_per_day.toFixed(2)}/day</span>
    </div>`;

    html += `<div class="comp-row">
        <span>Days on feed</span>
        <span>${Math.round(scenario.economics.days_on_feed)} days</span>
    </div>`;

    html += `<div class="comp-row">
        <span>Breakeven</span>
        <span>$${scenario.economics.breakeven_cwt}/cwt</span>
    </div>`;

    // Show ration changes
    html += `<div class="comp-header" style="margin-top:12px">Ration Changes</div>`;
    scenario.ration.forEach(item => {
        const orig = original.ration.find(o => o.key === item.key);
        const origLb = orig ? orig.as_fed_lb : 0;
        const diff = item.as_fed_lb - origLb;
        let changeStr = '';
        if (Math.abs(diff) >= 0.1) {
            const sign = diff > 0 ? '+' : '';
            const cls = diff > 0 ? 'up' : 'down';
            changeStr = `<span class="comp-change ${cls}">${sign}${diff.toFixed(1)} lb</span>`;
        }
        html += `<div class="comp-row">
            <span>${item.name}</span>
            <span>${item.as_fed_lb} lb ${changeStr}</span>
        </div>`;
    });

    // New ingredients that weren't in original
    original.ration.forEach(orig => {
        if (!scenario.ration.find(s => s.key === orig.key)) {
            html += `<div class="comp-row">
                <span>${orig.name}</span>
                <span class="comp-change down">removed</span>
            </div>`;
        }
    });

    html += `</div>`;
    container.innerHTML = html;
}


// ── Utilities ────────────────────────────────────────────────────────

function intVal(id, fallback) {
    const el = document.getElementById(id);
    return el ? parseInt(el.value) || fallback : fallback;
}

function floatVal(id, fallback) {
    const el = document.getElementById(id);
    return el ? parseFloat(el.value) || fallback : fallback;
}

// ── Bunk Management Log ──────────────────────────────────────────────

function setBunkScore(score) {
    // UI update
    document.querySelectorAll('.bunk-btn').forEach((btn, i) => {
        if (i === score) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });

    // Provide feedback
    const msg = document.getElementById('bunk-feedback');
    const hints = [
        "Score 0: Bunk is completely empty. Consider increasing feed call by 5%.",
        "Score 1: Perfect. Negligible crumbs remaining. Hold intake.",
        "Score 2: Acceptable. Thin layer of feed scattered. Monitor next feeding.",
        "Score 3: Overfed or sick. 25% feed remaining. Decrease feed call by 10%.",
        "Score 4: Untouched. Check water, weather, or acidosis immediately."
    ];
    msg.textContent = hints[score];
    msg.style.display = 'block';

    // Animate
    msg.style.animation = 'none';
    msg.offsetHeight; // trigger reflow
    msg.style.animation = 'fadeInUp 0.3s ease forwards';
}
