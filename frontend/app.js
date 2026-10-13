// app.js - draws everything on the page from window.IRIS (exported by export_data.py)
(function () {
  const D = window.IRIS, M = window.IrisModels;
  const F = D.features, CLS = D.classes, R = D.results;
  const COLORS = ["#7c5cf0", "#0d9488", "#e8735a"];
  const KEYS = ["lr", "knn", "dt"];
  const $ = (s) => document.querySelector(s);
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const f2 = (v) => Number(v).toFixed(2);
  const f3 = (v) => Number(v).toFixed(3);
  const pct = (v) => (v * 100).toFixed(1) + "%";

  // ---------- helpers ----------
  function marker(cls, x, y, r, extra) {
    const a = `fill="${COLORS[cls]}" fill-opacity=".78" stroke="#fff" stroke-width="1" ${extra || ""}`;
    if (cls === 0) return `<circle cx="${x}" cy="${y}" r="${r}" ${a}>`;
    if (cls === 1) return `<polygon points="${x},${y - r * 1.25} ${x - r * 1.15},${y + r * .9} ${x + r * 1.15},${y + r * .9}" ${a}>`;
    return `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" ${a}>`;
  }
  function markerEnd(cls) { return cls === 0 ? "</circle>" : cls === 1 ? "</polygon>" : "</rect>"; }
  function niceTicks(lo, hi, n = 5) {
    const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }
  const legendHTML = () => CLS.map((c, i) =>
    `<span><svg width="16" height="16" viewBox="-8 -8 16 16">${marker(i, 0, 0, 5)}${markerEnd(i)}</svg>${cap(c)}</span>`).join("");

  // ---------- scatter ----------
  // opts: {small, user:[x,y], neighbors:[ids]}
  function scatter(host, xi, yi, opts = {}) {
    const small = !!opts.small;
    const W = small ? 260 : 540, H = small ? 210 : 420;
    const m = small ? { l: 34, r: 8, t: 8, b: 32 } : { l: 52, r: 16, t: 14, b: 48 };
    let xs = D.rows.map((r) => r[xi]), ys = D.rows.map((r) => r[yi]);
    if (opts.user) { xs = xs.concat(opts.user[xi]); ys = ys.concat(opts.user[yi]); }
    const pad = (a) => (Math.max(...a) - Math.min(...a)) * 0.07;
    const x0 = Math.min(...xs) - pad(xs), x1 = Math.max(...xs) + pad(xs);
    const y0 = Math.min(...ys) - pad(ys), y1 = Math.max(...ys) + pad(ys);
    const sx = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
    const sy = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Scatter plot of ${F[xi].short} vs ${F[yi].short}">`;
    niceTicks(x0, x1, small ? 4 : 6).forEach((t) => {
      s += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${m.t}" y2="${H - m.b}" stroke="#eee"/><text x="${sx(t)}" y="${H - m.b + 14}" text-anchor="middle">${t}</text>`;
    });
    niceTicks(y0, y1, small ? 4 : 5).forEach((t) => {
      s += `<line y1="${sy(t)}" y2="${sy(t)}" x1="${m.l}" x2="${W - m.r}" stroke="#eee"/><text x="${m.l - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
    });
    s += `<text class="axis-label" x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${small ? F[xi].short : F[xi].label}</text>`;
    s += `<text class="axis-label" transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${small ? F[yi].short : F[yi].label}</text>`;
    if (opts.user && opts.neighbors) {
      opts.neighbors.forEach((id) => {
        const r = D.rows[id];
        s += `<line x1="${sx(opts.user[xi])}" y1="${sy(opts.user[yi])}" x2="${sx(r[xi])}" y2="${sy(r[yi])}" stroke="#333" stroke-width="1" stroke-dasharray="3 3" opacity=".6"/>`;
      });
    }
    const hi = new Set(opts.neighbors || []);
    D.rows.forEach((r, id) => {
      s += marker(r[4], sx(r[xi]), sy(r[yi]), small ? 3 : hi.has(id) ? 6.5 : 4.5,
        hi.has(id) ? 'stroke="#222" stroke-width="2"' : "") +
        `<title>Flower #${id}: ${cap(CLS[r[4]])} (${r[xi]}, ${r[yi]})</title>` + markerEnd(r[4]);
    });
    if (opts.user) {
      const ux = sx(opts.user[xi]), uy = sy(opts.user[yi]);
      const star = Array.from({ length: 10 }, (_, i) => {
        const a = (Math.PI / 5) * i - Math.PI / 2, rr = i % 2 ? 6 : 14;
        return `${ux + rr * Math.cos(a)},${uy + rr * Math.sin(a)}`;
      }).join(" ");
      s += `<polygon points="${star}" fill="#ffd23f" stroke="#111" stroke-width="1.5"><title>Your flower</title></polygon>`;
    }
    host.innerHTML = s + "</svg>";
  }

  // ---------- histogram ----------
  function histogram(host, fi) {
    const bins = 12, vals = D.rows.map((r) => r[fi]);
    const lo = Math.min(...vals), hi = Math.max(...vals), bw = (hi - lo) / bins;
    const counts = [0, 1, 2].map(() => new Array(bins).fill(0));
    D.rows.forEach((r) => counts[r[4]][Math.min(bins - 1, Math.floor((r[fi] - lo) / bw))]++);
    const ymax = Math.max(...counts.flat());
    const W = 260, H = 190, m = { l: 30, r: 8, t: 8, b: 30 };
    const sx = (v) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
    const sy = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of ${F[fi].short}">`;
    niceTicks(0, ymax, 4).forEach((t) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}" stroke="#eee"/><text x="${m.l - 5}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`; });
    [0, 1, 2].forEach((c) => counts[c].forEach((n, b) => {
      if (!n) return;
      s += `<rect x="${sx(lo + b * bw) + 1}" y="${sy(n)}" width="${sx(lo + bw) - sx(lo) - 2}" height="${sy(0) - sy(n)}" fill="${COLORS[c]}" fill-opacity=".55"><title>${cap(CLS[c])}: ${n} flowers</title></rect>`;
    }));
    niceTicks(lo, hi, 4).forEach((t) => { s += `<text x="${sx(t)}" y="${H - m.b + 14}" text-anchor="middle">${t}</text>`; });
    s += `<text class="axis-label" x="${W / 2}" y="${H - 4}" text-anchor="middle">cm</text></svg>`;
    host.innerHTML = `<p class="hist-title">${F[fi].short}</p>` + s;
  }

  // ---------- dots for split ----------
  const dots = (ids) => ids.map((id) => `<span class="dot s${D.rows[id][4]}" title="Flower #${id}: ${cap(CLS[D.rows[id][4]])}"></span>`).join("");
  const sortedByClass = (ids) => ids.slice().sort((a, b) => D.rows[a][4] - D.rows[b][4] || a - b);

  // ---------- confusion matrix ----------
  function confusion(cm) {
    const max = Math.max(...cm.flat());
    let t = `<table><tr><th></th><th colspan="3" class="axis">Predicted</th></tr><tr><th></th>${CLS.map((c) => `<th>${cap(c)}</th>`).join("")}</tr>`;
    cm.forEach((row, i) => {
      t += `<tr><th>${cap(CLS[i])}</th>`;
      row.forEach((v, j) => {
        const a = v / max, bg = i === j ? `rgba(76,175,125,${.15 + a * .75})` : v ? `rgba(224,85,61,${.25 + a * .6})` : "#f6f3ec";
        t += `<td class="cell" style="background:${bg};color:${a > .6 ? "#fff" : "#1f1b2e"}">${v}</td>`;
      });
      t += "</tr>";
    });
    return t + `</table><p class="axis">↑ True species (rows)</p>`;
  }

  // ====================================================================
  // HERO + STEP 1
  // ====================================================================
  $("#heroStats").innerHTML = [
    [D.meta.n, "flowers"], [4, "features"], [3, "species"],
    [D.split.train_counts.reduce((a, b) => a + b), "train samples"],
    [D.split.test_counts.reduce((a, b) => a + b), "test samples"],
  ].map(([n, l]) => `<div class="stat"><b>${n}</b><span>${l}</span></div>`).join("");
  $("#nRows").textContent = D.meta.n;
  $("#nDup").textContent = D.duplicates;
  $("#skv").textContent = D.meta.sklearn;
  $("#seedTxt").textContent = "random_state=" + D.split.seed;

  $("#overallStats").innerHTML = `<table><thead><tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr></thead><tbody>` +
    F.map((f, i) => `<tr><td>${f.short}</td><td class="num">${f2(D.overall.mean[i])}</td><td class="num">${f2(D.overall.std[i])}</td><td class="num">${D.overall.min[i]}</td><td class="num">${D.overall.max[i]}</td></tr>`).join("") + "</tbody></table>";

  CLS.forEach((c, i) => { $("#rowFilter").insertAdjacentHTML("beforeend", `<option value="${i}">${cap(c)}</option>`); });
  function drawRows() {
    const f = +$("#rowFilter").value;
    const rows = D.rows.map((r, id) => ({ r, id })).filter((x) => f < 0 || x.r[4] === f);
    $("#rowCount").textContent = `${rows.length} rows`;
    $("#rowTable").innerHTML = `<table><thead><tr><th>#</th>${F.map((x) => `<th>${x.short} (cm)</th>`).join("")}<th>Species</th></tr></thead><tbody>` +
      rows.map(({ r, id }) => `<tr><td>${id}</td>${r.slice(0, 4).map((v) => `<td class="num">${v}</td>`).join("")}<td style="text-align:left"><span class="chip c${r[4]}">${cap(CLS[r[4]])}</span></td></tr>`).join("") + "</tbody></table>";
  }
  $("#rowFilter").addEventListener("change", drawRows);
  drawRows();

  // ====================================================================
  // STEP 2 EXPLORE
  // ====================================================================
  $("#legend").innerHTML = legendHTML();
  [$("#selX"), $("#selY")].forEach((sel, k) => {
    F.forEach((f, i) => sel.insertAdjacentHTML("beforeend", `<option value="${i}" ${i === (k ? 3 : 2) ? "selected" : ""}>${f.label}</option>`));
  });
  function drawMain() {
    const xi = +$("#selX").value, yi = +$("#selY").value;
    scatter($("#mainScatter"), xi, yi);
    $("#scatterNote").innerHTML = `<b>What to look for</b><br>Do the three colours form separate clouds? ` +
      (xi === yi ? "You picked the same feature twice – choose two different ones." :
        `With <b>${F[xi].short.toLowerCase()}</b> and <b>${F[yi].short.toLowerCase()}</b> ` +
        (separationOf(xi, yi))) + `<br><br><span class="muted">Hover a point to see its details. Try the petal features, then the sepal features, and compare.</span>`;
  }
  // how well do the 2 chosen features split Versicolor/Virginica? (computed from class means & stds)
  function separationOf(xi, yi) {
    const gap = (fi) => Math.abs(D.stats[CLS[1]].mean[fi] - D.stats[CLS[2]].mean[fi]) /
      ((D.stats[CLS[1]].std[fi] + D.stats[CLS[2]].std[fi]) / 2);
    const g = Math.max(gap(xi), gap(yi)), setosaGap = (fi) => Math.abs(D.stats[CLS[0]].mean[fi] - D.stats[CLS[1]].mean[fi]) / ((D.stats[CLS[0]].std[fi] + D.stats[CLS[1]].std[fi]) / 2);
    return `Setosa is ${Math.max(setosaGap(xi), setosaGap(yi)) > 4 ? "clearly" : "only partly"} apart from the others, and the gap between Versicolor and Virginica is about <b>${g.toFixed(1)}</b> standard deviations (${g > 2.5 ? "easy to tell apart" : g > 1.5 ? "some overlap" : "heavy overlap"}).`;
  }
  $("#selX").addEventListener("change", drawMain);
  $("#selY").addEventListener("change", drawMain);
  drawMain();

  F.forEach((_, i) => { const d = document.createElement("div"); d.className = "chart"; $("#hists").appendChild(d); histogram(d, i); });
  [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]].forEach(([a, b]) => {
    const d = document.createElement("div"); d.className = "chart"; $("#pairs").appendChild(d); scatter(d, a, b, { small: true });
  });

  // heatmap
  $("#heatmap").innerHTML = `<table class="heat"><tr><th></th>${F.map((f) => `<th>${f.short.replace(" ", "<br>")}</th>`).join("")}</tr>` +
    D.corr.map((row, i) => `<tr><th>${F[i].short}</th>${row.map((v) => {
      const a = Math.abs(v), bg = v >= 0 ? `rgba(91,91,214,${a})` : `rgba(232,115,90,${a})`;
      return `<td style="background:${bg};color:${a > .55 ? "#fff" : "#1f1b2e"}">${v.toFixed(2)}</td>`;
    }).join("")}</tr>`).join("") + "</table>";

  // ANOVA bars
  const maxF = Math.max(...D.anova.map((a) => a.F));
  const rank = D.anova.map((a, i) => ({ ...a, i })).sort((a, b) => b.F - a.F);
  $("#anovaBars").innerHTML = rank.map((a) =>
    `<div class="bar-row"><span>${F[a.i].short}</span><div class="bar-track"><div class="bar" style="width:${(a.F / maxF) * 100}%"></div></div><span class="num">F = ${a.F.toFixed(0)}</span></div>`).join("") +
    `<p class="muted">All p-values are below ${Math.max(...D.anova.map((a) => a.p)).toExponential(0)}, so every feature differs between species; the ranking shows how strongly.</p>`;

  // class stats
  $("#classStats").innerHTML = `<table><thead><tr><th>Species</th>${F.map((f) => `<th>${f.short}</th>`).join("")}</tr></thead><tbody>` +
    CLS.map((c, i) => `<tr><td><span class="chip c${i}">${cap(c)}</span></td>${F.map((_, k) => `<td class="num">${f2(D.stats[c].mean[k])} ± ${f2(D.stats[c].std[k])}</td>`).join("")}</tr>`).join("") + "</tbody></table><p class='muted'>mean ± standard deviation, in cm</p>";

  const top2 = rank.slice(0, 2).map((r) => F[r.i].short.toLowerCase());
  const sepSetosaPetal = D.stats[CLS[0]].max[2] < Math.min(D.stats[CLS[1]].min[2], D.stats[CLS[2]].min[2]);
  $("#exploreInsight").innerHTML = `<b>What the data tells us</b><ul>
    <li>The strongest separating features are <b>${top2[0]}</b> and <b>${top2[1]}</b> (highest ANOVA F-scores).</li>
    <li>${sepSetosaPetal ? `Every Setosa has a petal length of at most <b>${D.stats[CLS[0]].max[2]} cm</b>, while Versicolor and Virginica are at least <b>${Math.min(D.stats[CLS[1]].min[2], D.stats[CLS[2]].min[2])} cm</b>: Setosa is perfectly separable.` : "Setosa overlaps with the others in petal length."}</li>
    <li>Versicolor and Virginica overlap a little – this is where a classifier can make mistakes.</li>
    <li>Petal length and petal width are strongly correlated (r = ${D.corr[2][3].toFixed(2)}), so they carry similar information.</li></ul>`;

  // ====================================================================
  // STEP 3 SPLIT
  // ====================================================================
  const nTrain = D.split.train_idx.length, nTest = D.split.test_idx.length;
  $("#trainN").textContent = `(${nTrain} flowers)`;
  $("#testN").textContent = `(${nTest} flowers)`;
  $("#trainDots").innerHTML = dots(sortedByClass(D.split.train_idx));
  $("#testDots").innerHTML = dots(sortedByClass(D.split.test_idx));
  $("#splitTable").innerHTML = `<table><thead><tr><th>Species</th><th>Total</th><th>Training</th><th>Test</th></tr></thead><tbody>` +
    CLS.map((c, i) => `<tr><td><span class="chip c${i}">${cap(c)}</span></td><td class="num">${D.split.train_counts[i] + D.split.test_counts[i]}</td><td class="num">${D.split.train_counts[i]}</td><td class="num">${D.split.test_counts[i]}</td></tr>`).join("") +
    `<tr><td><b>All</b></td><td class="num"><b>${nTrain + nTest}</b></td><td class="num"><b>${nTrain}</b></td><td class="num"><b>${nTest}</b></td></tr></tbody></table>
     <p class="muted">Stratification keeps the species equally represented in both sets.</p>`;
  $("#testSize").textContent = nTest;

  // ====================================================================
  // STEP 4 PREPROCESS
  // ====================================================================
  $("#scalerTable").innerHTML = `<table><thead><tr><th>Feature</th><th>Training mean</th><th>Training std</th><th>Example flower (raw)</th><th>Example (scaled z)</th></tr></thead><tbody>` +
    (() => {
      const ex = D.rows[D.split.test_idx[0]].slice(0, 4), z = M.scale(ex);
      $("#scalerExample").textContent = `Example: test flower #${D.split.test_idx[0]} (${cap(CLS[D.rows[D.split.test_idx[0]][4]])}) is converted with the training mean and std – the test set never influences these numbers.`;
      return F.map((f, i) => `<tr><td>${f.short}</td><td class="num">${f2(D.scaler.mean[i])}</td><td class="num">${f2(D.scaler.scale[i])}</td><td class="num">${ex[i]}</td><td class="num">${z[i].toFixed(2)}</td></tr>`).join("");
    })() + "</tbody></table>";

  // ====================================================================
  // STEP 5 TRAIN
  // ====================================================================
  const INFO = {
    lr: {
      tag: "Linear model",
      how: "Learns a weight for each feature and combines them into a score for every species. A softmax function turns the scores into probabilities.",
      params: "max_iter = 200 (default regularisation C = 1.0)",
      pros: ["Fast and simple", "Gives real probabilities", "Weights are easy to interpret"],
      cons: ["Assumes mostly straight-line boundaries"],
    },
    knn: {
      tag: "Instance-based",
      how: "Doesn't learn a formula. To classify a flower it finds the k most similar training flowers (closest in scaled feature space) and takes a majority vote.",
      params: "n_neighbors (k) = 5, Euclidean distance",
      pros: ["Very intuitive", "Handles curved boundaries"],
      cons: ["Needs scaling", "Slower with big datasets"],
    },
    dt: {
      tag: "Rule-based",
      how: "Learns a flowchart of yes/no questions such as “is petal length ≤ 2.45 cm?”. Each question splits the flowers into purer groups.",
      params: "max_depth = 3 (limits the tree so it doesn't overfit)",
      pros: ["Easy to read and explain", "No scaling needed"],
      cons: ["Can overfit if too deep", "Sensitive to small data changes"],
    },
  };
  $("#modelCards").innerHTML = KEYS.map((k) => `<div class="model-card">
      <span class="tag">${INFO[k].tag}</span><h3 style="margin-top:8px">${R[k].name}</h3>
      <p>${INFO[k].how}</p><p><b>Settings:</b> ${INFO[k].params}</p>
      <p><b>Strengths</b></p><ul>${INFO[k].pros.map((x) => `<li>${x}</li>`).join("")}</ul>
      <p><b>Watch out</b></p><ul>${INFO[k].cons.map((x) => `<li>${x}</li>`).join("")}</ul>
      <p class="muted">Test accuracy</p><div class="big">${pct(R[k].accuracy)}</div></div>`).join("");

  function treeHTML(n, label) {
    const lead = label ? `<span class="branch">${label}</span>` : "";
    if (n.leaf) return `<li>${lead}<span class="leaf" style="background:${COLORS[n.cls]}">${cap(CLS[n.cls])}</span> <span class="muted">n = ${n.samples} (${n.counts.join(" / ")})</span></li>`;
    return `<li>${lead}<span class="q">${F[n.feature].short} ≤ ${n.threshold.toFixed(2)} cm?</span> <span class="muted">n = ${n.samples}</span><ul>${treeHTML(n.left, "yes →")}${treeHTML(n.right, "no →")}</ul></li>`;
  }
  $("#treeView").innerHTML = `<ul>${treeHTML(D.tree)}</ul><p class="muted">Counts in brackets are Setosa / Versicolor / Virginica.</p>`;

  // ====================================================================
  // STEP 6 EVALUATE
  // ====================================================================
  const bestAcc = Math.max(...KEYS.map((k) => R[k].accuracy));
  const cols = ["accuracy", "precision", "recall", "f1"];
  $("#metricTable").innerHTML = `<table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th></tr></thead><tbody>` +
    KEYS.map((k) => `<tr><td>${R[k].name}</td>${cols.map((c) => `<td class="num ${R[k][c] === Math.max(...KEYS.map((q) => R[q][c])) ? "best" : ""}">${f3(R[k][c])}</td>`).join("")}</tr>`).join("") + "</tbody></table><p class='muted'>Precision, recall and F1 are macro-averaged. Green = best in that column.</p>";

  // grouped bar chart
  (function () {
    const W = 640, H = 280, m = { l: 44, r: 12, t: 14, b: 54 };
    const gw = (W - m.l - m.r) / cols.length, bw = gw / (KEYS.length + 1);
    const lo = 0.8, sy = (v) => H - m.b - ((v - lo) / (1 - lo)) * (H - m.t - m.b);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Bar chart comparing models">`;
    [0.8, 0.85, 0.9, 0.95, 1].forEach((t) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}" stroke="#eee"/><text x="${m.l - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`; });
    cols.forEach((c, ci) => {
      KEYS.forEach((k, ki) => {
        const x = m.l + ci * gw + bw / 2 + ki * bw, v = R[k][c];
        s += `<rect x="${x}" y="${sy(v)}" width="${bw - 3}" height="${sy(lo) - sy(v)}" fill="${["#5b5bd6", "#0d9488", "#e8a23d"][ki]}" rx="3"><title>${R[k].name} ${c}: ${f3(v)}</title></rect>`;
      });
      s += `<text x="${m.l + ci * gw + gw / 2}" y="${H - m.b + 16}" text-anchor="middle" class="axis-label">${["Accuracy", "Precision", "Recall", "F1"][ci]}</text>`;
    });
    KEYS.forEach((k, ki) => { s += `<rect x="${m.l + ki * 190}" y="${H - 22}" width="12" height="12" rx="3" fill="${["#5b5bd6", "#0d9488", "#e8a23d"][ki]}"/><text x="${m.l + ki * 190 + 18}" y="${H - 12}">${R[k].name}</text>`; });
    $("#metricBars").innerHTML = s + `</svg><p class="muted">Axis starts at 0.80 to make small differences visible – remember they are small.</p>`;
  })();

  $("#confusions").innerHTML = KEYS.map((k) => `<div class="cm"><h4>${R[k].name}</h4>${confusion(R[k].confusion)}</div>`).join("");

  // worked example from the user's own numbers (Logistic Regression, Virginica)
  (function () {
    const cm = R.lr.confusion, c = 2;
    const tp = cm[c][c], fn = cm[c].reduce((a, b) => a + b, 0) - tp, fp = cm.reduce((a, r) => a + r[c], 0) - tp;
    $("#workedExample").innerHTML = `<b>Worked example – ${R.lr.name}, species ${cap(CLS[c])}</b><br>
      Correctly identified (TP) = <b>${tp}</b>, missed (FN) = <b>${fn}</b>, wrongly labelled as ${cap(CLS[c])} (FP) = <b>${fp}</b>.<br>
      Precision = ${tp} ÷ (${tp} + ${fp}) = <b>${f3(tp / (tp + fp))}</b> &nbsp;·&nbsp;
      Recall = ${tp} ÷ (${tp} + ${fn}) = <b>${f3(tp / (tp + fn))}</b>`;
  })();

  // per-class report with tabs
  function drawReport(k) {
    document.querySelectorAll("#reportTabs button").forEach((b) => b.classList.toggle("active", b.dataset.k === k));
    $("#reportTable").innerHTML = `<table><thead><tr><th>Species</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead><tbody>` +
      R[k].per_class.map((p, i) => `<tr><td><span class="chip c${i}">${cap(CLS[i])}</span></td><td class="num">${f2(p.precision)}</td><td class="num">${f2(p.recall)}</td><td class="num">${f2(p.f1)}</td><td class="num">${p.support}</td></tr>`).join("") + "</tbody></table><p class='muted'>Support = number of test flowers of that species.</p>";
  }
  $("#reportTabs").innerHTML = KEYS.map((k) => `<button data-k="${k}">${R[k].name}</button>`).join("");
  document.querySelectorAll("#reportTabs button").forEach((b) => b.addEventListener("click", () => drawReport(b.dataset.k)));
  drawReport("lr");

  // misclassified flowers
  (function () {
    const rows = [];
    D.split.test_idx.forEach((id, j) => {
      const t = D.rows[id][4], preds = KEYS.map((k) => R[k].test_pred[j]);
      if (preds.some((p) => p !== t)) rows.push({ id, t, preds });
    });
    const cell = (p, t) => `<td style="text-align:left">${p === t ? '<span class="muted">✓ correct</span>' : `<span class="chip c${p}">${cap(CLS[p])}</span>`}</td>`;
    $("#misTable").innerHTML = rows.length ? `<table><thead><tr><th>Flower #</th><th>Measurements (SL, SW, PL, PW)</th><th>True species</th>${KEYS.map((k) => `<th>${R[k].name}</th>`).join("")}</tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${r.id}</td><td class="num">${D.rows[r.id].slice(0, 4).join(", ")}</td><td style="text-align:left"><span class="chip c${r.t}">${cap(CLS[r.t])}</span></td>${r.preds.map((p) => cell(p, r.t)).join("")}</tr>`).join("") + "</tbody></table><p class='muted'>Only test flowers that at least one model got wrong are listed. As expected, the mistakes are between Versicolor and Virginica.</p>"
      : "<p>No model made a mistake on the test set.</p>";
  })();

  $("#cvTable").innerHTML = `<table><thead><tr><th>Model</th><th>Test accuracy (1 split)</th><th>5-fold CV accuracy (mean ± std)</th><th>Fold scores</th></tr></thead><tbody>` +
    KEYS.map((k) => `<tr><td>${R[k].name}</td><td class="num">${f3(R[k].accuracy)}</td><td class="num">${f3(R[k].cv_mean)} ± ${f3(R[k].cv_std)}</td><td class="num">${R[k].cv_folds.map(f2).join(", ")}</td></tr>`).join("") + "</tbody></table>";

  // ====================================================================
  // PREDICT
  // ====================================================================
  const range = F.map((_, i) => ({ min: 0.1, max: Math.ceil(D.overall.max[i] * 1.6), step: 0.1 }));
  let cur = D.stats[CLS[1]].mean.map((v) => +v.toFixed(1));
  const presetDefs = [
    ...CLS.map((c, i) => ({ label: `${cap(c)} (average)`, v: () => D.stats[c].mean.map((x) => +x.toFixed(1)) })),
    { label: "🎲 Random test flower", v: () => D.rows[D.split.test_idx[Math.floor(Math.random() * nTest)]].slice(0, 4) },
    { label: "🚫 Unusual flower (huge)", v: () => [12.0, 1.0, 15.0, 6.0] },
  ];
  $("#presets").innerHTML = presetDefs.map((p, i) => `<button data-i="${i}">${p.label}</button>`).join("");
  $("#sliders").innerHTML = F.map((f, i) => `<div class="slider-row">
      <div class="top"><label for="n${i}">${f.label}</label><input type="number" id="n${i}" min="0" step="0.1" /></div>
      <input type="range" id="r${i}" min="${range[i].min}" max="${range[i].max}" step="${range[i].step}" aria-label="${f.label}" /></div>`).join("");

  function setValues(v) { cur = v.map(Number); F.forEach((_, i) => { $("#n" + i).value = cur[i]; $("#r" + i).value = cur[i]; }); update(); }
  document.querySelectorAll("#presets button").forEach((b) => b.addEventListener("click", () => setValues(presetDefs[b.dataset.i].v())));
  F.forEach((_, i) => {
    $("#r" + i).addEventListener("input", (e) => { cur[i] = +e.target.value; $("#n" + i).value = cur[i]; update(); });
    $("#n" + i).addEventListener("input", (e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) { cur[i] = v; $("#r" + i).value = v; update(); } });
  });

  const probBars = (probs) => probs.map((p, i) =>
    `<div class="prob"><span>${cap(CLS[i])}</span><div class="bar-track"><div class="bar" style="width:${p * 100}%;background:${COLORS[i]}"></div></div><span class="num">${(p * 100).toFixed(0)}%</span></div>`).join("");

  function update() {
    const issues = M.outOfRange(cur);
    $("#oodBox").innerHTML = issues.length
      ? `<div class="ood"><b>⚠ Unusual measurements – prediction unreliable</b><ul>${issues.map((x) => `<li>${x}</li>`).join("")}</ul>
         <p style="margin:6px 0 0;font-size:.9rem">These values are outside what the models saw in training, so the answer below is an extrapolation. Models always pick one of the 3 species, even for a flower that isn't an iris.</p></div>`
      : `<div class="insight ok" style="margin-top:8px">✓ All measurements are within the range of the training flowers.</div>`;
    const out = { lr: M.lr(cur), knn: M.knn(cur), dt: M.tree(cur) };
    $("#predCards").innerHTML = KEYS.map((k) => `<div class="pred-card" style="border-color:${COLORS[out[k].cls]}">
        <span class="muted">${R[k].name}</span>
        <div class="result" style="color:${COLORS[out[k].cls]}">${cap(CLS[out[k].cls])}</div>${probBars(out[k].probs)}
        ${k === "dt" ? `<p class="muted" style="margin:8px 0 0">Path: ${out.dt.path.map((p) => `${F[p.feature].short.toLowerCase()} ${p.left ? "≤" : ">"} ${p.threshold.toFixed(2)}`).join(" → ")}</p>` : ""}
        ${k === "knn" ? `<p class="muted" style="margin:8px 0 0">Vote of the ${D.knn.k} nearest training flowers.</p>` : ""}
        ${k === "lr" ? `<p class="muted" style="margin:8px 0 0">Softmax probabilities.</p>` : ""}</div>`).join("");
    const nb = out.knn.neighbors;
    $("#knnNeighbors").innerHTML = `<h3>The 5 nearest training flowers (what KNN voted on)</h3><div class="table-scroll"><table><thead><tr><th>Flower #</th><th>Species</th><th>Measurements (cm)</th><th>Distance (scaled)</th></tr></thead><tbody>` +
      nb.map((n) => `<tr><td>${n.id}</td><td style="text-align:left"><span class="chip c${n.cls}">${cap(CLS[n.cls])}</span></td><td class="num">${D.rows[n.id].slice(0, 4).join(", ")}</td><td class="num">${n.dist.toFixed(2)}</td></tr>`).join("") + "</tbody></table></div>";
    scatter($("#predScatter"), 2, 3, { user: cur, neighbors: nb.map((n) => n.id) });
  }
  setValues(cur);

  // ====================================================================
  // CONCLUSION
  // ====================================================================
  (function () {
    const best = KEYS.filter((k) => R[k].accuracy === bestAcc).map((k) => R[k].name);
    const cvBest = KEYS.reduce((a, b) => (R[b].cv_mean > R[a].cv_mean ? b : a));
    $("#conclusionText").innerHTML = `<p>All three simple classifiers separate the iris species very well using just 4 measurements.
      On the 30-flower test set, <b>${best.join(" and ")}</b> scored highest (${pct(bestAcc)} accuracy).
      Under 5-fold cross-validation, <b>${R[cvBest].name}</b> was highest (${f3(R[cvBest].cv_mean)} ± ${f3(R[cvBest].cv_std)}).
      The ranking changes depending on how we evaluate, which shows the models are effectively <b>tied</b> on this dataset.</p>
      <p><b>Key findings:</b> petal measurements are the most informative features; Setosa is perfectly separable; the only real confusion is between Versicolor and Virginica.</p>`;
  })();

  // reveal skill: smooth nav highlight not needed; keep simple
})();
