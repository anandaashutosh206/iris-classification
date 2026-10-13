// model.js - runs the TRAINED scikit-learn models in the browser.
// Parameters (scaler, coefficients, training data, tree) come from data.js.
(function (g) {
  const D = g.IRIS;

  const scale = (r) => r.map((v, i) => (v - D.scaler.mean[i]) / D.scaler.scale[i]);
  const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
  const softmax = (a) => {
    const m = Math.max(...a), e = a.map((x) => Math.exp(x - m)), s = e.reduce((p, c) => p + c, 0);
    return e.map((x) => x / s);
  };

  // Logistic Regression: softmax(W . x_scaled + b)
  function lr(r) {
    const z = scale(r);
    const logits = D.lr.coef.map((w, k) => w.reduce((s, wi, i) => s + wi * z[i], D.lr.intercept[k]));
    const probs = softmax(logits);
    return { cls: argmax(probs), probs };
  }

  // K-Nearest Neighbors: majority vote of the k closest training flowers
  function knn(r, k = D.knn.k) {
    const z = scale(r);
    const d = D.knn.X.map((x, j) => ({
      id: D.knn.ids[j], cls: D.knn.y[j],
      dist: Math.hypot(...x.map((v, i) => v - z[i])),
    }));
    d.sort((a, b) => a.dist - b.dist);
    const neighbors = d.slice(0, k);
    const counts = [0, 0, 0];
    neighbors.forEach((n) => counts[n.cls]++);
    return { cls: argmax(counts), probs: counts.map((c) => c / k), neighbors };
  }

  // Decision Tree: follow the yes/no questions down to a leaf
  function tree(r) {
    let n = D.tree;
    const path = [];
    while (!n.leaf) {
      const left = r[n.feature] <= n.threshold;
      path.push({ feature: n.feature, threshold: n.threshold, left });
      n = left ? n.left : n.right;
    }
    const tot = n.counts.reduce((a, b) => a + b, 0);
    return { cls: n.cls, probs: n.counts.map((c) => c / tot), path, leafSamples: tot };
  }

  // Simple out-of-range check (is the flower unlike the training data?)
  function outOfRange(r) {
    const issues = [];
    r.forEach((v, i) => {
      const f = D.features[i].short.toLowerCase();
      if (!(v > 0)) issues.push(`${f} must be greater than 0`);
      else if (v < D.train_range.min[i]) issues.push(`${f} (${v}) is below the training minimum (${D.train_range.min[i]})`);
      else if (v > D.train_range.max[i]) issues.push(`${f} (${v}) is above the training maximum (${D.train_range.max[i]})`);
    });
    return issues;
  }

  g.IrisModels = { lr, knn, tree, outOfRange, scale };
})(typeof window !== "undefined" ? window : globalThis);
