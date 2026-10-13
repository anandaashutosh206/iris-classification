// Optional check (needs Node.js): confirms the JavaScript models give the
// same predictions as scikit-learn on every test flower.
//   node frontend/verify.js
global.window = global;
require("./data.js");
require("./model.js");
const D = window.IRIS, M = window.IrisModels;
let bad = 0, total = 0;
for (const key of ["lr", "knn", "dt"]) {
  const fn = key === "dt" ? M.tree : M[key];
  D.split.test_idx.forEach((id, j) => {
    total++;
    if (fn(D.rows[id].slice(0, 4)).cls !== D.results[key].test_pred[j]) bad++;
  });
}
console.log(`Checked ${total} predictions against scikit-learn: ${bad} mismatches`);
