import dataset from "./dataset.js";
import tranning from "./tranning.js";
import * as utils from "./utils.js";
import * as render from "./render.js";

async function run() {
  const fitCallbacks = tfvis.show.fitCallbacks(
    { name: "Treino", tab: "Modelo" },
    ["loss", "acc"],
    { callbacks: ["onEpochEnd"] },
  );

  const { model, minv, maxv } = await tranning(
    dataset.train.data,
    dataset.train.labels,
    fitCallbacks,
  );

  const probs = await utils.predictApproval(model, dataset.test.data, minv, maxv);
  const testPoints = dataset.test.data.map(([x, y], i) => {
    const approved = dataset.test.labels[i][1] === 1;
    const prob = probs[i];
    return { x, y, approved, prob, correct: prob >= 0.5 === approved };
  });

  const points = render.gridPoints();
  const grid = { points, probs: await utils.predictApproval(model, points, minv, maxv) };

  const canvas = document.getElementById("boundary");
  render.render(canvas, grid, testPoints);
  render.attachTooltip(canvas, document.getElementById("tooltip"), testPoints);

  const exportButton = document.getElementById("export");
  exportButton.disabled = false;
  exportButton.addEventListener("click", () =>
    utils.exportModel(model, minv, maxv, "downloads://modelo-aprovacao"),
  );

  const hits = testPoints.filter((p) => p.correct).length;
  document.getElementById("status").textContent =
    `Acurácia no teste: ${hits}/${testPoints.length} (${((hits / testPoints.length) * 100).toFixed(1)}%). Passe o mouse nos pontos para detalhes.`;
}

run();
