const WIDTH = 720;
const HEIGHT = 480;
const MARGIN = { top: 16, right: 16, bottom: 48, left: 56 };
const GRID_SIZE = 120;
const DOMAIN = { x: [0, 25], y: [35, 102] };

const COLORS = {
  approved: [42, 120, 214],
  rejected: [227, 73, 72],
  neutral: [240, 239, 236],
  text: "#52514e",
  grid: "#e6e5e1",
  boundary: "#0b0b0b",
  surface: "#fcfcfb",
};

const plot = {
  x0: MARGIN.left,
  y0: MARGIN.top,
  w: WIDTH - MARGIN.left - MARGIN.right,
  h: HEIGHT - MARGIN.top - MARGIN.bottom,
};

const scaleX = (v) =>
  plot.x0 + ((v - DOMAIN.x[0]) / (DOMAIN.x[1] - DOMAIN.x[0])) * plot.w;
const scaleY = (v) =>
  plot.y0 + plot.h - ((v - DOMAIN.y[0]) / (DOMAIN.y[1] - DOMAIN.y[0])) * plot.h;

const gridIndex = (i, j) => j * (GRID_SIZE + 1) + i;

// Approval probability -> diverging color (red <-> gray <-> blue)
function probColor(p) {
  const t = Math.abs(p - 0.5) * 2;
  const pole = p >= 0.5 ? COLORS.approved : COLORS.rejected;
  const rgb = COLORS.neutral.map((c, i) => Math.round(c + (pole[i] - c) * t));
  return `rgba(${rgb.join(",")}, 0.35)`;
}

// [hours, attendance] points covering the whole chart plane
export function gridPoints() {
  const points = [];
  for (let j = 0; j <= GRID_SIZE; j++) {
    for (let i = 0; i <= GRID_SIZE; i++) {
      points.push([
        DOMAIN.x[0] + (i / GRID_SIZE) * (DOMAIN.x[1] - DOMAIN.x[0]),
        DOMAIN.y[0] + (j / GRID_SIZE) * (DOMAIN.y[1] - DOMAIN.y[0]),
      ]);
    }
  }
  return points;
}

// Marching squares: segments where the probability crosses 50%
function boundarySegments(points, probs) {
  const segments = [];
  const lerp = (a, b, va, vb) => {
    const t = (0.5 - va) / (vb - va);
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  };

  for (let j = 0; j < GRID_SIZE; j++) {
    for (let i = 0; i < GRID_SIZE; i++) {
      const corners = [
        gridIndex(i, j),
        gridIndex(i + 1, j),
        gridIndex(i + 1, j + 1),
        gridIndex(i, j + 1),
      ];
      const crossings = [];
      for (let k = 0; k < 4; k++) {
        const a = corners[k];
        const b = corners[(k + 1) % 4];
        if (probs[a] >= 0.5 !== probs[b] >= 0.5) {
          crossings.push(lerp(points[a], points[b], probs[a], probs[b]));
        }
      }
      for (let k = 0; k + 1 < crossings.length; k += 2) {
        segments.push([crossings[k], crossings[k + 1]]);
      }
    }
  }
  return segments;
}

function drawAxes(ctx) {
  ctx.strokeStyle = COLORS.grid;
  ctx.fillStyle = COLORS.text;
  ctx.lineWidth = 1;
  ctx.font = "12px sans-serif";

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let v = 0; v <= DOMAIN.x[1]; v += 2) {
    const x = scaleX(v);
    ctx.beginPath();
    ctx.moveTo(x, plot.y0);
    ctx.lineTo(x, plot.y0 + plot.h);
    ctx.stroke();
    ctx.fillText(v, x, plot.y0 + plot.h + 6);
  }
  ctx.fillText("Horas de estudo por semana", plot.x0 + plot.w / 2, HEIGHT - 18);

  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let v = 40; v <= DOMAIN.y[1]; v += 10) {
    const y = scaleY(v);
    ctx.beginPath();
    ctx.moveTo(plot.x0, y);
    ctx.lineTo(plot.x0 + plot.w, y);
    ctx.stroke();
    ctx.fillText(`${v}%`, plot.x0 - 8, y);
  }
  ctx.save();
  ctx.translate(14, plot.y0 + plot.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Frequência no curso", 0, 0);
  ctx.restore();
}

function drawPoint(ctx, { x, y, approved, correct }) {
  const cx = scaleX(x);
  const cy = scaleY(y);
  const fill = `rgb(${(approved ? COLORS.approved : COLORS.rejected).join(",")})`;

  ctx.lineWidth = 2;
  ctx.strokeStyle = COLORS.surface;
  ctx.fillStyle = fill;
  ctx.beginPath();
  if (approved) ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  else ctx.rect(cx - 4.5, cy - 4.5, 9, 9);
  ctx.fill();
  ctx.stroke();

  if (!correct) {
    ctx.strokeStyle = COLORS.boundary;
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, Math.PI * 2);
    ctx.stroke();
  }
}

export function render(canvas, grid, testPoints) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = WIDTH * dpr;
  canvas.height = HEIGHT * dpr;
  canvas.style.width = `${WIDTH}px`;
  canvas.style.height = `${HEIGHT}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  ctx.fillStyle = COLORS.surface;
  ctx.fillRect(0, 0, WIDTH, HEIGHT);

  // Background: approval probability at each point of the plane
  const cellW = plot.w / GRID_SIZE;
  const cellH = plot.h / GRID_SIZE;
  grid.points.forEach(([x, y], k) => {
    ctx.fillStyle = probColor(grid.probs[k]);
    ctx.fillRect(scaleX(x) - cellW / 2, scaleY(y) - cellH / 2, cellW + 1, cellH + 1);
  });

  drawAxes(ctx);

  // The "deformed line": decision boundary at 50%
  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x0, plot.y0, plot.w, plot.h);
  ctx.clip();
  ctx.strokeStyle = COLORS.boundary;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (const [[ax, ay], [bx, by]] of boundarySegments(grid.points, grid.probs)) {
    ctx.moveTo(scaleX(ax), scaleY(ay));
    ctx.lineTo(scaleX(bx), scaleY(by));
  }
  ctx.stroke();
  ctx.restore();

  testPoints.forEach((p) => drawPoint(ctx, p));
}

export function attachTooltip(canvas, tooltip, testPoints) {
  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;

    let nearest = null;
    let best = 12;
    for (const p of testPoints) {
      const d = Math.hypot(scaleX(p.x) - mx, scaleY(p.y) - my);
      if (d < best) {
        best = d;
        nearest = p;
      }
    }

    if (!nearest) {
      tooltip.style.display = "none";
      return;
    }

    tooltip.innerHTML = `
      <strong>${nearest.x}h de estudo · ${nearest.y}% de frequência</strong><br>
      Real: ${nearest.approved ? "Aprovado" : "Reprovado"}<br>
      Aprovação prevista: ${(nearest.prob * 100).toFixed(1)}%
      ${nearest.correct ? "" : "<br><em>Modelo errou</em>"}
    `;
    tooltip.style.display = "block";
    tooltip.style.left = `${scaleX(nearest.x) + 12}px`;
    tooltip.style.top = `${scaleY(nearest.y) - 12}px`;
  });

  canvas.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
  });
}

export function attachPredictionForm(form, output, predict) {
  form.querySelector("button").disabled = false;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const hours = Number(form.elements.hours.value);
    const attendance = Number(form.elements.attendance.value);
    const prob = await predict(hours, attendance);

    const approved = prob >= 0.5;
    const color = `rgb(${(approved ? COLORS.approved : COLORS.rejected).join(",")})`;
    const marker = approved
      ? `<svg width="10" height="10"><circle cx="5" cy="5" r="5" fill="${color}" /></svg>`
      : `<svg width="10" height="10"><rect width="10" height="10" fill="${color}" /></svg>`;

    output.innerHTML = `
      ${marker}
      <span><strong>${approved ? "Aprovado" : "Reprovado"}</strong>:
      ${(prob * 100).toFixed(1)}% de chance de aprovação
      (${hours}h de estudo · ${attendance}% de frequência)</span>
    `;
  });
}
