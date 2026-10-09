export function norm(xs, min, max) {
  const t2d = tf.tensor2d(xs);
  const minv = min || t2d.min(0);
  const maxv = max || t2d.max(0);

  const nume = t2d.sub(minv);
  const deno = maxv.sub(minv);

  return { tensor: nume.div(deno), minv, maxv };
}

export async function predictApproval(model, xs, minv, maxv) {
  const { tensor } = norm(xs, minv, maxv);
  const pred = model.predict(tensor);
  const values = await pred.data();
  tf.dispose([tensor, pred]);
  // Saída softmax: [rejeição, aprovação] por linha
  return xs.map((_, i) => values[i * 2 + 1]);
}

// Salva o modelo junto com os limites da normalização, necessários para
// normalizar novas entradas do mesmo jeito que no treino.
export async function exportModel(model, minv, maxv, destination) {
  model.setUserDefinedMetadata({
    normalization: { min: await minv.array(), max: await maxv.array() },
  });
  return model.save(destination);
}
