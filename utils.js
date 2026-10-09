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
  // Softmax output: [rejection, approval] per row
  return xs.map((_, i) => values[i * 2 + 1]);
}

// Saves the model along with the normalization bounds, needed to normalize
// new inputs the same way as in training.
export async function exportModel(model, minv, maxv, destination) {
  model.setUserDefinedMetadata({
    normalization: { min: await minv.array(), max: await maxv.array() },
  });
  return model.save(destination);
}
