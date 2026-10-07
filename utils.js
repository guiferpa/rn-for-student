import tf from "@tensorflow/tfjs-node";

export function norm(xs, min, max) {
  const t2d = tf.tensor2d(xs);
  const minv = min || t2d.min(0);
  const maxv = max || t2d.max(0);

  const nume = t2d.sub(minv);
  const deno = maxv.sub(minv);

  return { tensor: nume.div(deno), minv, maxv };
}
