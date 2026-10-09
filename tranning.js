/**
 * @file training.js
 * @description Sequential Neural Network model built with TensorFlow.js to predict
 * student approval based on study hours and course attendance. Features a custom
 * Min-Max normalization pipeline that prevents data leakage by applying training bounds
 * to test and inference sets.
 *
 * DATA STRUCTURE EXPLANATION:
 * - train / test: The dataset is split into training (24 samples) to fit the model weights
 *   and testing (6 samples) to rigorously evaluate generalization on unseen data.
 * - data (Inputs): A 2D array where each row represents a student with 2 features:
 *   [0] Weekly Study Hours (numeric range, e.g., 1 to 16 hours)
 *   [1] Course Attendance Rate (percentage range, e.g., 45% to 99%)
 * - labels (Outputs): One-hot encoded categorical vectors matching the 2-unit softmax output:
 *   [1, 0] -> Failed (Class 0)
 *   [0, 1] -> Approved (Class 1)
 */

import * as utils  from "./utils.js";

export default async function tranning(xs, ys, callbacks) {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [2],
      units: 16,
      activation: "relu",
    }),
  );

  model.add(
    tf.layers.dense({
      units: 16,
      activation: "relu",
    }),
  );

  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax",
    }),
  );

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const { tensor, minv, maxv } = utils.norm(xs);

  await model.fit(tensor, tf.tensor2d(ys), {
    verbose: 0,
    epochs: 500,
    shuffle: true,
    callbacks: callbacks || {
      onEpochEnd(epoch, log) {
        console.log(`Epoch: ${epoch}, Loss: ${log.loss}`);
      },
    },
  });

  return { model, minv, maxv };
}

