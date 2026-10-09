/**
 * @file tranning.js
 * @description Sequential Neural Network model built with TensorFlow.js to predict
 * student approval based on study hours and course attendance. Features a custom
 * Min-Max normalization pipeline that prevents data leakage by applying training bounds
 * to test and inference sets.
 *
 * MODEL:
 * - Two hidden dense layers of 16 units with relu activation, so the network can bend
 *   the decision boundary into the L-shaped corner of the approval rule.
 * - Softmax output with 2 units, trained with Adam (learning rate 0.01) for 500 epochs.
 *
 * DATA STRUCTURE EXPLANATION:
 * - Approval rule: attendance >= 80% AND weekly study hours >= 15.
 * - train / test: The dataset is split into training (400 samples, with 5% of rare
 *   approvals below the minimum as noise) to fit the model weights and testing
 *   (100 clean samples, none repeated from training) to evaluate generalization.
 * - data (Inputs): A 2D array where each row represents a student with 2 features:
 *   [0] Weekly Study Hours (numeric range, 0 to 24 hours)
 *   [1] Course Attendance Rate (percentage range, 40% to 100%)
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

