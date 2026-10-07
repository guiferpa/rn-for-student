import dataset from "./dataset.js";
import tranning from "./tranning.js";
import * as utils from "./utils.js";

async function run() {
  const { model, minv, maxv } = await tranning(
    dataset.train.data,
    dataset.train.labels,
  );

  dataset.test.data.forEach(async function (data) {
    const { tensor } = utils.norm([data], minv, maxv);
    const pred = model.predict(tensor);
    const [rejection, approval] = await pred.data();

    const rejected = rejection > approval;
    const color = rejected ? "\x1b[31m" : "\x1b[32m";
    const reset = "\x1b[0m";
    console.log(
      `${color}Student [${data}] -> Rejection: ${(rejection * 100).toFixed(2)}% | Approval: ${(approval * 100).toFixed(2)}%${reset}`,
    );
  });
}

run();
