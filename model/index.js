import * as tf from "@tensorflow/tfjs-node";
const model = tf.sequential();

const train = async () => {
  console.log("processando...");

  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  const x = tf.tensor([1, 2, 3, 4], [4, 1]);
  const y = tf.tensor([[10], [20], [30], [40]]);

  await model.fit(x, y, { epochs: 1000 });
};

const displayResult = (x) =>
  `\n ###### ${x
    .toString()
    .replace("Tensor", "")
    .replace("[", "")
    .replace("]", "")
    .trim()}  ######`;

export const execute = async (ns) => {
  await train();

  const input = tf.tensor(ns, [3, 1]);

  let output = model.predict(input).dataSync();

  return displayResult(tf.tensor(output).ceil());
};
