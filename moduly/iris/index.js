import * as tf from "/node_modules/@tensorflow/tfjs";
import * as tfvis from './node_modules/@tensorflow/tfjs-vis';

import * as data from './data';
import * as loader from './loader';
import * as ui from './ui';

let model;

/* NASTAVENÍ MODELU + TRENOVANI
 * @param xTrain Training feature data, a `tf.Tensor` of shape
 *   [numTrainExamples, 4]. The second dimension include the features
 *   petal length, petalwidth, sepal length and sepal width.
 * @param yTrain One-hot training labels, a `tf.Tensor` of shape
 *   [numTrainExamples, 3].
 * @param xTest Test feature data, a `tf.Tensor` of shape [numTestExamples, 4].
 * @param yTest One-hot test labels, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 * @returns The trained `tf.Model` instance.*/
async function trainModel(xTrain, yTrain, xTest, yTest) {
  ui.status('Trenuji model.. Prosim strpeni');

  const params = ui.loadTrainParametersFromUI();

  // Definice tpologie modelu: 2 husté vrstvy: 10 neuronů a 3 protože jsou 3 různé výstupy a pravděpodobnost se musí rovnat nule. Sekvenřní model.
  // Sigmoid (1|0 - klasifikace 1.layeru) a softmax (adds to 1)  aktivace.
  // Adam optimizace
  // categoricalCrossentropy funguje líp než RMSE (pro predikce cen - regrese)
  const model = tf.sequential();
  model.add(tf.layers.dense(
      {units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]]}));
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  model.summary();

  const optimizer = tf.train.adam(params.learningRate);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const trainLogs = [];
  const lossContainer = document.getElementById('lossCanvas');
  const accContainer = document.getElementById('accuracyCanvas');
  const beginMs = performance.now();

  // TRAIN: MODEL.FIT
  const history = await model.fit(xTrain, yTrain, {
    epochs: params.epochs,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        const secPerEpoch =
            (performance.now() - beginMs) / (1000 * (epoch + 1));
        ui.status(`Trenuji model... Odhadem ${
            secPerEpoch.toFixed(4)} vterin za epochu`)
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ['loss', 'val_loss'])
        tfvis.show.history(accContainer, trainLogs, ['acc', 'val_acc'])
        calculateAndDrawConfusionMatrix(model, xTest, yTest);
      },
    }
  });

  const secPerEpoch = (performance.now() - beginMs) / (1000 * params.epochs);
  ui.status(
      `Uceni modelu uspesne:  ${secPerEpoch.toFixed(4)} vterin za epochu`);
  return model;
}

/* INFERENCE  on manually-input Iris flower data.
* @param model The instance of `tf.Model` to run the inference with.*/
async function predictOnManualInput(model) {
  if (model == null) {
    ui.setManualInputWinnerMessage('Prosim nahrajte nebo trenujte model.');
    return;
  }

  // Use a `tf.tidy` scope to make sure that WebGL memory allocated for the `predict` call is released at the end.
  tf.tidy(() => {
    // Prepare input data as a 2D `tf.Tensor`.
    const inputData = ui.getManualInputData();
    const input = tf.tensor2d([inputData], [1, 4]);

    // Call `model.predict` to get the prediction output as probabilities for
    // the Iris flower categories.

    const predictOut = model.predict(input);
    const logits = Array.from(predictOut.dataSync());
    const winner = data.IRIS_CLASSES[predictOut.argMax(-1).dataSync()[0]];
    ui.setManualInputWinnerMessage(winner);
    ui.renderLogitsForManualInput(logits);
  });
}

/* Confusion matrix
 * */
async function calculateAndDrawConfusionMatrix(model, xTest, yTest) {
  const [preds, labels] = tf.tidy(() => {
    const preds = model.predict(xTest).argMax(-1);
    const labels = yTest.argMax(-1);
    return [preds, labels];
  });

  const confMatrixData = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = document.getElementById('confusion-matrix');
  tfvis.render.confusionMatrix(
      container,
      {values: confMatrixData, labels: data.IRIS_CLASSES},
      {shadeDiagonal: true},
  );

  tf.dispose([preds, labels]);
}

/* 2. Inference on some test Iris flower data.
 * @param model The instance of `tf.Model` to run the inference with.
 * @param xTest Test data feature, a `tf.Tensor` of shape [numTestExamples, 4].
 * @param yTest Test true labels, one-hot encoded, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 */
async function evaluateModelOnTestData(model, xTest, yTest) {
  ui.clearEvaluateTable();

  tf.tidy(() => {
    const xData = xTest.dataSync();
    const yTrue = yTest.argMax(-1).dataSync();
    const predictOut = model.predict(xTest);
    const yPred = predictOut.argMax(-1);
    ui.renderEvaluateTable(
        xData, yTrue, yPred.dataSync(), predictOut.dataSync());
    calculateAndDrawConfusionMatrix(model, xTest, yTest);
  });

  predictOnManualInput(model);
}

const HOSTED_MODEL_JSON_URL =
    'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json';

/* ..HLAVNÍ FUNKCE: GetData + Model + Trénování s evaluací + ..
* */
async function iris() {
  const [xTrain, yTrain, xTest, yTest] = data.getIrisData(0.15);      //získání TENSORŮ a nastavení testSplitu, 15% testování

  const localLoadButton = document.getElementById('load-local');      //getujeme Load/Save/Remove Buttony
  const localSaveButton = document.getElementById('save-local');
  const localRemoveButton = document.getElementById('remove-local');

  //TVORBA MODELU při kliku na button
  document.getElementById('train-from-scratch')
      .addEventListener('click', async () => {
        model = await trainModel(xTrain, yTrain, xTest, yTest);  //předávám tensory
        await evaluateModelOnTestData(model, xTest, yTest);
        localSaveButton.disabled = false;
      });

  if (await loader.urlExists(HOSTED_MODEL_JSON_URL)) {
    ui.status('Model dostupny: ' + HOSTED_MODEL_JSON_URL);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      ui.clearEvaluateTable();
      model = await loader.loadHostedPretrainedModel(HOSTED_MODEL_JSON_URL);
      await predictOnManualInput(model);
      localSaveButton.disabled = false;
    });
  }

  localLoadButton.addEventListener('click', async () => {
    model = await loader.loadModelLocally();
    await predictOnManualInput(model);
  });

  localSaveButton.addEventListener('click', async () => {
    await loader.saveModelLocally(model);
    await loader.updateLocalModelStatus();
  });

  localRemoveButton.addEventListener('click', async () => {
    await loader.removeModelLocally();
    await loader.updateLocalModelStatus();
  });

  await loader.updateLocalModelStatus();

  ui.status('Vyckavam.');
  ui.wireUpEvaluateTableCallbacks(() => predictOnManualInput(model));
}

iris();
// endregion