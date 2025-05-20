const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

let model = null;

async function loadModel() {
  const modelDir = path.join(__dirname, 'ai', 'models', 'price_prediction_model');
  const modelJson = path.join(modelDir, 'model.json');
  if (fs.existsSync(modelJson)) {
    try {
      model = await tf.loadLayersModel(`file://${modelJson}`);
      console.log('Loaded TensorFlow model from', modelJson);
      return;
    } catch (err) {
      console.error('Error loading model:', err);
    }
  } else {
    console.warn(`Model file not found at ${modelJson}, using fallback model`);
  }
  model = createFallbackModel();
  console.log('Fallback model initialized');
}

function createFallbackModel(numFeatures = 1) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 32, inputShape: [numFeatures], activation: 'relu' }));
  m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  m.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
  return m;
}

async function predict(features) {
  if (!model) await loadModel();
  const numFeatures = Array.isArray(features) ? features.length : 1;
  if (!model.inputs || model.inputs[0].shape[1] !== numFeatures) {
    console.warn('Rebuilding fallback model for', numFeatures, 'features');
    model = createFallbackModel(numFeatures);
  }
  const input = tf.tensor2d([features]);
  const outputTensor = model.predict(input);
  const output = Array.isArray(outputTensor) ? outputTensor[0] : outputTensor;
  const data = await output.data();
  tf.dispose([input, output]);
  return data[0];
}

async function testPredictor() {
  const dummyInput = [0];
  try {
    const result = await predict(dummyInput);
    console.log('Test prediction:', result);
  } catch (err) {
    console.error('Error during test prediction:', err);
  }
}

module.exports = { loadModel, predict, createFallbackModel, testPredictor }; 