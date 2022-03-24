import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js';

const CLOTHING_TYPES = ["T-shirt/top", "Trouser", "Pullover", 
                     "Dress", "Coat", "Sandal", "Shirt", 
                     "Sneaker", "Bag", "Ankle boot"];

// Grab a reference to the MNIST input values (pixel data).

const INPUTS = TRAINING_DATA.inputs;

// Grab reference to the MNIST output values.

const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays in the same way so inputs still match outputs indexes.

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Now actually create and define model architecture.

const model = tf.sequential();

model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));

model.add(tf.layers.dense({units: 16, activation: 'relu'}));

model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

model.summary();

train();


function normalize(tensor, min, max) {

  const result = tf.tidy(function() {
    
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return NORMALIZED_VALUES;

  });

  return result;

}


async function train() { 

  // Compile the model with the defined optimizer and specify our loss function to use.

  model.compile({

    optimizer: 'adam',

    loss: 'categoricalCrossentropy',

    metrics: ['accuracy']

  });

  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {

    shuffle: true,        // Ensure data is shuffled again before using each epoch.

    validationSplit: 0.2,

    batchSize: 512,       // Update weights after every 512 examples.      

    epochs: 50,           // Go over the data 50 times!

    callbacks: {onEpochEnd: logProgress}

  });

  

  OUTPUTS_TENSOR.dispose();

  INPUTS_TENSOR.dispose();

  evaluate(); // Once trained we can evaluate the model.

}


function logProgress(epoch, logs) {
  
  console.log('Data for epoch ' + epoch, Math.sqrt(logs.loss));
  
}


const PREDICTION_ELEMENT = document.getElementById('prediction');

function evaluate() {

  const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select random from all example inputs. 

 

  let answer = tf.tidy(function() {

    let newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();

    

    let output = model.predict(newInput);

    output.print();

    return output.squeeze().argMax();    

  });
  
  answer.array().then(function(index) {

    PREDICTION_ELEMENT.innerText = CLOTHING_TYPES[index];

    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');

    answer.dispose();

    drawImage(INPUTS[OFFSET]);

  });

}


const CANVAS = document.getElementById('canvas');

const CTX = CANVAS.getContext('2d');


function drawImage(digit) {

  var imageData = CTX.getImageData(0, 0, 28, 28);

  

  for (let i = 0; i < digit.length; i++) {

    imageData.data[i * 4] = digit[i] * 255;      // Red Channel.

    imageData.data[i * 4 + 1] = digit[i] * 255;  // Green Channel.

    imageData.data[i * 4 + 2] = digit[i] * 255;  // Blue Channel.

    imageData.data[i * 4 + 3] = 255;             // Alpha Channel.

  }

  // Render the updated array of data to the canvas itself.

  CTX.putImageData(imageData, 0, 0); 

  // Perform a new classification after a certain interval.

  setTimeout(evaluate, 2000);

}
