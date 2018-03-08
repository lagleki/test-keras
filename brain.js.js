const brain = require('brain.js');

const fs = require('fs-extra');
const sin = fs.readFileSync('sinwave.csv', {
  encoding: 'utf8'
}).split(/[\n\r]+/).map(i => parseFloat(i));
const lg = console.log.bind(console);
const seq_predict = 1;
const last = 5;
let trainingSet = [];
for (let i = 0; i < sin.length - last - seq_predict; i++) {
  trainingSet.push({
    input: sin.slice(i, i + last),
    output: sin.slice(i + last, i + last + seq_predict)
  });
}

fs.writeFileSync('da-out.csv', JSON.stringify(trainingSet), {
  encoding: 'utf8'
});

var network = new brain.recurrent.LSTM({hiddenLayers: [50,100]});

function Train() {
  console.log(trainingSet.slice(-1), trainingSet.length);
  network.train(trainingSet, {
    iterations: 20, // the maximum times to iterate the training data --> number greater than 0
    errorThresh: 0.05, // the acceptable error percentage from training data --> number between 0 and 1
    log: true, // true to use console.log, when a function is supplied it is used --> Either true or a function
    logPeriod: 10, // iterations between logging out --> number greater than 0
    learningRate: 0.005, // scales with delta to effect traiing rate --> number between 0 and 1
    momentum: 0.1, // scales with next layer's change value --> number between 0 and 1
    callback: null, // a periodic call back that can be triggered while training --> null or function
    callbackPeriod: 10, // the number of iterations through the training data between callback calls --> number greater than 0
    timeout: Infinity // the max number of milliseconds to train for --> number greater than 0
  });
  for (let i = 0; i < 10; i++) {
    console.log(`from `, JSON.stringify(trainingSet[i].input), `predicted `, network.run(
      trainingSet[i]
    ), `but really is`, trainingSet[i].output);
  }
}
Train();
