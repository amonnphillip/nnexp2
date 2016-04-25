
var nodeSigmoid = function() {
  return {
    inputs: [],
    inputWeights: [],
    output: 0,
    error: 0,
    outputLinks: [],
    getMatchingWeightForNode: function(node) {
      for(var index = 0;index < this.inputs.length;index ++) {
        if (this.inputs[index] == node) {
          return this.inputWeights[index];
        }
      }
    },
    linkToInput: function(node) {
      this.inputs.push(node);
      this.inputWeights.push(1);
      node.linkOutput(this);
    },
    linkOutput: function(node) {
      this.outputLinks.push(node);
    },
    forward: function() {
      var val = 0;
      for (var index = 0;index < this.inputs.length;index ++) {
        val += this.inputs[index].output * this.inputWeights[index];
      }

      this.output = 1.0 / (1.0 + Math.exp(-val));
    },
    backward: function(learnRate) {
      var error = 0;
      for (var outputLinkIndex = 0;outputLinkIndex < this.outputLinks.length;outputLinkIndex ++) {
        var n = this.outputLinks[outputLinkIndex].getMatchingWeightForNode(this);
        error += this.outputLinks[outputLinkIndex].error * n;
      }

      this.error = this.output * ((1 - this.output) * error);

      for (var weightIndex = 0;weightIndex < this.inputs.length;weightIndex ++) {
        var tweakAmount = this.error * this.inputs[weightIndex].output;
        tweakAmount *= learnRate;
        this.inputWeights[weightIndex] += tweakAmount;
      }
    },
    backwardWithExpectedOutput: function(learnRate, expectedOutput) {
      this.error = this.output * ((1 - this.output) * (expectedOutput - this.output));
      for (var weightIndex = 0;weightIndex < this.inputs.length;weightIndex ++) {
        var tweakAmount = this.error * this.inputs[weightIndex].output;
        tweakAmount *= learnRate;
        this.inputWeights[weightIndex] += tweakAmount;
      }
    },
    backwardWithError: function(learnRate, error) {
      this.error = error * learnRate;
      for (var weightIndex = 0;weightIndex < this.inputs.length;weightIndex ++) {
        var tweakAmount = this.error * this.inputs[weightIndex].output;
        tweakAmount *= learnRate;
        this.inputWeights[weightIndex] += tweakAmount;
      }
    },
    getWeightCount: function() {
      return this.inputWeights.length;
    },
    getWeight: function(weightIndex) {
      return this.inputWeights[weightIndex];
    },
    getOutput: function() {
      return this.output;
    }
  }
};

var inputNode = function() {
  return {
    output: 0,
    outputLinks: [],
    setOutput: function(value) {
      this.output = value;
    },
    linkOutput: function(node) {
      this.outputLinks.push(node);
    }
  }
};

var layer = function() {
  return {
    nodes: [],
    initialize: function(layerSize, isInput) {
      this.nodes = [];
      for (var index = 0;index < layerSize;index ++) {
        if (isInput) {
          this.nodes.push(new inputNode());
        } else {
          this.nodes.push(new nodeSigmoid());
        }
      }
    },
    linkOutputToInputs: function(layer) {
      // We assume the same number of nodes in the layer
      for (var index = 0;index < layer.nodes.length;index ++) {
        for (var nodeLinkIndex = 0;nodeLinkIndex < this.nodes.length;nodeLinkIndex ++) {
          layer.nodes[index].linkToInput(this.nodes[(nodeLinkIndex) % this.nodes.length]);
        }
      }
    },
    forward: function() {
      for (var index = 0;index < this.nodes.length;index ++) {
        this.nodes[index].forward();
      }
    },
    backward: function(nodeIndex, learnRate) {
      this.nodes[nodeIndex].backward(learnRate);
    },
    backwardOutputLayer: function(nodeIndex, learnRate, expectedOutput) {
      this.nodes[nodeIndex].backwardWithExpectedOutput(learnRate, expectedOutput);
    },
    backwardOutputLayerWithError: function(nodeIndex, learnRate, error) {
      this.nodes[nodeIndex].backwardWithError(learnRate, error);
    },
    setNodeOutput: function(nodeIndex, value) {
      this.nodes[nodeIndex].setOutput(value);
    },
    getNodeOutput: function(nodeIndex) {
      return this.nodes[nodeIndex].getOutput();
    },
    getNodeCount: function() {
      return this.nodes.length;
    },
    displayToConsole: function() {
      var out = 'inputs: ';

      for (var index = 0;index < this.nodes.length;index ++) {
        for (var inputIndex = 0;inputIndex < this.nodes[index].inputs.length;inputIndex ++) {
          out += this.nodes[index].inputs[inputIndex].output.toString() + ' ';
        }
        out += ',';
      }
      console.log(out);

      out = 'weights:';
      for (var index = 0;index < this.nodes.length;index ++) {
        for (var weightIndex = 0;weightIndex < this.nodes[index].getWeightCount();weightIndex ++) {
          out += this.nodes[index].getWeight(weightIndex).toString() + ' ';
        }
        out += ','
      }
      console.log(out);

      out = 'error:  ';
      for (var index = 0;index < this.nodes.length;index ++) {
        out += this.nodes[index].error.toString() + ',';
      }
      console.log(out);

      out = 'output: ';
      for (var index = 0;index < this.nodes.length;index ++) {
        out += this.nodes[index].output.toString() + ',';
      }
      console.log(out);
    }
  }
};

var temporalWindow = function() {
  return {
    frames: [],
    set: function(frames) {
      this.frames = JSON.parse(JSON.stringify(frames));
    },
    copy: function() {
      return JSON.parse(JSON.stringify(this.frames));
    },
    pushFrame: function(state, maxFrames) {
      this.frames.unshift(state);

      if (this.frames.length > maxFrames) {
        this.frames.pop();
      }
    },
    getFrames: function() {
      return this.frames;
    }
  }
};

var network = function() {
  return {
    layers: [],
    windowOfStates: [],
    maxWindowStates: 2000,
    learnWindow: 50,
    learnRate: 0.01,
    actionsAsOutputs: [
      [1, 0, 0, 0], // up
      [0, 1, 0, 0], // left
      [0, 0, 1, 0], // down
      [0, 0, 0, 1] // right
    ],
    epsilon: 2000,
    buckets: [0,0,0,0,0],
    actionChoiceBuckets: [0,0,0,0,0],
    temporalWindowFrames: 4,
    temporalWindow: new temporalWindow(),
    normalize: function(value, max) {
      return value / max;
    },
    normalizeAndClamp: function(value, max) {
      if (value > max) {
        value = max;
      }
      return this.normalize(value, max);
    },
    distance: function(x, y, x1, y1) {
      return Math.sqrt(Math.pow(x - x1, 2) + Math.pow(y - y1, 2));
    },
    moveActor: function(action, gameState) {
      var posx = gameState.actorPosx;
      var posy = gameState.actorPosy;

      if (action[0] > 0) {
        posy --;
      } else if (action[1] > 0) {
        posx --;
      } else if (action[2] > 0) {
        posy ++;
      } else if (action[3] > 0) {
        posx ++;
      }

      if (posy < 0 ||
        posy >= gameState.areaMax ||
        posx < 0 ||
        posx >= gameState.areaMax) {
        posx = gameState.actorPosx;
        posy = gameState.actorPosy;
      }

      if (gameState.actorPosx === gameState.goalPosx &&
        gameState.actorPosy === gameState.goalPosy) {
        gameState.score ++;
        posx = 0;
        posy = 0;
        gameState.steps = 0;
      }

      gameState.actorPosx = posx;
      gameState.actorPosy = posy;

      gameState.currentDistance = this.distance(gameState.actorPosx , gameState.actorPosy, gameState.goalPosx, gameState.goalPosy);

      gameState.steps ++;
    },
    evaluationFunction: function(action, gameState) {
      var error = 0;

      var posx = gameState.actorPosx;
      var posy = gameState.actorPosy;

      if (action[0] > 0) {
        posy --;
      } else if (action[1] > 0) {
        posx --;
      } else if (action[2] > 0) {
        posy ++;
      } else if (action[3] > 0) {
        posx ++;
      }

      var hitWall = false;
      if (posy < 0 ||
        posy >= gameState.areaMax ||
        posx < 0 ||
        posx >= gameState.areaMax) {
        posx = gameState.actorPosx;
        posy = gameState.actorPosy;
        hitWall = true;
      }

      var currentDistance = this.distance(posx , posy, gameState.goalPosx, gameState.goalPosy);

      if (hitWall) {
        error = 1;
      } else if (posx === gameState.goalPosx &&
        posy === gameState.goalPosy) {
        error = 0;
      } else if (currentDistance < gameState.currentDistance) {
        error = 0.25;
      } else if (currentDistance > gameState.currentDistance) {
        error = 0.75;
      } else{
        error = 0.5;
      }

      var stepsError = this.normalizeAndClamp(gameState.steps, gameState.maxSteps);

      error = (error + stepsError) / 2;

      if (error > 1) {
        error = 1;
      }

      return error;
    },
    initialize: function(layers) {
      // Create the layers
      this.layers = [];
      for (var index = 0;index < layers.length;index ++) {
        var l = new layer();
        l.initialize(layers[index], index === 0);
        this.layers.push(l);
      }

      // link the layers
      for (var index = 1;index < layers.length;index ++) {
        this.layers[index - 1].linkOutputToInputs(this.layers[index]);
      }
    },
    saveStateInTemporalWindow: function(state, action) {
      var frame = this.stateToInputNodes(state, action);
      this.temporalWindow.pushFrame(frame, this.temporalWindowFrames);
    },
    stateToInputNodes: function(state, action) {
      if (typeof action === 'undefined') {
        var action = [0,0,0,0];
      }

      return [
        this.normalize(state.actorPosx, state.areaMax),
        this.normalize(state.actorPosy, state.areaMax),
        this.normalize(state.goalPosx, state.areaMax),
        this.normalize(state.goalPosy, state.areaMax),
        this.normalize(state.currentDistance, this.distance(0, 0, state.areaMax, state.areaMax)),
        this.normalizeAndClamp(state.steps, state.maxSteps),
        action[0],
        action[1],
        action[2],
        action[3]
      ];
    },
    temporalWindowToInputNodes: function(temporalWindow) {
      var inputs = [];
      var frames = temporalWindow.getFrames();
      frames.forEach((frame) => {
        inputs = inputs.concat(frame);
      });

      return inputs;
    },
    temporalWindowToInputNodesWithCurrentState: function(temporalWindow, state) {
      var inputs = [];
      var frames = temporalWindow.getFrames();

      frames.unshift(this.stateToInputNodes(state));
      if (frames.length > this.temporalWindowFrames) {
        frames.pop();
      }

      frames.forEach((frame) => {
        inputs = inputs.concat(frame);
      });

      return inputs;
    },
    simulation: function(maxIterations) {
      /*
        A very simple move to the center game
       */
      var simulationIteration = 0;
      var layerDisplayCount = 1000;

      // Out variables for the game
      var gameState = {
        areaMax: 20,
        actorPosx: 0,
        actorPosy: 0,
        goalPosx: 19,
        goalPosy: 19,
        currentDistance: 0,
        score: 0,
        maxSteps: 39,
        steps: 0
      };
      gameState.currentDistance = this.distance(gameState.actorPosx , gameState.actorPosy, gameState.goalPosx, gameState.goalPosy);

      this.saveStateInTemporalWindow(gameState);
      this.saveStateInTemporalWindow(gameState);
      this.saveStateInTemporalWindow(gameState);

      while (simulationIteration < maxIterations) {

        // Run sim

        // Generate inputs
        //gameState.currentDistance = this.distance(gameState.actorPosx , gameState.actorPosy, gameState.goalPosx, gameState.goalPosy);
        //var maxDistance = this.distance(0 , 0, gameState.areaMax, gameState.areaMax);
/*
        var inputs = [
          this.normalize(gameState.actorPosx, gameState.areaMax),
          this.normalize(gameState.actorPosy, gameState.areaMax),
          this.normalize(gameState.goalPosx, gameState.areaMax),
          this.normalize(gameState.goalPosy, gameState.areaMax),
          this.normalize(gameState.currentDistance, maxDistance)
        ];*/

        // Train the nn
        this.tick(simulationIteration, gameState);

        // Display
        layerDisplayCount --;
        if (layerDisplayCount <= 0) {
          layerDisplayCount = 10;
          //this.displayToConsole();
        }

        process.stdout.write("\u001b[2J\u001b[0;0H");
        console.log('simulationIteration: ' + simulationIteration);
        console.log('steps: ' + gameState.steps);
        console.log('score: ' + gameState.score);
        console.log('actor pos: ' + gameState.actorPosx + ',' + gameState.actorPosy);
        for (var y = 0;y < gameState.areaMax;y ++) {
          var out = '';
          for (var x = 0;x < gameState.areaMax;x ++) {
            if (x === gameState.actorPosx &&
              y === gameState.actorPosy) {
              out += 'X';
            } else if (x === gameState.goalPosx &&
              y === gameState.goalPosy) {
              out += 'O';
            } else {
              out += ' ';
            }
          }
          console.log(out);
        }

        simulationIteration ++;
      }
    },
    tick: function(iteration, state) {

      //this.saveStateInTemporalWindow(state, );

      var inputs = this.temporalWindowToInputNodesWithCurrentState(this.temporalWindow, state);

      //var inputs = this.temporalWindowToInputNodes(this.temporalWindow);

      var inputLayer = this.layers[0];
      for (var nodeIndex = 0;nodeIndex < inputLayer.getNodeCount();nodeIndex ++) {
        inputLayer.setNodeOutput(nodeIndex, inputs[nodeIndex]);
      }

      this.forward();

      var output = [];
      var outputLayer = this.layers[this.layers.length - 1];
      for (var nodeIndex = 0;nodeIndex < outputLayer.getNodeCount();nodeIndex ++) {
        output.push(outputLayer.getNodeOutput(nodeIndex));
      }

      // Get an action from the output and update simulation
      var action;
      if (iteration < this.epsilon) {
        var actionChoice = Math.floor(Math.random() * this.actionsAsOutputs.length);

        //this.actionChoiceBuckets[actionChoice] ++;
        //console.log('actionChoiceBuckets: ' + this.actionChoiceBuckets[0] + ' ' + this.actionChoiceBuckets[1] + ' ' + this.actionChoiceBuckets[2] + ' ' + this.actionChoiceBuckets[3] + ' ' + this.actionChoiceBuckets[4]);

        action = this.actionsAsOutputs[actionChoice];
      } else {
        var strongestSignal = 0;
        for (var actionIndex = 0;actionIndex < this.actionsAsOutputs.length;actionIndex ++) {
          if (output[actionIndex] > strongestSignal) {
            strongestSignal = output[actionIndex];
            action = this.actionsAsOutputs[actionIndex];
          }
        }
      }


      var error = this.evaluationFunction(action, state);
      this.moveActor(action, state);

      this.saveStateInTemporalWindow(state, action);


      var temporalWindowCopy = new temporalWindow();
      temporalWindowCopy.set(this.temporalWindow.copy());

      this.windowOfStates.push({
        temporalWindow: temporalWindowCopy,
        gameState: JSON.parse(JSON.stringify(state)),
        //action: action,
        error: error
      });
      this.train(this.windowOfStates[this.windowOfStates.length - 1]);
/*
      for (var layerIndex = this.layers.length - 1;layerIndex > 0;layerIndex --) {
        var layer = this.layers[layerIndex];
        for (var nodeIndex = 0;nodeIndex < layer.nodes.length;nodeIndex ++) {
          if (layerIndex === this.layers.length - 1) {
            layer.backwardOutputLayerWithError(nodeIndex, this.learnRate, outputAdjustedForError[nodeIndex]);
          } else {
            layer.backward(nodeIndex, this.learnRate);
          }
        }
      }*/

      // Train based on previous input states from our 'memory' window
      if (this.windowOfStates.length > this.learnWindow) {
        for (var trainIndex = 0;trainIndex < this.learnWindow;trainIndex ++) {
          var index = Math.floor(Math.random() * this.windowOfStates.length);
          this.train(this.windowOfStates[index]);
        }
      }

      // Manage state window
      while (this.windowOfStates.length >= this.maxWindowStates) {
        var index = Math.floor(Math.random() * this.windowOfStates.length);
        this.windowOfStates.splice(index, 1);
      }

      return output;
    },
    train: function(state) { // TODO: Should rename state to window or something
      var inputLayer = this.layers[0];

      var inputs = this.temporalWindowToInputNodes(state.temporalWindow);

      inputs[6] = 0; // TODO: A bit of a hack! May need to restructure things
      inputs[7] = 0;
      inputs[8] = 0;
      inputs[9] = 0;

      for (var nodeIndex = 0;nodeIndex < inputLayer.getNodeCount();nodeIndex ++) {
        inputLayer.setNodeOutput(nodeIndex, inputs[nodeIndex]);
      }

      this.forward();

      var output = [];
      var outputLayer = this.layers[this.layers.length - 1];
      for (var nodeIndex = 0;nodeIndex < outputLayer.getNodeCount();nodeIndex ++) {
        output.push(outputLayer.getNodeOutput(nodeIndex));
      }

      var lowestError = 1;
      var bestAction = this.actionsAsOutputs[0];
      for (var actionIndex = 0;actionIndex < this.actionsAsOutputs.length;actionIndex ++) {
        var thisActionError = this.evaluationFunction(this.actionsAsOutputs[actionIndex], state.gameState);
        if (thisActionError < lowestError) {
          bestAction = this.actionsAsOutputs[actionIndex];
          lowestError = thisActionError;
        }
      }

      //console.log('best action: ' + bestAction[0] + ' ' + bestAction[1] + ' ' + bestAction[2] + ' ' + bestAction[3] + ' ' + bestAction[4]);
      /*
      if (bestAction[0] > 0) {
        this.buckets[0] ++;
      } else if (bestAction[1] > 0) {
        this.buckets[1] ++;
      } else if (bestAction[2] > 0) {
        this.buckets[2] ++;
      } else if (bestAction[3] > 0) {
        this.buckets[3] ++;
      }*/
      //console.log('buckets: ' + this.buckets[0] + ' ' + this.buckets[1] + ' ' + this.buckets[2] + ' ' + this.buckets[3] + ' ' + this.buckets[4]);

/*
      if (bestAction[0] > 0) {
        console.log('best action: up');
      } else if (bestAction[1] > 0) {
        console.log('best action: left');
      } else if (bestAction[2] > 0) {
        console.log('best action: down');
      } else if (bestAction[3] > 0) {
        console.log('best action: right');
      }*/

      for (var layerIndex = this.layers.length - 1;layerIndex > 0;layerIndex --) {
        var layer = this.layers[layerIndex];
        for (var nodeIndex = 0;nodeIndex < layer.nodes.length;nodeIndex ++) {
          if (layerIndex === this.layers.length - 1) {
            if (bestAction[nodeIndex] > 0) {
              layer.backwardOutputLayer(nodeIndex, this.learnRate * 2, bestAction[nodeIndex]);
            } else {
              layer.backwardOutputLayer(nodeIndex, this.learnRate, bestAction[nodeIndex]);
            }
            //layer.backwardOutputLayer(nodeIndex, this.learnRate, bestAction[nodeIndex]);
            //layer.backwardOutputLayerWithError(nodeIndex, this.learnRate, state.error - lowestError);
          } else {
            layer.backward(nodeIndex, this.learnRate);
          }
        }
      }
    },
    forward: function() {
      for (var index = 1;index < this.layers.length;index ++) {
        this.layers[index].forward();
      }
    },
    displayToConsole: function() {
      for (var index = 1;index < this.layers.length;index ++) {
        this.layers[index].displayToConsole();
      }
      console.log('');
    }
  }
};

// Initialize our network
var theNetwork = new network();
theNetwork.initialize([
  10 * 4, 10, 10, 10, 10, 4
]);

// Train it
theNetwork.simulation(2000000);

