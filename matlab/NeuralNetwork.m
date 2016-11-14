%% Neural network
classdef NeuralNetwork
    %% Properties
    properties (GetAccess = 'public', SetAccess = 'private')
        p_topology;
        p_layers;
        p_outputValue;
        p_error;
        p_learningRate;
        p_momentum;
        p_bestSolution;
        p_bestIteration;
        p_bestError;
    end
    
    %% Public methods
    methods
        %% Constructor
        % prmTopology: 1xn topology matrix with n being the number of layers
        %  and the values of the matrix represent the number of neurons in
        %  each layer (without the bias). ex: [2 3 3 1]
        %  Input layer has as many neurons as inputs there are (not validated).
        % prmLearningRate: Learning rate from 0 to 1.
        % prmMomentum: Delta weight multiplier from 0 to 1.
        function result = NeuralNetwork(prmTopology, prmLearningRate, prmMomentum)
            result.p_topology = prmTopology;
            result.p_outputValue = 0;
            result.p_error = 0;
            result.p_learningRate = prmLearningRate;
            result.p_momentum = prmMomentum;
            result.p_bestSolution = realmax;
            result.p_bestIteration = 0;
            result.p_bestError = realmax;
            
            numLayers = length(prmTopology);
            result.p_layers = Layer.empty(0,numLayers);
            
            % Add each layer
            for i = 1:numLayers
                numNeurons = prmTopology(i);
                bitLastLayer = 0;
                if(i == numLayers)
                    bitLastLayer = 1;
                end
                
                if (bitLastLayer)
                    layer = Layer(numNeurons);
                else
                    numOutputsNextLayer = prmTopology(i + 1); % Calculates the outputs for each neuron in next layer. Last layer has no output
                    layer = Layer(numNeurons + 1); % Creates a layer with empty matrix of neurons from topology. Last layer has no bias
                end
                
                % Add neurons to layer
                for j = 1:numNeurons
                    % Add output weights to layer. Last layer has no output weights
                    weights = 0;
                    if (~bitLastLayer)
                        weights = 2*rand(numOutputsNextLayer, 1) - 1; % Random weights (1 row for each neuron in next layer). Values between -1 and 1
                    end
                    
                    neuron = Neuron(weights);
                    layer = layer.UpdateNeuron(neuron, j);
                end
                
                % Add bias neuron to layer. Last layer has no bias
                if (~bitLastLayer)
                    weights = 2*rand(numOutputsNextLayer, 1) - 1; % Values between -1 and 1
                    neuron = Neuron(weights);
                    layer = layer.UpdateNeuron(neuron, j + 1);
                end
                
                result.p_layers(i) = layer;
            end
        end
        
        %% TrainNeuralNetwork
        % Trains neural network for the number of iterations provided
        function result = TrainNeuralNetwork(result, prmInputValues, prmExpectedValue, prmMaxIterations)
            for i = 1:prmMaxIterations
                % Feedforward
                result = result.Feedforward(prmInputValues);
                outputValue = result.p_outputValue;
                % Backpropagation
                result = result.Backpropagation(prmExpectedValue);
                errorValue = result.p_error;
                
                % Calculate best solution based on error
                if(errorValue < result.p_bestError)
                    result.p_bestSolution = outputValue;
                    result.p_bestIteration = i;
                    result.p_bestError = errorValue;
                end
                
                fprintf('Iteration: %d | output: %d | error %d \n', i, outputValue, errorValue);
            end
            
            disp(result.p_topology);
            fprintf('BEST: iteration: %d | output: %d | error %d \n', result.p_bestIteration, result.p_bestSolution, result.p_bestError);
        end
        
        %% Feedforward
        function result = Feedforward(result, prmInputValues)
            % Add output values to bias neurons in all layers. Last layer has no bias
            for i = 1:length(result.p_layers) - 1
                layer = result.p_layers(i);
                lastNeuronPos = length(layer.p_neurons);
                biasNeuron = layer.p_neurons(lastNeuronPos); % Bias neuron is the last
                biasNeuron = biasNeuron.SetOutputValue(1);
                layer = layer.UpdateNeuron(biasNeuron, lastNeuronPos); % Updates neuron in layer with new value
                result.p_layers(i) = layer;
            end
            
            % Add input values to input neurons excuding bias
            layer = result.p_layers(1);
            for i = 1:length(layer.p_neurons) - 1
                neuron = layer.p_neurons(i);
                neuron = neuron.SetOutputValue(prmInputValues(i));
                layer = UpdateNeuron(layer, neuron, i); % Updates neuron in layer with new value
            end
            result.p_layers(1) = layer;
            
            % Calculate output for all neurons in all layers excluding bias
            for i = 2:length(result.p_layers)
                prevLayer = result.p_layers(i - 1);
                layer = result.p_layers(i);
                numNeurons = length(layer.p_neurons) - 1; % removing bias in every layer
                
                % Last layer has no bias
                if(i == length(result.p_layers))
                   numNeurons = numNeurons + 1; % Bias was removed before, add it
                end
                
                % Get output of each neuron of the layer
                for j = 1:numNeurons
                    neuron = layer.p_neurons(j);
                    neuron = neuron.CalculateOutputValue(prevLayer, j);
                    layer = layer.UpdateNeuron(neuron, j); % Updates neuron in layer with new value
                    result.p_layers(i) = layer;
                end
            end
            
            lastLayer = result.p_layers(length(result.p_layers));
            lastNeuron = lastLayer.p_neurons(1);
            result.p_outputValue = lastNeuron.p_outputValue;
        end
        
        %% Backpropagation
        % Propagates the error backwards in the neural network to adjust
        % weights.
        % Uses root mean square of the error.
        function result = Backpropagation(result, prmTargetValue)
            error = 0.0;
            
            % Calculate error from last layer using root mean square
            lastLayer = result.p_layers(length(result.p_layers));
            for i = 1:length(lastLayer.p_neurons)
                neuron = lastLayer.p_neurons(i);
                deltaError = prmTargetValue - neuron.p_outputValue;
                error = error + (deltaError * deltaError);
            end
            error = error/length(lastLayer.p_neurons);
            result.p_error = sqrt(error);
            
            % Calculate gradient in last layer
            for i = 1:length(lastLayer.p_neurons)
                neuron = lastLayer.p_neurons(i);
                neuron = neuron.CalculateOutputGradient(prmTargetValue);
                lastLayer = lastLayer.UpdateNeuron(neuron, i); % Updates neuron in layer with new value
                result.p_layers(length(result.p_layers)) = lastLayer;
            end
            
            % Calculate gradient in previous layers (excluding input and output layer)
            for i = length(result.p_layers) - 1:-1:2
                layer = result.p_layers(i);
                nextLayer = result.p_layers(i + 1);
                
                for j = 1:length(layer.p_neurons)
                    neuron = layer.p_neurons(j);
                    neuron = neuron.CalculateLayerGradient(nextLayer);
                    layer = layer.UpdateNeuron(neuron, j); % Updates neuron in layer with new value
                    result.p_layers(i) = layer;
                end
            end
            
            % Calculate weights with calculated gradient for all layers (excluding input layer)
            for i = length(result.p_layers):-1:2
                layer = result.p_layers(i);
                prevLayer = result.p_layers(i - 1);
                
                % Loop over every neuron on current layer
                numNeurosCurrentLayer = length(layer.p_neurons) - 1;
                
                % If current layer has 1 neuron, it has no bias.
                if(length(layer.p_neurons) == 1)
                    numNeurosCurrentLayer = numNeurosCurrentLayer + 1;
                end
                
                for j = 1:numNeurosCurrentLayer
                    
                    % Calculate delta weights of neurons in previous layer
                    for k = 1:length(prevLayer.p_neurons)
                        neuronPrevLayer = prevLayer.p_neurons(k);
                        prevDeltaWeight = neuronPrevLayer.p_outputWeightsDelta(j); % Output weight from previous layer to current neuron
                        deltaWeight = (result.p_learningRate * neuronPrevLayer.p_outputValue * neuronPrevLayer.p_gradient)...
                            + (result.p_momentum * prevDeltaWeight);
                        
                        weight = neuronPrevLayer.p_outputWeights(j);
                        neuronPrevLayer = neuronPrevLayer.UpdateOutputWeightDelta(deltaWeight, j);
                        neuronPrevLayer = neuronPrevLayer.UpdateOutputWeight(weight + deltaWeight, j);
                        
                        prevLayer = prevLayer.UpdateNeuron(neuronPrevLayer, k); % Updates neuron in layer with new value
                        result.p_layers(i - 1) = prevLayer;
                    end
                end
                
            end
            
        end
    end
    
    %% Private methods
    methods(Access = private)
        
    end
end
