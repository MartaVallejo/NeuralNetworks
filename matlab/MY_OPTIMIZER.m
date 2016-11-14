function xbest = MY_OPTIMIZER(FUN, DIM, ftarget, maxfunevals)
% MY_OPTIMIZER(FUN, DIM, ftarget, maxfunevals)
% samples new points uniformly randomly in [-5,5]^DIM
% and evaluates them on FUN until ftarget of maxfunevals
% is reached, or until 1e8 * DIM fevals are conducted. 

  maxfunevals = min(1e8 * DIM, maxfunevals); 
  popsize = min(maxfunevals, 200);
  fbest = inf;
  for iter = 1:ceil(maxfunevals/popsize)
    xpop = 10 * rand(DIM, popsize) - 5;      % new solutions
    [fvalues, idx] = sort(feval(FUN, xpop)); % evaluate
    if fbest > fvalues(1)                    % keep best
      fbest = fvalues(1);
      xbest = xpop(:,idx(1));
    end
    if feval(FUN, 'fbest') < ftarget         % COCO-task achieved
      break;                                 % (works also for noisy functions)
    end
  end 

  
  
  
useGeneticAlgorithm = 1;
numInputsNeurons = DIM; % use dimensions
numOutputNeurons = 1;
learningRate = 0.1; % Learning rate for backpropagation. Values from 0 to 1
momentum = 0.5; % Delta weight multiplier for backpropagation. Values from 0 to 1
mutationRate = 0.01; % Mutation rate for Genetic Algorithm
expecteValue = 0.5; % test value, TODO: change to expected value by COCO
numOfNeuralNetworks = randi([5, 20]); % Random number of neural networks to evaluate
numInitialPopulationGA = randi([5, 20]); % Random number of neural networks to evaluate
numOfIterations = 500; % Random number of iterations to evaluate each neural network

if(useGeneticAlgorithm) % Use Genetic Algorithm
    
    % Generate all neural networks to evaluate with different topologies
    for i = 1:numOfNeuralNetworks
        
        numLayers = randi([3, 10]); % Random Number of layers
        topology = zeros(0, numLayers);
        
        % input layer
        topology(1) = numInputsNeurons; % Neurons in input layer
        % for each hidden layer, create random number of neurons
        for j = 2:numLayers - 1
            topology(j) = randi([2, 10]); % Random number of neurons
        end
        topology(numLayers) = numOutputNeurons; % Neurons in last layer
        
        %TEST
        %topology = [2 3 2 4 1];
        
        numInputs = topology(1); % Number of inputs = neurons in first layer
        inputValues = 2*rand(numInputs,1)-1; % Random input values between -1 and 1
        
        % For each topology generate a neural network with same topology and
        % with different random weights (weights are random in the constructor)
        % Generate initial population
        population = NeuralNetwork.empty(0, numInitialPopulationGA);
        for j = 1:numInitialPopulationGA
            nn = NeuralNetwork(topology, learningRate, momentum);
            population(j) = nn;
        end
        
        isFound = 0;
        bestError = realmax;
        numOfCurrentGenerations = 1;
        while (~isFound && numOfCurrentGenerations < numOfIterations)
            % Evaluate fitness for every element of the population
            
            % Evaluate population
            for j = 1:numInitialPopulationGA
                % Evaluate fitness (fitness is error. The higher the error, the lower the fitness)
                nn = population(j);
                nn = nn.TrainNeuralNetwork_GeneticAlgorithm(inputValues, expecteValue);
                
                if(nn.p_error < bestError)
                    bestError = nn.p_error;
                    bestNeuralNetwork = nn;
                    currentBest = nn;
                end
                
                if (nn.p_error == 0)
                    isFound = 1;
                    break;
                end
                
                population(j) = nn;
            end
            
            % if not found, create new generation
            if (~isFound)
                
                % Create new population with new generations. Every
                % generation will use the current best neuron network
                for populationIndex = 1:length(population)
                    element1 = population(populationIndex); % element at populationIndex as parent 1
                    % parent 2 is the best found so far.
                    element2 = currentBest;
                    
                    % Create new generation with crossover. Crossover
                    % is the average of the weights using the current
                    % neural network and the best neural network.
                    
                    % For each layer (both elements have the same number of layers)
                    for numLayerIndex = 1:length(element1.p_layers) - 1
                        % Gets both element's layers
                        layer1 = element1.p_layers(numLayerIndex);
                        layer2 = element2.p_layers(numLayerIndex);
                        
                        % For each neuron (both elements have the same number of neurons)
                        for numNeuronIndex = 1:length(layer1.p_neurons)
                            neuron1 = layer1.p_neurons(numNeuronIndex);
                            neuron2 = layer2.p_neurons(numNeuronIndex);
                            
                            % For each output weights, crossover
                            for numWeightIndex = 1:length(neuron1.p_outputWeights)
                                weight1 = neuron1.p_outputWeights(numWeightIndex);
                                weight2 = neuron2.p_outputWeights(numWeightIndex);
                                crossoverWeights = (weight1 + weight2) / 2;
                                
                                % Mutate element using mutationRate
                                bitMutate = (randi([1, 100])) / 100;
                                if (bitMutate <= mutationRate)
                                    mutationMomentum = (randi([1, 100])) / 100;
                                    crossoverWeights = crossoverWeights * mutationMomentum;
                                end
                                
                                neuron1 = neuron1.UpdateOutputWeight(crossoverWeights, numWeightIndex); % update value of neuron
                                layer1 = layer1.UpdateNeuron(neuron1, numNeuronIndex); % update neuron in layer
                                element1 = element1.UpdateLayer(layer1, numLayerIndex); % update layer in neural network
                                population(populationIndex) = element1;% update neural network in population
                            end
                        end
                    end
                end
            end
            
            numOfCurrentGenerations = numOfCurrentGenerations + 1;
        end
        
        % Display best weights of neural network with topology = i
        disp(bestNeuralNetwork.p_topology);
        
        fprintf('BEST: Output: %d | error %d \n', numOfCurrentGenerations, bestNeuralNetwork.p_outputValue, bestNeuralNetwork.p_error);
        
        %nn = nn.TrainNeuralNetwork_GeneticAlgorithm(inputValues, expecteValue, numOfIterations);
    end
else % Use BackPropagation
    % Generate all neural networks to evaluate with different topologies
    for i = 1:numOfNeuralNetworks
        
        numLayers = randi([3, 10]); % Random Number of layers
        topology = zeros(0, numLayers);
        
        % input layer
        topology(numLayers) = numInputsNeurons; % Neurons in input layer
        % for each hidden layer, create random number of neurons
        for j = 2:numLayers - 1
            topology(j) = randi([2, 10]); % Random number of neurons
        end
        topology(numLayers) = numOutputNeurons; % Neurons in last layer
        
        %TEST
        %topology = [2 3 2 4 1];
        
        numInputs = topology(1); % Number of inputs = neurons in first layer
        inputValues = 2*rand(numInputs,1)-1; % Random input values between -1 and 1
        
        % Create neural network with topology
        nn = NeuralNetwork(topology, learningRate, momentum);
        nn = nn.TrainNeuralNetwork_Backpropagation(inputValues, expecteValue, numOfIterations);
        
    end
end

  
