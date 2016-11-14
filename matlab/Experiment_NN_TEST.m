%{
---------------------------------------------------------------------
Test experiment using a neural network to find an expected answer with a given
function.
Author: Carlos Chavez
---------------------------------------------------------------------
%}
clc;
addpath('C:\code\HW\bbob.v15.03\matlab');
numOutputNeurons = 1;
learningRate = 0.1; % Learning rate for backpropagation. Values from 0 to 1
momentum = 0.5; % Delta weight multiplier. Values from 0 to 1
expecteValue = 0.5; % test value

numLayers = randi([3, 10]); % RandomnNumber of layers
topology = zeros(0, numLayers);
% for each layer, create random number of neurons
for i = 1:numLayers - 1
    topology(i) = randi([2, 10]); % Random number of neurons
end
topology(numLayers) = numOutputNeurons; % Last layer

%TEST
%topology = [2 2 2 2 2 2 1];

numInputs = topology(1); % Number of inputs = neurons in first layer
inputValues = 2*rand(numInputs,1)-1; % Random input values between -1 and 1

% Create neural network with topology
nn = NeuralNetwork(topology, learningRate, momentum);
nn = nn.TrainNeuralNetwork(inputValues, expecteValue, 500);


