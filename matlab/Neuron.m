%% Neuron
classdef Neuron
    %% Properties
    properties (GetAccess = 'public', SetAccess = 'private')
        p_outputValue;
        p_outputWeights;
        p_outputWeightsDelta;
        p_gradient;
    end
    
    %% Public methods
    methods
        %% Constructor
        % Receives a matrix of the output weights for the next layers of
        % neurons
        function result = Neuron(prmOutputWeights)
            result.p_outputValue = 0;
            result.p_gradient = 0;
            result.p_outputWeights = prmOutputWeights;
            result.p_outputWeightsDelta = result.p_outputWeights * 0;
        end
        
        %% SetOutputValue
        % Sets the output value
        function result = SetOutputValue(result, prmOutputValue)
            result.p_outputValue = prmOutputValue;
        end
        
        %% UpdateOutputWeight
        % Updates the output weight in the given index
        function result = UpdateOutputWeight(result, prmValue, prmWeightIndex)
            result.p_outputWeights(prmWeightIndex) = prmValue;
        end
        
        %% UpdateOutputWeightDelta
        % Updates the output weight delta in the given index
        function result = UpdateOutputWeightDelta(result, prmValue, prmWeightIndex)
            result.p_outputWeightsDelta(prmWeightIndex) = prmValue;
        end
        
        %% CalculateOutputValue
        % Calculates the output value, using the previous layer
        function result = CalculateOutputValue(result, prmPreviousLayer, prmWeightIndex)
            value = 0;
            
            % Sum all outputs from neurons and multiply to the weights
            for i = 1:length(prmPreviousLayer.p_neurons)
                neuron = prmPreviousLayer.p_neurons(i);
                val = (neuron.p_outputValue * neuron.p_outputWeights(prmWeightIndex)); % output times weight
                value = value + val;
            end
            
            % Use activation function
            newValue = Neuron.ActivationFunctionHT(value); % Uses hyperbolic tangent
                        
            result = result.SetOutputValue(newValue);
        end
        
        %% CalculateOutputGradient
        % Calculates the output gradient, using the target value
        function result = CalculateOutputGradient(result, prmTargetValue)
            deltaError = prmTargetValue - result.p_outputValue;
            result.p_gradient = deltaError * result.ActivationFunctionHTDerivative(result.p_outputValue);
        end
        
        %% CalculateLayerGradient
        % Calculates the layer gradient, using the next layer
        function result = CalculateLayerGradient(result, prmNextLayer)
            sumGradient = 0;
            numNeurons = length(prmNextLayer.p_neurons) - 1; % Gets neurons on next layer without bias
            
            % If next layer has 1 neuron, it has no bias.
            if(length(prmNextLayer.p_neurons) == 1)
                numNeurons = numNeurons + 1;
            end
            
            % Calculate sum of gradients times weights in nodes in next layer
            for i = 1:numNeurons
                neuron = prmNextLayer.p_neurons(i);
                sumGradient = sumGradient + (result.p_outputWeights(i) * neuron.p_gradient); % Weight times gradient
            end
            
            result.p_gradient = sumGradient * result.ActivationFunctionHTDerivative(result.p_outputValue);
        end
    end
    
    %% Static methods
    methods(Static)
        %% ActivationFunctionS
        % Uses sigmoid function. Results are values from 0 to 1
        function result = ActivationFunctionS(inputValue)
            result = logsig(inputValue);
        end
        
        %% ActivationFunctionHT
        % Uses hyperbolic tangent function. Results are values from -1 to 1
        function result = ActivationFunctionHT(inputValue)
            result = tanh(inputValue);
        end
        
        %% ActivationFunctionHTS
        % Uses hyperbolic tangent sigmoid function. Results are values from -1 to 1
        function result = ActivationFunctionHTS(inputValue)
            result = tansig(inputValue);
        end
        
        %% ActivationFunctionHTDerivative
        % Uses hyperbolic tangent function derivative.
        function result = ActivationFunctionHTDerivative(inputValue)
            result = 1 - (tanh(inputValue) * tanh(inputValue));
        end
    end
end
