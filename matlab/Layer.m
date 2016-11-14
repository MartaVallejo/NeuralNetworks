%% Layer of the neural network with its neurons
classdef Layer
    %% Properties
    properties (GetAccess = 'public', SetAccess = 'private')
        p_neurons;
    end
    
    %% Public methods
    methods
        %% Constructor
        % Creates a layer with an empty matrix of nx1 where n is the number
        % of neurons. Each row is a neuron.
        function result = Layer(prmNeurons)
            result.p_neurons = Neuron.empty(prmNeurons, 0);
            disp(result.p_neurons)
        end
        
        %% UpdateNeuron
        % Updates a neuron in the matrix of neurons at the specified index
        function result = UpdateNeuron(result, prmNeuron, prmIndex)
            result.p_neurons(prmIndex) = prmNeuron;
        end
    end
end
