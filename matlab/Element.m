%% Element
classdef Element
    %% Properties
    properties
       fitness
       value
    end
    
    %% Methods
    methods
        %% Constructor
        function result = Element(newValue)
            result.value = newValue;
            result.fitness = 0;
        end
    end
end