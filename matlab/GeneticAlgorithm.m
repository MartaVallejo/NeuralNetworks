%% Genetic algorithm
classdef GeneticAlgorithm
    
    %% Properties
    properties (GetAccess = 'public', SetAccess = 'private')
        p_mutationRate;
        p_populationCount;
        p_bestFitness;
        p_bestSolution;
        p_population;
    end
    
    %% Public methods
    methods
        %% Constructor
        function result = GeneticAlgorithm(prmMutationRate, prmPopulationCount)
            result.p_mutationRate = prmMutationRate;
            result.p_populationCount = prmPopulationCount;
            result.p_bestFitness = 0;
            result.p_bestSolution = inf;
            result.p_population = 2*rand(0, prmPopulationCount)-1; % Generates initial random population with values between -1 and 1
        end
        
        %% RunGeneticAlgorithm
        % Executes genetic algorithm to find the expected element within a
        % maximum number of generations
        function result = RunGeneticAlgorithm(result, prmExpectedValue, prmMaxGenerations)
            isFound = 0;
            numOfGenerations = 0;
            
            % 1) Generate initial population
            population = result.GeneratePopulation(result.populationCount, result.p_populationCount);
            
            while (~isFound && numOfGenerations < prmMaxGenerations)
                %2) Evaluate fitness for every element of the population
                for i = 1:length(population)
                    element = population(i);
                    fitness = EvaluateFitness(result, element, result.expectedElement);
                    element.fitness = fitness;
                    
                    %fprintf('Val: %s | Fitness: %d\n', element.value, element.fitness);
                    
                    if (fitness == 100)
                        isFound = 1;
                        foundElement = element;
                        break;
                    end
                    population(i) = element;
                end
                
                % if not found, create new generation
                if (~isFound)
                    % Create mating pool
                    matingPool = GenerateNewMatingPool(result, population);
                    
                    % 3) Create new population with new generations
                    for i = 1:length(population)
                        element1 = population(i); % element at i (default, should be changed by matingpool)
                        element2 = population(1); % element at first position (default, should be changed by matingpool)
                        randomPos1 = randi([1, length(matingPool)]);
                        randomPos2 = randi([1, length(matingPool)]);
                        
                        if (~isempty(matingPool))
                            % Choose parents using fitness (higher fitness, higher probability) using matingpool
                            element1 = matingPool(randomPos1);
                            element2 = matingPool(randomPos2);
                        end
                        
                        % 3.1) Create new generation with crossover, randomly mutate
                        newGenerationElement = CreateNewGeneration(result, element1, element2);
                        
                        % 3.2) Mutate element
                        newMutatedGenerationElement = MutateElement(result, newGenerationElement);
                        
                        population(i) = newMutatedGenerationElement;
                    end
                end
                
                numOfGenerations = numOfGenerations + 1;
                fprintf('Generations: %d\n', numOfGenerations);
            end
            
            fprintf('Best: in gen %d\n', foundElement.value, numOfGenerations);
        end
        
    end
    
    %% Private methods
    methods(Access = private)
        
        %% GeneratePopulation
        % Generates initial population with random numbers
        function result = GeneratePopulation(obj, prmPopulationNumber, numCharacters)
            result = {};
            TotalChars = length(obj.alphabet);
            
            for i = 1:prmPopulationNumber
                newElementValue = '';
                for j = 1:numCharacters
                    randomPosition = randi([1, TotalChars]); % Random position for character
                    character = obj.alphabet(randomPosition); % Random character
                    newElementValue = strcat(newElementValue, character);
                end
                
                newElement = Element(newElementValue);
                result = [result, newElement];
            end
        end
        
        %% GenerateNewMatingPool
        % Generates new mating pool
        % Elements are created n times as fitness they have.
        function result = GenerateNewMatingPool(obj, population)
            result = {};
            for elem = population
                numOfElements = elem.fitness;
                for i = 1:numOfElements
                    result = [result, elem];
                end
            end
        end
        
        %% EvaluateFitness
        % Evaluates fitness of element
        function result = EvaluateFitness(obj, element, expectedElementValue)
            error = expectedElementValue - element;
            
            result = error / expectedElementValue;
        end
        
        %% CreateNewGeneration
        % Creates a new generation using 2 parent elements
        function result = CreateNewGeneration(obj, element1, element2)
            newElementValue = '';
            elementValuePool = {};
            
            % Fill in pool with values of elements n times for probabilities (n = fitness)
            for i = 1:element1.fitness
                elementValuePool = [elementValuePool, element1.value];
            end
            for i = 1:element2.fitness
                elementValuePool = [elementValuePool, element2.value];
            end
            
            for i = 1:length(element1.value)
                numRand = randi([1, length(elementValuePool)]);
                tmpElement = char(elementValuePool(numRand)); % get random value (probabilities help)
                tmpElementValue = tmpElement(i); % get char of value at position i
                newElementValue = strcat(newElementValue, tmpElementValue); % create new value with char obtained
            end
            
            result = Element(newElementValue);
            
        end
        
        %% MutateElement
        % Mutates element to create a new mutated element
        % Changes a character randomly depending on mutation rate
        function result = MutateElement(obj, element)
            resultValue = '';
            TotalChars = length(obj.alphabet);
            
            for i = 1:length(element.value)
                currentCharElement = element.value(i);
                bitMutate = (randi([1, 100])) / 100;
                if (bitMutate <= obj.mutationRate)
                    randomPosition = randi([1, TotalChars]);
                    currentCharElement = obj.alphabet(randomPosition);
                end
                
                resultValue = strcat(resultValue, currentCharElement);
            end
            
            result = Element(resultValue);
        end
        
    end
end