%{
---------------------------------------------------------------------
Test experiment using a genetic algorithm to find a value.
Author: Carlos Chavez
---------------------------------------------------------------------
%}
clc;
addpath('C:\code\HW\bbob.v15.03\matlab');

expectedElement = 0.5; % test value
mutationRate = 0.01; % specify percentage of mutation rate
populationCount = 50; % specify number of elements in the population
numOfGenerations = 100; % specify max number of iterations
gao = GeneticAlgorithm(mutationRate, populationCount);
gao = gao.RunGeneticAlgorithm(expectedElement, numOfGenerations);
