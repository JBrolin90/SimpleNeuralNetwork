using System;
using BackPropagation;
using BackPropagation.NNLib;
using Xunit;
using Xunit.Abstractions;

namespace SimpleNeuralNetwork.Tests
{
    public class OneNeuronIntegrationTests
    {
        private readonly ITestOutputHelper _output;

        public OneNeuronIntegrationTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void OneNeuronTest_ShouldLearnLinearFunction_WithUnitActivation()
        {
            // Arrange
            var oneNeuronTest = new OneNeuronTest();

            // Act - Create and train the network
            oneNeuronTest.CreateNetwork();
            var trainingData = oneNeuronTest.CreateTrainingData();
            oneNeuronTest.Train();

            // Assert training data is correctly set up
            Assert.Equal(2, trainingData.Length);
            Assert.Equal(0, trainingData[0]);
            Assert.Equal(1, trainingData[1]);

            // Test predictions to verify the network learned the correct linear function
            // Expected: y = 0.3 * x + 0.3 (approximately)
            var network = oneNeuronTest.GetNetwork();
            
            // Test input 0 -> should output approximately 0.3
            var prediction0 = network.Predict([0]);
            _output.WriteLine($"Input 0 -> Output: {prediction0[0]} (Expected: ~0.3)");
            Assert.True(Math.Abs(prediction0[0] - 0.3) < 0.01, 
                $"Expected output for input 0 to be close to 0.3, but got {prediction0[0]}");

            // Test input 1 -> should output approximately 0.6
            var prediction1 = network.Predict([1]);
            _output.WriteLine($"Input 1 -> Output: {prediction1[0]} (Expected: ~0.6)");
            Assert.True(Math.Abs(prediction1[0] - 0.6) < 0.01, 
                $"Expected output for input 1 to be close to 0.6, but got {prediction1[0]}");

            // Test additional values to verify linearity
            var prediction10 = network.Predict([10]);
            _output.WriteLine($"Input 10 -> Output: {prediction10[0]} (Expected: ~3.3)");
            Assert.True(Math.Abs(prediction10[0] - 3.3) < 0.1, 
                $"Expected output for input 10 to be close to 3.3, but got {prediction10[0]}");

            var predictionNeg5 = network.Predict([-5]);
            _output.WriteLine($"Input -5 -> Output: {predictionNeg5[0]} (Expected: ~-1.2)");
            Assert.True(Math.Abs(predictionNeg5[0] - (-1.2)) < 0.1, 
                $"Expected output for input -5 to be close to -1.2, but got {predictionNeg5[0]}");
        }

        [Fact]
        public void OneNeuronTest_NetworkStructure_ShouldBeCorrect()
        {
            // Arrange & Act
            var oneNeuronTest = new OneNeuronTest();
            oneNeuronTest.CreateNetwork();
            var network = oneNeuronTest.GetNetwork();

            // Assert network structure
            Assert.NotNull(network);
            
            // Verify network can handle single input
            var testInput = new double[] { 0.5 };
            var output = network.Predict(testInput);
            
            Assert.Single(output); // Should have exactly one output
        }

        [Fact]
        public void OneNeuronTest_TrainingData_ShouldMatchExpectedValues()
        {
            // Arrange
            var oneNeuronTest = new OneNeuronTest();

            // Act
            var trainingData = oneNeuronTest.CreateTrainingData();

            // Assert
            Assert.NotNull(trainingData);
            Assert.Equal(2, trainingData.Length);
            Assert.Equal(0, trainingData[0]);
            Assert.Equal(1, trainingData[1]);
        }

        [Fact]
        public void OneNeuronTest_Training_ShouldImproveAccuracy()
        {
            // Arrange
            var oneNeuronTest = new OneNeuronTest();
            oneNeuronTest.CreateNetwork();
            oneNeuronTest.CreateTrainingData();
            var network = oneNeuronTest.GetNetwork();

            // Get initial predictions (before training)
            var initialPrediction0 = network.Predict([0])[0];
            var initialPrediction1 = network.Predict([1])[0];
            
            var initialError0 = Math.Abs(initialPrediction0 - 0.3);
            var initialError1 = Math.Abs(initialPrediction1 - 0.6);
            var initialTotalError = initialError0 + initialError1;

            _output.WriteLine($"Initial prediction for 0: {initialPrediction0} (target: 0.3, error: {initialError0})");
            _output.WriteLine($"Initial prediction for 1: {initialPrediction1} (target: 0.6, error: {initialError1})");
            _output.WriteLine($"Initial total error: {initialTotalError}");

            // Act - Train the network
            oneNeuronTest.Train();

            // Get final predictions (after training)
            var finalPrediction0 = network.Predict([0])[0];
            var finalPrediction1 = network.Predict([1])[0];
            
            var finalError0 = Math.Abs(finalPrediction0 - 0.3);
            var finalError1 = Math.Abs(finalPrediction1 - 0.6);
            var finalTotalError = finalError0 + finalError1;

            _output.WriteLine($"Final prediction for 0: {finalPrediction0} (target: 0.3, error: {finalError0})");
            _output.WriteLine($"Final prediction for 1: {finalPrediction1} (target: 0.6, error: {finalError1})");
            _output.WriteLine($"Final total error: {finalTotalError}");

            // Assert that training improved the accuracy
            Assert.True(finalTotalError < initialTotalError, 
                $"Training should improve accuracy. Initial error: {initialTotalError}, Final error: {finalTotalError}");
            
            // Assert that final error is reasonably small
            Assert.True(finalTotalError < 0.1, 
                $"Final total error should be small (< 0.1), but was {finalTotalError}");
        }

        [Fact]
        public void OneNeuronTest_MultipleRuns_ShouldProduceConsistentResults()
        {
            // This test verifies that the training process is deterministic
            // or at least produces consistently good results

            var results = new double[3][];
            
            for (int run = 0; run < 3; run++)
            {
                var oneNeuronTest = new OneNeuronTest();
                oneNeuronTest.CreateNetwork();
                oneNeuronTest.CreateTrainingData();
                oneNeuronTest.Train();
                
                var network = oneNeuronTest.GetNetwork();
                results[run] = new double[]
                {
                    network.Predict([0])[0],
                    network.Predict([1])[0]
                };
                
                _output.WriteLine($"Run {run + 1}: Input 0 -> {results[run][0]}, Input 1 -> {results[run][1]}");
            }

            // All runs should produce similar results (within tolerance)
            for (int i = 1; i < results.Length; i++)
            {
                Assert.True(Math.Abs(results[i][0] - results[0][0]) < 0.01,
                    $"Results should be consistent across runs for input 0");
                Assert.True(Math.Abs(results[i][1] - results[0][1]) < 0.01,
                    $"Results should be consistent across runs for input 1");
            }
        }

        [Fact]
        public void OneNeuronTest_LinearRelationship_ShouldBeVerifiable()
        {
            // Arrange & Act
            var oneNeuronTest = new OneNeuronTest();
            oneNeuronTest.CreateNetwork();
            oneNeuronTest.CreateTrainingData();
            oneNeuronTest.Train();
            
            var network = oneNeuronTest.GetNetwork();

            // Test various inputs to verify linear relationship
            var testInputs = new double[] { -2, -1, 0, 1, 2, 3, 5 };
            var predictions = new double[testInputs.Length];
            
            for (int i = 0; i < testInputs.Length; i++)
            {
                predictions[i] = network.Predict([testInputs[i]])[0];
                _output.WriteLine($"Input: {testInputs[i]} -> Output: {predictions[i]}");
            }

            // Verify the relationship is approximately linear: y = 0.3x + 0.3
            for (int i = 0; i < testInputs.Length; i++)
            {
                var expected = 0.3 * testInputs[i] + 0.3;
                var actual = predictions[i];
                var error = Math.Abs(actual - expected);
                
                Assert.True(error < 0.1, 
                    $"For input {testInputs[i]}, expected ~{expected:F3} but got {actual:F3} (error: {error:F3})");
            }
        }

        [Theory]
        [InlineData(0, 0.3)]
        [InlineData(1, 0.6)]
        [InlineData(2, 0.9)]
        [InlineData(-1, 0.0)]
        [InlineData(10, 3.3)]
        [InlineData(-5, -1.2)]
        public void OneNeuronTest_SpecificInputs_ShouldProduceExpectedOutputs(double input, double expectedOutput)
        {
            // Arrange
            var oneNeuronTest = new OneNeuronTest();
            oneNeuronTest.CreateNetwork();
            oneNeuronTest.CreateTrainingData();
            oneNeuronTest.Train();
            
            var network = oneNeuronTest.GetNetwork();

            // Act
            var actualOutput = network.Predict([input])[0];

            // Assert
            var tolerance = Math.Abs(expectedOutput) > 1 ? 0.2 : 0.05; // Larger tolerance for larger values
            _output.WriteLine($"Input: {input}, Expected: {expectedOutput}, Actual: {actualOutput}, Tolerance: {tolerance}");
            
            Assert.True(Math.Abs(actualOutput - expectedOutput) < tolerance,
                $"For input {input}, expected output ~{expectedOutput} but got {actualOutput}");
        }

    [Fact]
    public void TrainSingleNeuron_ConvergesToCorrectWeightsAndBiases()
    {
        // Arrange
        var test = new OneNeuronTest();
        test.CreateNetwork();
        test.CreateTrainingData();

        // Act
        test.Train();
        var network = test.GetNetwork();
        
        // Assert - After training on inputs [0,1] -> outputs [0.3,0.6]
        // The network should learn approximately: y = 0.3 * x + 0.3
        // So weight ≈ 0.3 and bias ≈ 0.3
        var weights = network.Weigths[0][0]; // First (and only) layer, first (and only) node
        
        // Test specific predictions to verify the learned function
        var prediction0 = network.Predict([0]);
        var prediction1 = network.Predict([1]);

        // Allow some tolerance for convergence
        Assert.True(Math.Abs(weights[0] - 0.3) < 0.01, $"Weight should be approximately 0.3, but was {weights[0]}");
        Assert.True(Math.Abs(prediction0[0] - 0.3) < 0.01, $"Prediction for input 0 should be approximately 0.3, but was {prediction0[0]}");
        Assert.True(Math.Abs(prediction1[0] - 0.6) < 0.01, $"Prediction for input 1 should be approximately 0.6, but was {prediction1[0]}");
    }

    [Fact]
    public void OneNeuronTraining_CompleteIntegrationTest_VerifiesAllAspects()
    {
        // Arrange & Act
        var oneNeuronTest = new OneNeuronTest();
        oneNeuronTest.Execute();
        var network = oneNeuronTest.GetNetwork();

        // Assert: Verify network structure
        Assert.Single(network.Layers);
        var outputLayer = network.Layers[0];
        Assert.NotNull(outputLayer);
        Assert.Single(outputLayer.Neurons);

        // Assert: Verify learned parameters are close to expected values
        var outputNeuron = outputLayer.Neurons[0];
        
        // Expected: f(0) ≈ 0.3, f(1) ≈ 0.6
        // This means: output ≈ weight * input + bias
        // From training data: 0.3 = w*0 + b → b ≈ 0.3
        //                    0.6 = w*1 + b → w ≈ 0.3
        
        // Get weights and biases from the network structure
        var weights = network.Weigths[0][0]; // First layer, first neuron
        var biases = network.Biases[0][0]; // First layer, first neuron
        
        Assert.InRange(biases[0], 0.25, 0.35);
        Assert.Single(weights);
        Assert.InRange(weights[0], 0.25, 0.35);

        // Assert: Verify network can make accurate predictions
        var prediction1 = network.Predict(new[] { 0.0 });
        var prediction2 = network.Predict(new[] { 1.0 });
        
        Assert.InRange(prediction1[0], 0.29, 0.31); // f(0) ≈ 0.3
        Assert.InRange(prediction2[0], 0.59, 0.61); // f(1) ≈ 0.6

        // Assert: Verify linear relationship holds
        var prediction3 = network.Predict(new[] { 0.5 });
        var expectedMidpoint = (prediction1[0] + prediction2[0]) / 2;
        Assert.InRange(prediction3[0], expectedMidpoint - 0.01, expectedMidpoint + 0.01);

        // Output summary for verification
        _output.WriteLine($"Final weight: {weights[0]:F6}");
        _output.WriteLine($"Final bias: {biases[0]:F6}");
        _output.WriteLine($"f(0) = {prediction1[0]:F6}");
        _output.WriteLine($"f(1) = {prediction2[0]:F6}");
        _output.WriteLine($"f(0.5) = {prediction3[0]:F6}");
        _output.WriteLine("✅ One neuron successfully learned linear relationship: f(x) ≈ 0.3x + 0.3");
    }
    }
}
