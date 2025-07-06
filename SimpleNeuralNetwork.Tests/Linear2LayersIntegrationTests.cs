using System;
using Xunit;
using BackPropagation;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class Linear2LayersIntegrationTests
    {
        private const double Tolerance = 1e-6;
        
        [Fact]
        public void Linear2LayersTest_ShouldCompleteWithoutErrors()
        {
            // Arrange
            var test = new Linear2LayersTest();
            
            // Act & Assert - Should not throw any exceptions
            var exception = Record.Exception(() => test.DoIt());
            Assert.Null(exception);
        }

        [Fact]
        public void Linear2LayersTest_NetworkCreation_ShouldSucceed()
        {
            // Arrange
            var test = new Linear2LayersTest();
            
            // Act & Assert - Should not throw any exceptions
            var exception = Record.Exception(() => test.CreateNetwork());
            Assert.Null(exception);
        }

        [Fact]
        public void Linear2LayersTest_TrainingDataCreation_ShouldSucceed()
        {
            // Arrange
            var test = new Linear2LayersTest();
            
            // Act & Assert - Should not throw any exceptions
            var exception = Record.Exception(() => test.CreateTrainingData());
            Assert.Null(exception);
        }

        [Fact]
        public void Linear2LayersTest_Training_ShouldSucceed()
        {
            // Arrange
            var test = new Linear2LayersTest();
            test.CreateNetwork();
            test.CreateTrainingData();
            
            // Act & Assert - Should not throw any exceptions
            var exception = Record.Exception(() => test.Train());
            Assert.Null(exception);
        }

        [Fact]
        public void Linear2LayersTest_FullWorkflow_ShouldProduceExpectedOutput()
        {
            // Arrange
            var test = new Linear2LayersTest();
            
            // Capture console output
            var originalConsoleOut = Console.Out;
            using var consoleOutput = new System.IO.StringWriter();
            Console.SetOut(consoleOutput);
            
            try
            {
                // Act
                test.DoIt();
                
                // Assert
                var output = consoleOutput.ToString();
                
                // Check that the output contains expected predictions
                Assert.Contains("0 =>", output);
                Assert.Contains("1 =>", output);
                Assert.Contains("0.5 =>", output);
                Assert.Contains("After training:", output);
                
                // The output should not contain "Algorithms differ" which would indicate a problem
                Assert.DoesNotContain("Algorithms differ", output);
            }
            finally
            {
                Console.SetOut(originalConsoleOut);
            }
        }

        [Fact]
        public void Linear2LayersTest_RepeatedExecution_ShouldBeConsistent()
        {
            // Arrange & Act
            var outputs = new string[3];
            
            for (int i = 0; i < 3; i++)
            {
                var test = new Linear2LayersTest();
                
                // Capture console output
                var originalConsoleOut = Console.Out;
                using var consoleOutput = new System.IO.StringWriter();
                Console.SetOut(consoleOutput);
                
                try
                {
                    test.DoIt();
                    outputs[i] = consoleOutput.ToString();
                }
                finally
                {
                    Console.SetOut(originalConsoleOut);
                }
            }
            
            // Assert - All runs should produce the same output (deterministic)
            Assert.Equal(outputs[0], outputs[1]);
            Assert.Equal(outputs[1], outputs[2]);
        }

        [Fact]
        public void Linear2LayersTest_OutputFormat_ShouldBeCorrect()
        {
            // Arrange
            var test = new Linear2LayersTest();
            
            // Capture console output
            var originalConsoleOut = Console.Out;
            using var consoleOutput = new System.IO.StringWriter();
            Console.SetOut(consoleOutput);
            
            try
            {
                // Act
                test.DoIt();
                
                // Assert
                var output = consoleOutput.ToString();
                var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                
                // Should have at least 6 lines of output (before and after training)
                Assert.True(lines.Length >= 6, $"Expected at least 6 lines of output, got {lines.Length}");
                
                // Check format of prediction lines
                var predictionLines = lines.Where(line => line.Contains(" => ")).ToArray();
                Assert.True(predictionLines.Length >= 6, $"Expected at least 6 prediction lines, got {predictionLines.Length}");
                
                // Each prediction line should have the format "input => output"
                foreach (var line in predictionLines)
                {
                    Assert.Contains(" => ", line);
                    var parts = line.Split(" => ");
                    Assert.Equal(2, parts.Length);
                    
                    // Input should be a number
                    Assert.True(double.TryParse(parts[0], out _), $"Input '{parts[0]}' should be a number");
                    
                    // Output should be a number
                    Assert.True(double.TryParse(parts[1], out _), $"Output '{parts[1]}' should be a number");
                }
            }
            finally
            {
                Console.SetOut(originalConsoleOut);
            }
        }

        [Fact]
        public void Linear2LayersTest_LearningBehavior_ShouldBeReasonable()
        {
            // Arrange
            var test = new Linear2LayersTest();
            
            // Capture console output
            var originalConsoleOut = Console.Out;
            using var consoleOutput = new System.IO.StringWriter();
            Console.SetOut(consoleOutput);
            
            try
            {
                // Act
                test.DoIt();
                
                // Assert
                var output = consoleOutput.ToString();
                var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                
                // Find the "After training:" line
                var afterTrainingIndex = Array.FindIndex(lines, line => line.Contains("After training:"));
                Assert.True(afterTrainingIndex >= 0, "Should contain 'After training:' line");
                
                // Extract final predictions
                var finalPredictions = new Dictionary<string, double>();
                for (int i = afterTrainingIndex + 1; i < lines.Length; i++)
                {
                    var line = lines[i];
                    if (line.Contains(" => "))
                    {
                        var parts = line.Split(" => ");
                        if (parts.Length == 2 && double.TryParse(parts[0], out var inputVal) && double.TryParse(parts[1], out var outputVal))
                        {
                            finalPredictions[parts[0]] = outputVal;
                        }
                    }
                }
                
                // Check that we have predictions for the expected inputs
                Assert.True(finalPredictions.ContainsKey("0"), "Should have prediction for input 0");
                Assert.True(finalPredictions.ContainsKey("1"), "Should have prediction for input 1");
                Assert.True(finalPredictions.ContainsKey("0.5"), "Should have prediction for input 0.5");
                
                // For a linear function y = 0.3 * x, predictions should be reasonable
                // Allow for some training error
                var pred0 = finalPredictions["0"];
                var pred1 = finalPredictions["1"];
                var pred05 = finalPredictions["0.5"];
                
                Assert.True(Math.Abs(pred0 - 0.3) < 0.1, $"Prediction for input 0 should be close to 0.3, got {pred0}");
                Assert.True(Math.Abs(pred1 - 0.6) < 0.1, $"Prediction for input 1 should be close to 0.6, got {pred1}");
                Assert.True(Math.Abs(pred05 - 0.45) < 0.1, $"Prediction for input 0.5 should be close to 0.45, got {pred05}");
                
                // The relationship should be roughly linear
                var slope = (pred1 - pred0) / (1 - 0);
                Assert.True(Math.Abs(slope - 0.3) < 0.1, $"Slope should be close to 0.3, got {slope}");
            }
            finally
            {
                Console.SetOut(originalConsoleOut);
            }
        }

        [Fact]
        public void Linear2LayersTest_TrainingConvergence_ShouldNotShowAlgorithmDifferences()
        {
            // Arrange
            var test = new Linear2LayersTest();
            
            // Capture console output
            var originalConsoleOut = Console.Out;
            using var consoleOutput = new System.IO.StringWriter();
            Console.SetOut(consoleOutput);
            
            try
            {
                // Act
                test.CreateNetwork();
                test.CreateTrainingData();
                test.Train();
                
                // Assert
                var output = consoleOutput.ToString();
                
                // The training should not show "Algorithms differ" which would indicate
                // inconsistency between the neural network and the manual verifier
                Assert.DoesNotContain("Algorithms differ", output);
            }
            finally
            {
                Console.SetOut(originalConsoleOut);
            }
        }

        [Fact]
        public void Linear2LayersTest_PredictionAccuracy_ShouldImproveWithTraining()
        {
            // This test verifies that the network learns the linear relationship y = 0.3 * x
            // by checking that predictions become more accurate after training
            
            // Arrange
            var test = new Linear2LayersTest();
            
            // Capture console output
            var originalConsoleOut = Console.Out;
            using var consoleOutput = new System.IO.StringWriter();
            Console.SetOut(consoleOutput);
            
            try
            {
                // Act
                test.DoIt();
                
                // Assert
                var output = consoleOutput.ToString();
                
                // Extract predictions before and after training
                var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                
                // Find initial predictions (before "After training:")
                var beforeTrainingPredictions = new Dictionary<string, double>();
                var afterTrainingPredictions = new Dictionary<string, double>();
                
                bool isAfterTraining = false;
                foreach (var line in lines)
                {
                    if (line.Contains("After training:"))
                    {
                        isAfterTraining = true;
                        continue;
                    }
                    
                    if (line.Contains(" => "))
                    {
                        var parts = line.Split(" => ");
                        if (parts.Length == 2 && double.TryParse(parts[0], out var inputVal) && double.TryParse(parts[1], out var outputVal))
                        {
                            if (isAfterTraining)
                            {
                                afterTrainingPredictions[parts[0]] = outputVal;
                            }
                            else
                            {
                                beforeTrainingPredictions[parts[0]] = outputVal;
                            }
                        }
                    }
                }
                
                // We should have predictions for both before and after training
                Assert.True(beforeTrainingPredictions.Count > 0, "Should have predictions before training");
                Assert.True(afterTrainingPredictions.Count > 0, "Should have predictions after training");
                
                // Check that common inputs have improved predictions
                foreach (var kvp in beforeTrainingPredictions)
                {
                    var input = kvp.Key;
                    var beforePrediction = kvp.Value;
                    
                    if (afterTrainingPredictions.ContainsKey(input))
                    {
                        var afterPrediction = afterTrainingPredictions[input];
                        var expectedOutput = 0.3 * double.Parse(input);
                        
                        var beforeError = Math.Abs(beforePrediction - expectedOutput);
                        var afterError = Math.Abs(afterPrediction - expectedOutput);
                        
                        // After training should be at least as good as before training
                        Assert.True(afterError <= beforeError + 0.01, 
                            $"Prediction for input {input} should improve or stay the same: before error={beforeError}, after error={afterError}");
                    }
                }
            }
            finally
            {
                Console.SetOut(originalConsoleOut);
            }
        }
    }
}
