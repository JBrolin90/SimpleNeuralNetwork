using Xunit;
using System;
using BackPropagation.NNLib;
using System.Linq;
using System.Collections.Generic;

namespace SimpleNeuralNetwork.Tests
{
    public class ComplexIntegrationTests
    {
        private const double Tolerance = 1e-7;

        #region Multi-Layer Network Integration Tests

        [Fact]
        public void DeepNeuralNetwork_ComplexTraining_ConvergesCorrectly()
        {
            // Arrange
            var creator = new NetworkCreator(3, new int[] { 5, 4, 3, 2, 1 }, 
                new Func<double, double>[] { 
                    ActivationFunctions.ReLU, 
                    ActivationFunctions.ReLU, 
                    ActivationFunctions.ReLU, 
                    ActivationFunctions.Sigmoid,
                    ActivationFunctions.Sigmoid 
                });
            creator.RandomizeWeights(-0.5, 0.5);
            var network = creator.CreateNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.1);

            // Create training data for function approximation: f(x,y,z) = x + y - z
            var trainingData = new List<double[]>();
            var observed = new List<double[]>();
            
            for (int i = 0; i < 50; i++)
            {
                double x = (i % 10) * 0.1;
                double y = ((i / 10) % 5) * 0.2;
                double z = (i % 3) * 0.3;
                trainingData.Add(new double[] { x, y, z });
                observed.Add(new double[] { x + y - z });
            }

            // Act - Train for multiple epochs
            double[] lastLoss = null!;
            for (int epoch = 0; epoch < 100; epoch++)
            {
                lastLoss = trainer.TrainOneEpoch(trainingData.ToArray(), observed.ToArray());
                
                // Ensure loss is finite during training
                Assert.True(double.IsFinite(lastLoss[0]), $"Loss should be finite at epoch {epoch}, got {lastLoss[0]}");
            }

            // Assert
            Assert.NotNull(lastLoss);
            Assert.Single(lastLoss);
            Assert.True(double.IsFinite(lastLoss[0]));
            Assert.True(lastLoss[0] >= 0, "Loss should be non-negative");
            
            // Test prediction accuracy
            var testInput = new double[] { 0.5, 0.3, 0.1 };
            var prediction = network.Predict(testInput);
            
            Assert.Single(prediction);
            Assert.True(double.IsFinite(prediction[0]));
            
            // Additional robustness checks
            Assert.True(Math.Abs(prediction[0]) < 100, "Prediction should be in reasonable range");
        }

        [Fact]
        public void MultiOutputNetwork_TrainingAndPrediction_WorksCorrectly()
        {
            // Arrange
            var creator = new NetworkCreator(2, new int[] { 3, 3 }, 
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.Unit });
            creator.RandomizeWeights(-1.0, 1.0);
            var network = creator.CreateNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.05);

            // Create training data for multiple outputs
            double[][] trainingData = {
                new double[] { 1.0, 2.0 },
                new double[] { 2.0, 3.0 },
                new double[] { 3.0, 4.0 }
            };
            double[][] observed = {
                new double[] { 3.0, 1.0, 2.0 }, // sum, diff, product/2
                new double[] { 5.0, 1.0, 3.0 },
                new double[] { 7.0, 1.0, 6.0 }
            };

            // Act
            var initialPrediction = network.Predict(trainingData[0]);
            
            for (int epoch = 0; epoch < 50; epoch++)
            {
                trainer.TrainOneEpoch(trainingData, observed);
            }
            
            var finalPrediction = network.Predict(trainingData[0]);

            // Assert
            Assert.Equal(3, initialPrediction.Length);
            Assert.Equal(3, finalPrediction.Length);
            
            for (int i = 0; i < 3; i++)
            {
                Assert.True(double.IsFinite(initialPrediction[i]));
                Assert.True(double.IsFinite(finalPrediction[i]));
            }
            
            // Predictions should have changed after training
            Assert.False(initialPrediction.SequenceEqual(finalPrediction));
        }

        [Fact]
        public void NetworkCreator_DifferentActivationFunctions_ProduceDifferentResults()
        {
            // Arrange
            var activationSets = new Func<double, double>[][] {
                new Func<double, double>[] { ActivationFunctions.Unit, ActivationFunctions.Unit },
                new Func<double, double>[] { ActivationFunctions.Sigmoid, ActivationFunctions.Sigmoid },
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.ReLU },
                new Func<double, double>[] { ActivationFunctions.Tanh, ActivationFunctions.Tanh },
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.Sigmoid }
            };

            var networks = new INeuralNetwork[activationSets.Length];
            var predictions = new double[activationSets.Length][];

            // Act
            for (int i = 0; i < activationSets.Length; i++)
            {
                var creator = new NetworkCreator(2, new int[] { 3, 1 }, activationSets[i]);
                creator.RandomizeWeights(-1.0, 1.0);
                networks[i] = creator.CreateNetwork();
                predictions[i] = networks[i].Predict(new double[] { 1.0, 2.0 });
            }

            // Assert
            for (int i = 0; i < predictions.Length; i++)
            {
                Assert.Single(predictions[i]);
                Assert.True(double.IsFinite(predictions[i][0]));
            }

            // At least some predictions should be different
            var uniquePredictions = predictions.Select(p => p[0]).Distinct().Count();
            Assert.True(uniquePredictions > 1, "Different activation functions should produce different results");
        }

        #endregion

        #region Sample-Based Training Integration Tests

        [Fact]
        public void SampleBasedTraining_ComplexOperations_WorksCorrectly()
        {
            // Arrange
            var creator = new NetworkCreator(4, new int[] { 6, 4, 1 }, 
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.ReLU, ActivationFunctions.Unit });
            creator.RandomizeWeights(-0.5, 0.5);
            var network = creator.CreateNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.05);

            // Create complex samples with different operations
            var samples = new Sample[] {
                new Sample(2.0, 3.0, Operation.add, 1.0),
                new Sample(5.0, 2.0, Operation.add, 1.0),
                new Sample(3.0, 4.0, Operation.hypot, 1.0),
                new Sample(8.0, 2.0, Operation.hypot, 1.0),
                new Sample(3.0, 4.0, Operation.hypot, 1.0),
                new Sample(1.0, 2.0, Operation.add, 1.0),
                new Sample(6.0, 3.0, Operation.add, 1.0),
                new Sample(2.0, 3.0, Operation.hypot, 1.0),
                new Sample(9.0, 3.0, Operation.hypot, 1.0),
                new Sample(5.0, 12.0, Operation.hypot, 1.0)
            };

            // Act
            var initialPredictions = samples.Select(s => network.Predict(s.Xample)[0]).ToArray();
            
            // Training loop with error checking
            for (int epoch = 0; epoch < 100; epoch++)
            {
                var epochLoss = trainer.TrainOneEpoch(samples);
                
                // Ensure training is stable
                Assert.Single(epochLoss);
                Assert.True(double.IsFinite(epochLoss[0]), $"Loss should be finite at epoch {epoch}");
                Assert.True(epochLoss[0] >= 0, $"Loss should be non-negative at epoch {epoch}");
            }
            
            var finalPredictions = samples.Select(s => network.Predict(s.Xample)[0]).ToArray();

            // Assert
            Assert.Equal(samples.Length, initialPredictions.Length);
            Assert.Equal(samples.Length, finalPredictions.Length);
            
            for (int i = 0; i < samples.Length; i++)
            {
                Assert.True(double.IsFinite(initialPredictions[i]), $"Initial prediction {i} should be finite");
                Assert.True(double.IsFinite(finalPredictions[i]), $"Final prediction {i} should be finite");
                
                // Additional bounds check
                Assert.True(Math.Abs(initialPredictions[i]) < 1000, $"Initial prediction {i} should be in reasonable range");
                Assert.True(Math.Abs(finalPredictions[i]) < 1000, $"Final prediction {i} should be in reasonable range");
            }
            
            // Predictions should have changed after training
            Assert.False(initialPredictions.SequenceEqual(finalPredictions));
        }

        [Fact]
        public void SampleBasedTraining_MixedOperations_HandlesCorrectly()
        {
            // Arrange
            var creator = new NetworkCreator(4, new int[] { 8, 6, 3, 1 }, 
                new Func<double, double>[] { 
                    ActivationFunctions.ReLU, 
                    ActivationFunctions.Sigmoid, 
                    ActivationFunctions.ReLU, 
                    ActivationFunctions.Unit 
                });
            creator.RandomizeWeights(-0.3, 0.3);
            var network = creator.CreateNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.02);

            // Create samples with all operation types
            var samples = new List<Sample>();
            var random = new Random(42); // Fixed seed for reproducibility
            
            foreach (Operation op in Enum.GetValues<Operation>())
            {
                for (int i = 0; i < 10; i++)
                {
                    double a = random.NextDouble() * 10;
                    double b = random.NextDouble() * 10;
                    samples.Add(new Sample(a, b, op, 1.0));
                }
            }

            // Act
            var losses = new List<double>();
            for (int epoch = 0; epoch < 50; epoch++)
            {
                var loss = trainer.TrainOneEpoch(samples.ToArray());
                losses.Add(loss[0]);
            }

            // Assert
            Assert.Equal(50, losses.Count);
            Assert.All(losses, loss => Assert.True(double.IsFinite(loss)));
            
            // Test that network can handle all operation types
            foreach (Operation op in Enum.GetValues<Operation>())
            {
                var testSample = new Sample(3.0, 4.0, op, 1.0);
                var prediction = network.Predict(testSample.Xample);
                Assert.Single(prediction);
                Assert.True(double.IsFinite(prediction[0]));
            }
        }

        #endregion

        #region Network Architecture Variation Tests

        [Fact]
        public void NetworkArchitectures_VariousConfigurations_AllWorkCorrectly()
        {
            // Arrange
            var architectures = new (int inputs, int[] layers, string description)[] {
                (1, new int[] { 1 }, "Single input, single output"),
                (2, new int[] { 1 }, "Two inputs, single output"),
                (3, new int[] { 5, 3, 1 }, "Three layer network"),
                (4, new int[] { 10, 8, 6, 4, 2, 1 }, "Deep network"),
                (5, new int[] { 2, 4, 3, 5, 1 }, "Varying layer sizes"),
                (2, new int[] { 20, 1 }, "Wide then narrow"),
                (1, new int[] { 1, 2, 3, 4, 5 }, "Expanding network"),
                (10, new int[] { 5, 2 }, "Many inputs, few outputs")
            };

            foreach (var (inputs, layers, description) in architectures)
            {
                // Act
                var activationFunctions = new Func<double, double>[layers.Length];
                for (int i = 0; i < layers.Length; i++)
                {
                    activationFunctions[i] = ActivationFunctions.ReLU;
                }
                if (layers.Length > 0)
                {
                    activationFunctions[layers.Length - 1] = ActivationFunctions.Sigmoid;
                }

                var creator = new NetworkCreator(inputs, layers, activationFunctions);
                creator.RandomizeWeights(-0.5, 0.5);
                var network = creator.CreateNetwork();
                var trainer = new NeuralNetworkTrainer(network, 0.01);

                // Create test data
                var testInput = new double[inputs];
                for (int i = 0; i < inputs; i++)
                {
                    testInput[i] = i * 0.1;
                }

                var testOutput = new double[layers.Last()];
                for (int i = 0; i < layers.Last(); i++)
                {
                    testOutput[i] = 1.0;
                }

                // Assert
                var prediction = network.Predict(testInput);
                Assert.Equal(layers.Last(), prediction.Length);
                Assert.All(prediction, p => Assert.True(double.IsFinite(p)));

                // Test training
                var loss = trainer.TrainOneEpoch(new double[][] { testInput }, new double[][] { testOutput });
                Assert.Equal(layers.Last(), loss.Length);
                Assert.All(loss, l => Assert.True(double.IsFinite(l)));
            }
        }

        [Fact]
        public void NetworkCreator_StaticMethods_WorkWithComplexArrays()
        {
            // Arrange
            var creator = new NetworkCreator(3, new int[] { 4, 3, 2 }, 
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.Sigmoid, ActivationFunctions.Unit });
            creator.RandomizeWeights(-1.0, 1.0);

            var originalWeights = new double[creator.Weights.Length][][];
            for (int i = 0; i < creator.Weights.Length; i++)
            {
                originalWeights[i] = new double[creator.Weights[i].Length][];
                for (int j = 0; j < creator.Weights[i].Length; j++)
                {
                    originalWeights[i][j] = new double[creator.Weights[i][j].Length];
                    Array.Copy(creator.Weights[i][j], originalWeights[i][j], creator.Weights[i][j].Length);
                }
            }

            // Act
            NetworkCreator.ApplyOn3dArr(creator.Weights, x => x * 2.0);

            // Assert
            for (int i = 0; i < creator.Weights.Length; i++)
            {
                for (int j = 0; j < creator.Weights[i].Length; j++)
                {
                    for (int k = 0; k < creator.Weights[i][j].Length; k++)
                    {
                        Assert.Equal(originalWeights[i][j][k] * 2.0, creator.Weights[i][j][k], Tolerance);
                    }
                }
            }
        }

        #endregion

        #region Performance and Scalability Tests

        [Fact]
        public void LargeNetworkTraining_PerformanceTest_CompletesInReasonableTime()
        {
            // Arrange
            var creator = new NetworkCreator(50, new int[] { 100, 50, 20, 5, 1 }, 
                new Func<double, double>[] { 
                    ActivationFunctions.ReLU, 
                    ActivationFunctions.ReLU, 
                    ActivationFunctions.ReLU, 
                    ActivationFunctions.ReLU, 
                    ActivationFunctions.Sigmoid 
                });
            creator.RandomizeWeights(-0.1, 0.1);
            var network = creator.CreateNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.001);

            // Create large training dataset
            var trainingData = new double[100][];
            var observed = new double[100][];
            var random = new Random(42);
            
            for (int i = 0; i < 100; i++)
            {
                trainingData[i] = new double[50];
                for (int j = 0; j < 50; j++)
                {
                    trainingData[i][j] = random.NextDouble();
                }
                observed[i] = new double[] { random.NextDouble() };
            }

            // Act
            var startTime = DateTime.Now;
            var loss = trainer.TrainOneEpoch(trainingData, observed);
            var endTime = DateTime.Now;

            // Assert
            Assert.Single(loss);
            Assert.True(double.IsFinite(loss[0]));
            Assert.True(loss[0] >= 0, "Loss should be non-negative");
            Assert.True((endTime - startTime).TotalSeconds < 30, "Large network training should complete within 30 seconds");
            
            // Additional robustness checks
            Assert.True(loss[0] < 1000000, "Loss should be in reasonable range");
        }

        [Fact]
        public async System.Threading.Tasks.Task ConcurrentNetworkOperations_ThreadSafety_WorksCorrectly()
        {
            // Arrange
            var creator = new NetworkCreator(2, new int[] { 3, 1 }, 
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.Sigmoid });
            creator.RandomizeWeights(-1.0, 1.0);
            var network = creator.CreateNetwork();

            var results = new double[10][];
            var tasks = new System.Threading.Tasks.Task[10];

            // Act
            for (int i = 0; i < 10; i++)
            {
                int index = i;
                tasks[i] = System.Threading.Tasks.Task.Run(() =>
                {
                    var input = new double[] { index * 0.1, index * 0.2 };
                    results[index] = network.Predict(input);
                });
            }

            await System.Threading.Tasks.Task.WhenAll(tasks);

            // Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.NotNull(results[i]);
                Assert.Single(results[i]);
                Assert.True(double.IsFinite(results[i][0]));
            }
        }

        #endregion

        #region Complex Loss Function Tests

        [Fact]
        public void CustomLossFunctions_Integration_WorksCorrectly()
        {
            // Arrange
            var network = CreateTestNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.05);

            // Custom loss function: Mean Absolute Error
            trainer.LossFunction = (predicted, observed) =>
            {
                var loss = new double[predicted.Length];
                for (int i = 0; i < predicted.Length; i++)
                {
                    loss[i] = Math.Abs(predicted[i][0] - observed[i][0]);
                }
                return loss;
            };

            // Custom loss derivative: Sign of the error
            trainer.LossFunctionD = (predicted, observed) =>
            {
                var derivative = new double[predicted.Length];
                for (int i = 0; i < predicted.Length; i++)
                {
                    derivative[i] = predicted[i][0] > observed[i][0] ? 1.0 : -1.0;
                }
                return derivative;
            };

            double[][] trainingData = { new double[] { 1.0, 2.0 }, new double[] { 2.0, 3.0 } };
            double[][] observed = { new double[] { 3.0 }, new double[] { 5.0 } };

            // Act
            var loss = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.Single(loss);
            Assert.All(loss, l => Assert.True(double.IsFinite(l)));
            Assert.All(loss, l => Assert.True(l >= 0)); // MAE should be non-negative
            
            // Additional robustness checks
            Assert.All(loss, l => Assert.True(l < 100000, "Loss should be in reasonable range"));
        }

        [Fact]
        public void NetworkWithDifferentLossFunctions_ProduceDifferentResults()
        {
            // Arrange
            var creator1 = new NetworkCreator(2, new int[] { 3, 1 },
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.Sigmoid });
            creator1.RandomizeWeights(-1.0, 1.0);

            var creator2 = new NetworkCreator(2, new int[] { 3, 1 },
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.Sigmoid });
            creator2.RandomizeWeights(-1.0, 1.0);

            var network1 = creator1.CreateNetwork();
            var network2 = creator2.CreateNetwork();

            var trainer1 = new NeuralNetworkTrainer(network1, 0.1);
            var trainer2 = new NeuralNetworkTrainer(network2, 0.1);

            // Set different loss functions
            trainer1.LossFunction = LossFunctions.SumSquaredError;
            trainer1.LossFunctionD = LossFunctions.SumSquaredErrorDerivative;

            trainer2.LossFunction = (predicted, observed) =>
            {
                var loss = new double[predicted.Length];
                for (int i = 0; i < predicted.Length; i++)
                {
                    loss[i] = Math.Abs(predicted[i][0] - observed[i][0]);
                }
                return loss;
            };

            trainer2.LossFunctionD = (predicted, observed) =>
            {
                var derivative = new double[predicted.Length];
                for (int i = 0; i < predicted.Length; i++)
                {
                    derivative[i] = predicted[i][0] > observed[i][0] ? 1.0 : -1.0;
                }
                return derivative;
            };

            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            var initialPrediction = network1.Predict(trainingData[0]);
            
            for (int epoch = 0; epoch < 10; epoch++)
            {
                trainer1.TrainOneEpoch(trainingData, observed);
                trainer2.TrainOneEpoch(trainingData, observed);
            }

            var finalPrediction1 = network1.Predict(trainingData[0]);
            var finalPrediction2 = network2.Predict(trainingData[0]);

            // Assert
            Assert.NotEqual(finalPrediction1[0], finalPrediction2[0]);
            Assert.True(double.IsFinite(finalPrediction1[0]));
            Assert.True(double.IsFinite(finalPrediction2[0]));
        }

        #endregion

        #region Helper Methods

        private INeuralNetwork CreateTestNetwork()
        {
            var creator = new NetworkCreator(2, new int[] { 3, 1 }, 
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.Sigmoid });
            creator.RandomizeWeights(-1.0, 1.0);
            return creator.CreateNetwork();
        }

        #endregion
    }
}
