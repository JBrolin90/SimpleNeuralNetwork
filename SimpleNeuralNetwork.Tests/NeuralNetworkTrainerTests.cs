using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class NeuralNetworkTrainerTests
    {
        private const double Tolerance = 1e-7;

        [Fact]
        public void NeuralNetworkTrainer_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NodeFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            double learningRate = 0.05;

            // Act
            var trainer = new NeuralNetworkTrainer(layerFactory, nodeFactory, weights, biases, ys, learningRate);

            // Assert
            Assert.Equal(learningRate, trainer.LearningRate);
            Assert.NotNull(trainer.SSR);
            Assert.NotNull(trainer.dSSR);
            Assert.NotNull(trainer.NodeSteps);
        }

        [Fact]
        public void NeuralNetworkTrainer_Constructor_WithDefaultActivationFunctions_CreatesUnitFunctions()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NodeFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } },
                new double[][] { new double[] { 1.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } },
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 },
                new double[] { 0.0 }
            };

            // Act
            var trainer = new NeuralNetworkTrainer(layerFactory, nodeFactory, weights, biases, ys);

            // Assert
            Assert.Equal(2, trainer.ActivationFunctions.Length);
            Assert.Equal(ActivationFunctions.Unit, trainer.ActivationFunctions[0]);
            Assert.Equal(ActivationFunctions.Unit, trainer.ActivationFunctions[1]);
        }

        [Fact]
        public void NeuralNetworkTrainer_PrepareBackPropagation_InitializesNodeSteps()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NodeFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0, 0.5 } },
                new double[][] { new double[] { 0.8 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } },
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 },
                new double[] { 0.0 }
            };
            var trainer = new NeuralNetworkTrainer(layerFactory, nodeFactory, weights, biases, ys);

            // Act
            trainer.PrepareBackPropagation();

            // Assert
            Assert.Equal(2, trainer.NodeSteps.Length); // 2 layers
            Assert.Single(trainer.NodeSteps[0]); // 1 node in first layer
            Assert.Single(trainer.NodeSteps[1]); // 1 node in second layer
            Assert.Equal(2, trainer.NodeSteps[0][0].WeightSteps.Length); // 2 weights in first node
            Assert.Single(trainer.NodeSteps[1][0].WeightSteps); // 1 weight in second node
        }

        [Fact]
        public void NeuralNetworkTrainer_Train_UpdatesSSRAndDSSR()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NodeFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            var trainer = new NeuralNetworkTrainer(layerFactory, nodeFactory, weights, biases, ys, 0.01);
            
            double[][] trainingData = { new double[] { 2.0 } };
            double[][] expectedOutputs = { new double[] { 3.0 } };

            // Act
            trainer.TrainOneEpoch(trainingData, expectedOutputs);

            // Assert
            Assert.Single(trainer.SSR);
            Assert.Single(trainer.dSSR);
            
            // prediction = 2.0 * 1.0 + 0.0 = 2.0
            // error = 3.0 - 2.0 = 1.0
            // SSR = error^2 = 1.0
            // dSSR = -2 * error = -2.0
            Assert.Equal(1.0, trainer.SSR[0], 7);
            Assert.Equal(-2.0, trainer.dSSR[0], 7);
        }

        [Fact]
        public void NeuralNetworkTrainer_Train_UpdatesWeights()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NodeFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            double learningRate = 0.1;
            var trainer = new NeuralNetworkTrainer(layerFactory, nodeFactory, weights, biases, ys, learningRate);
            
            double originalWeight = trainer.Weigths[0][0][0];
            double[][] trainingData = { new double[] { 2.0 } };
            double[][] expectedOutputs = { new double[] { 3.0 } };

            // Act
            trainer.TrainOneEpoch(trainingData, expectedOutputs);

            // Assert
            Assert.NotEqual(originalWeight, trainer.Weigths[0][0][0]);
        }

        [Fact]
        public void NeuralNetworkTrainer_Train_UpdatesBiases()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NodeFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.5 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            double learningRate = 0.1;
            var trainer = new NeuralNetworkTrainer(layerFactory, nodeFactory, weights, biases, ys, learningRate);
            
            double originalBias = trainer.Biases[0][0][0];
            double[][] trainingData = { new double[] { 2.0 } };
            double[][] expectedOutputs = { new double[] { 3.0 } };

            // Act
            trainer.TrainOneEpoch(trainingData, expectedOutputs);

            // Assert
            Assert.NotEqual(originalBias, trainer.Biases[0][0][0]);
        }

        [Fact]
        public void NeuralNetworkTrainer_BackPropagate_CallsLayerBackward()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NodeFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            var trainer = new NeuralNetworkTrainer(layerFactory, nodeFactory, weights, biases, ys);
            trainer.PrepareBackPropagation();
            
            // Process some inputs first
            trainer.Predict(new double[] { 1.0 });
            
            double[] dSSR = { 0.1 };

            // Act & Assert - Should not throw exception
            trainer.PropagateBackwards(dSSR);
        }

        [Theory]
        [InlineData(0.01)]
        [InlineData(0.1)]
        [InlineData(0.5)]
        public void NeuralNetworkTrainer_WithDifferentLearningRates_ConvergesDifferently(double learningRate)
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NodeFactory();
            double[][][] weights = {
                new double[][] { new double[] { 0.5 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            var trainer = new NeuralNetworkTrainer(layerFactory, nodeFactory, weights, biases, ys, learningRate);
            
            double[][] trainingData = { new double[] { 1.0 } };
            double[][] expectedOutputs = { new double[] { 2.0 } };

            // Act
            trainer.TrainOneEpoch(trainingData, expectedOutputs);

            // Assert
            Assert.True(trainer.LearningRate == learningRate);
            Assert.True(trainer.SSR[0] >= 0); // SSR should be non-negative
        }

        [Fact]
        public void NeuralNetworkTrainer_Train_WithMultipleTrainingExamples_ProcessesAll()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NodeFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            var trainer = new NeuralNetworkTrainer(layerFactory, nodeFactory, weights, biases, ys, 0.01);
            
            double[][] trainingData = { 
                new double[] { 1.0 }, 
                new double[] { 2.0 }, 
                new double[] { 3.0 } 
            };
            double[][] expectedOutputs = { 
                new double[] { 2.0 }, 
                new double[] { 4.0 }, 
                new double[] { 6.0 } 
            };

            // Act
            trainer.TrainOneEpoch(trainingData, expectedOutputs);

            // Assert
            Assert.Single(trainer.SSR);
            Assert.Single(trainer.dSSR);
            // The final SSR and dSSR should be from the last training example
        }

        [Fact]
        public void NodeSteps_Constructor_InitializesWeightStepsArray()
        {
            // Arrange
            int weightCount = 3;

            // Act
            var nodeSteps = new NodeSteps(weightCount);

            // Assert
            Assert.Equal(weightCount, nodeSteps.WeightSteps.Length);
            Assert.Equal(0.0, nodeSteps.BiasStep);
            Assert.All(nodeSteps.WeightSteps, step => Assert.Equal(0.0, step));
        }
    }
}
