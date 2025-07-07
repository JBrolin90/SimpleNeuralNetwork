using Xunit;
using System;
using BackPropagation.NNLib;
using System.Linq;

namespace SimpleNeuralNetwork.Tests
{
    public class ErrorHandlingAndBoundaryTests
    {
        private const double Tolerance = 1e-7;

        #region Neural Network Boundary Tests

        [Fact]
        public void NeuralNetwork_Predict_WithNullInput_ThrowsException()
        {
            // Arrange
            var network = CreateSimpleNetwork();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => network.Predict(null!));
        }

        [Fact]
        public void NeuralNetwork_Predict_WithEmptyInput_ReturnsValidOutput()
        {
            // Arrange
            var network = CreateNetworkWithZeroInputs();

            // Act
            var result = network.Predict(Array.Empty<double>());

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.True(double.IsFinite(result[0]));
        }

        [Fact]
        public void NeuralNetwork_Predict_WithWrongInputSize_ThrowsException()
        {
            // Arrange
            var network = CreateSimpleNetwork(); // Expects 2 inputs

            // Act & Assert
            Assert.Throws<IndexOutOfRangeException>(() => network.Predict(new double[] { 1.0 })); // Only 1 input
            Assert.Throws<IndexOutOfRangeException>(() => network.Predict(new double[] { 1.0, 2.0, 3.0 })); // 3 inputs
        }

        [Fact]
        public void NeuralNetwork_Predict_WithInfiniteInputs_HandlesGracefully()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            double[] inputs = { double.PositiveInfinity, double.NegativeInfinity };

            // Act
            var result = network.Predict(inputs);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            // Result might be infinite, but should not be NaN
            Assert.False(double.IsNaN(result[0]));
        }

        [Fact]
        public void NeuralNetwork_Predict_WithNaNInputs_ReturnsNaN()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            double[] inputs = { double.NaN, 1.0 };

            // Act
            var result = network.Predict(inputs);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.True(double.IsNaN(result[0]));
        }

        [Fact]
        public void NeuralNetwork_Predict_WithExtremeValues_HandlesCorrectly()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            double[] inputs = { double.MaxValue, double.MinValue };

            // Act
            var result = network.Predict(inputs);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.True(double.IsFinite(result[0]) || double.IsInfinity(result[0]));
        }

        [Fact]
        public void NeuralNetwork_Predict_WithVerySmallValues_HandlesCorrectly()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            double[] inputs = { double.Epsilon, -double.Epsilon };

            // Act
            var result = network.Predict(inputs);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.True(double.IsFinite(result[0]));
        }

        #endregion

        #region Training Boundary Tests

        [Fact]
        public void NeuralNetworkTrainer_TrainOneEpoch_WithInfiniteObservedValues_HandlesGracefully()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { double.PositiveInfinity } };

            // Act
            var result = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            // Loss might be infinite, but should not be NaN
            Assert.False(double.IsNaN(result[0]));
        }

        [Fact]
        public void NeuralNetworkTrainer_TrainOneEpoch_WithNaNObservedValues_ReturnsNaN()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { double.NaN } };

            // Act
            var result = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.True(double.IsNaN(result[0]));
        }

        [Fact]
        public void NeuralNetworkTrainer_TrainOneEpoch_WithExtremelyHighLearningRate_HandlesCorrectly()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            var trainer = new NeuralNetworkTrainer(network, 1000.0); // Very high learning rate
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            var result = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.True(double.IsFinite(result[0]) || double.IsInfinity(result[0]));
        }

        [Fact]
        public void NeuralNetworkTrainer_TrainOneEpoch_WithVerySmallLearningRate_HandlesCorrectly()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            var trainer = new NeuralNetworkTrainer(network, double.Epsilon); // Very small learning rate
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            var result = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.True(double.IsFinite(result[0]));
        }

        [Fact]
        public void NeuralNetworkTrainer_TrainOneEpoch_WithInfiniteLearningRate_HandlesCorrectly()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            var trainer = new NeuralNetworkTrainer(network, double.PositiveInfinity);
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            var result = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            // Result might be infinite or NaN, but should not crash
            Assert.True(double.IsFinite(result[0]) || double.IsInfinity(result[0]) || double.IsNaN(result[0]));
        }

        [Fact]
        public void NeuralNetworkTrainer_TrainOneEpoch_WithNaNLearningRate_HandlesCorrectly()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            var trainer = new NeuralNetworkTrainer(network, double.NaN);
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            var result = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            // With NaN learning rate, loss should be NaN
            Assert.True(double.IsNaN(result[0]));
        }

        #endregion

        #region Activation Function Boundary Tests

        [Fact]
        public void ActivationFunctions_Sigmoid_WithExtremeValues_HandlesCorrectly()
        {
            // Arrange & Act
            var sigmoidMax = ActivationFunctions.Sigmoid(double.MaxValue);
            var sigmoidMin = ActivationFunctions.Sigmoid(double.MinValue);
            var sigmoidPosInf = ActivationFunctions.Sigmoid(double.PositiveInfinity);
            var sigmoidNegInf = ActivationFunctions.Sigmoid(double.NegativeInfinity);
            var sigmoidNaN = ActivationFunctions.Sigmoid(double.NaN);

            // Assert
            Assert.Equal(1.0, sigmoidMax, Tolerance);
            Assert.Equal(0.0, sigmoidMin, Tolerance);
            Assert.Equal(1.0, sigmoidPosInf, Tolerance);
            Assert.Equal(0.0, sigmoidNegInf, Tolerance);
            Assert.True(double.IsNaN(sigmoidNaN));
        }

        [Fact]
        public void ActivationFunctions_ReLU_WithExtremeValues_HandlesCorrectly()
        {
            // Arrange & Act
            var reluMax = ActivationFunctions.ReLU(double.MaxValue);
            var reluMin = ActivationFunctions.ReLU(double.MinValue);
            var reluPosInf = ActivationFunctions.ReLU(double.PositiveInfinity);
            var reluNegInf = ActivationFunctions.ReLU(double.NegativeInfinity);
            var reluNaN = ActivationFunctions.ReLU(double.NaN);

            // Assert
            Assert.Equal(double.MaxValue, reluMax);
            Assert.Equal(0.0, reluMin);
            Assert.Equal(double.PositiveInfinity, reluPosInf);
            Assert.Equal(0.0, reluNegInf);
            Assert.True(double.IsNaN(reluNaN));
        }

        [Fact]
        public void ActivationFunctions_Tanh_WithExtremeValues_HandlesCorrectly()
        {
            // Arrange & Act
            var tanhMax = ActivationFunctions.Tanh(double.MaxValue);
            var tanhMin = ActivationFunctions.Tanh(double.MinValue);
            var tanhPosInf = ActivationFunctions.Tanh(double.PositiveInfinity);
            var tanhNegInf = ActivationFunctions.Tanh(double.NegativeInfinity);
            var tanhNaN = ActivationFunctions.Tanh(double.NaN);

            // Assert
            Assert.Equal(1.0, tanhMax, Tolerance);
            Assert.Equal(-1.0, tanhMin, Tolerance);
            Assert.Equal(1.0, tanhPosInf, Tolerance);
            Assert.Equal(-1.0, tanhNegInf, Tolerance);
            Assert.True(double.IsNaN(tanhNaN));
        }

        [Fact]
        public void ActivationFunctions_Unit_WithExtremeValues_ReturnsInput()
        {
            // Arrange & Act
            var unitMax = ActivationFunctions.Unit(double.MaxValue);
            var unitMin = ActivationFunctions.Unit(double.MinValue);
            var unitPosInf = ActivationFunctions.Unit(double.PositiveInfinity);
            var unitNegInf = ActivationFunctions.Unit(double.NegativeInfinity);
            var unitNaN = ActivationFunctions.Unit(double.NaN);

            // Assert
            Assert.Equal(double.MaxValue, unitMax);
            Assert.Equal(double.MinValue, unitMin);
            Assert.Equal(double.PositiveInfinity, unitPosInf);
            Assert.Equal(double.NegativeInfinity, unitNegInf);
            Assert.True(double.IsNaN(unitNaN));
        }

        #endregion

        #region Loss Function Boundary Tests

        [Fact]
        public void LossFunctions_SquaredError_WithExtremeValues_HandlesCorrectly()
        {
            // Arrange
            double[] prediction = { double.MaxValue };
            double[] observed = { double.MinValue };

            // Act
            var loss = LossFunctions.SquaredError(prediction, observed);

            // Assert
            Assert.Single(loss);
            Assert.True(double.IsPositiveInfinity(loss[0]));
        }

        [Fact]
        public void LossFunctions_SquaredError_WithNaNValues_ReturnsNaN()
        {
            // Arrange
            double[] prediction = { double.NaN };
            double[] observed = { 1.0 };

            // Act
            var loss = LossFunctions.SquaredError(prediction, observed);

            // Assert
            Assert.Single(loss);
            Assert.True(double.IsNaN(loss[0]));
        }

        [Fact]
        public void LossFunctions_SquaredError_WithInfiniteValues_HandlesCorrectly()
        {
            // Arrange
            double[] prediction = { double.PositiveInfinity };
            double[] observed = { double.PositiveInfinity };

            // Act
            var loss = LossFunctions.SquaredError(prediction, observed);

            // Assert
            Assert.Single(loss);
            Assert.True(double.IsNaN(loss[0])); // Infinity - Infinity = NaN
        }

        [Fact]
        public void LossFunctions_SumSquaredError_WithMultipleExtremeValues_HandlesCorrectly()
        {
            // Arrange
            double[][] predictions = { new double[] { double.MaxValue, double.MinValue } };
            double[][] observed = { new double[] { double.MinValue, double.MaxValue } };

            // Act
            var loss = LossFunctions.SumSquaredError(predictions, observed);

            // Assert
            Assert.Equal(2, loss.Length);
            Assert.True(double.IsPositiveInfinity(loss[0]));
            Assert.True(double.IsPositiveInfinity(loss[1]));
        }

        [Fact]
        public void LossFunctions_SumSquaredErrorDerivative_WithExtremeValues_HandlesCorrectly()
        {
            // Arrange
            double[][] predictions = { new double[] { double.MaxValue } };
            double[][] observed = { new double[] { double.MinValue } };

            // Act
            var derivative = LossFunctions.SumSquaredErrorDerivative(predictions, observed);

            // Assert
            Assert.Single(derivative);
            Assert.True(double.IsPositiveInfinity(derivative[0]));
        }

        #endregion

        #region InputProcessor Boundary Tests

        [Fact]
        public void InputProcessor_ProcessInputs_WithNullInputs_ThrowsException()
        {
            // Arrange
            var mockLayer = new MockLayer();
            var inputProcessor = new InputProcessor(mockLayer, 0, new double[] { 1.0, 2.0 }, new double[] { 0.5 });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => inputProcessor.ProcessInputs(null!));
        }

        [Fact]
        public void InputProcessor_ProcessInputs_WithWrongInputSize_ThrowsException()
        {
            // Arrange
            var mockLayer = new MockLayer();
            var inputProcessor = new InputProcessor(mockLayer, 0, new double[] { 1.0, 2.0 }, new double[] { 0.5 }); // Expects 2 inputs

            // Act & Assert
            Assert.Throws<IndexOutOfRangeException>(() => inputProcessor.ProcessInputs(new double[] { 1.0 })); // Only 1 input
            Assert.Throws<IndexOutOfRangeException>(() => inputProcessor.ProcessInputs(new double[] { 1.0, 2.0, 3.0 })); // 3 inputs
        }

        [Fact]
        public void InputProcessor_ProcessInputs_WithInfiniteInputs_HandlesCorrectly()
        {
            // Arrange
            var mockLayer = new MockLayer();
            var inputProcessor = new InputProcessor(mockLayer, 0, new double[] { 1.0, 1.0 }, new double[] { 0.0 });

            // Act
            inputProcessor.ProcessInputs(new double[] { double.PositiveInfinity, double.NegativeInfinity });

            // Assert
            Assert.True(double.IsNaN(inputProcessor.Y)); // Infinity + (-Infinity) = NaN
        }

        [Fact]
        public void InputProcessor_ProcessInputs_WithNaNInputs_ReturnsNaN()
        {
            // Arrange
            var mockLayer = new MockLayer();
            var inputProcessor = new InputProcessor(mockLayer, 0, new double[] { 1.0, 1.0 }, new double[] { 0.0 });

            // Act
            inputProcessor.ProcessInputs(new double[] { double.NaN, 1.0 });

            // Assert
            Assert.True(double.IsNaN(inputProcessor.Y));
        }

        [Fact]
        public void InputProcessor_ProcessInputs_WithExtremeWeights_HandlesCorrectly()
        {
            // Arrange
            var mockLayer = new MockLayer();
            // Use infinity values to actually produce NaN when combined
            var inputProcessor = new InputProcessor(mockLayer, 0, new double[] { double.PositiveInfinity, double.NegativeInfinity }, new double[] { 0.0 });

            // Act
            inputProcessor.ProcessInputs(new double[] { 1.0, 1.0 });

            // Assert
            // PositiveInfinity + NegativeInfinity = NaN
            Assert.True(double.IsNaN(inputProcessor.Y));
        }

        #endregion

        #region Memory Stress Tests

        [Fact]
        public void NeuralNetwork_VeryLargeNetwork_DoesNotCrash()
        {
            // Arrange
            var creator = new NetworkCreator(100, new int[] { 500, 200, 50, 1 }, 
                new Func<double, double>[] { ActivationFunctions.ReLU, ActivationFunctions.ReLU, ActivationFunctions.ReLU, ActivationFunctions.Sigmoid });
            creator.RandomizeWeights(-0.1, 0.1);

            // Act
            var network = creator.CreateNetwork();
            var inputs = new double[100];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = i * 0.01;
            }

            // This should not crash or throw an exception
            var result = network.Predict(inputs);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.True(double.IsFinite(result[0]));
        }

        [Fact]
        public void NeuralNetworkTrainer_ExtensiveTraining_DoesNotCrash()
        {
            // Arrange
            var network = CreateSimpleNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act - Train for many epochs
            for (int epoch = 0; epoch < 1000; epoch++)
            {
                var loss = trainer.TrainOneEpoch(trainingData, observed);
                Assert.NotNull(loss);
                Assert.Single(loss);
                Assert.True(double.IsFinite(loss[0]));
            }

            // Assert - Network should still be functional
            var prediction = network.Predict(trainingData[0]);
            Assert.NotNull(prediction);
            Assert.Single(prediction);
            Assert.True(double.IsFinite(prediction[0]));
        }

        #endregion

        #region Helper Methods

        private INeuralNetwork CreateSimpleNetwork()
        {
            var creator = new NetworkCreator(2, new int[] { 2, 1 }, 
                new Func<double, double>[] { ActivationFunctions.Sigmoid, ActivationFunctions.Sigmoid });
            creator.RandomizeWeights(-1.0, 1.0);
            return creator.CreateNetwork();
        }

        private INeuralNetwork CreateNetworkWithZeroInputs()
        {
            var creator = new NetworkCreator(0, new int[] { 1 }, 
                new Func<double, double>[] { ActivationFunctions.Sigmoid });
            creator.RandomizeWeights(-1.0, 1.0);
            return creator.CreateNetwork();
        }

        // Mock layer for testing
        private class MockLayer : ILayer
        {
            public int Index { get; set; }
            public INeuron[] Neurons { get; set; } = Array.Empty<INeuron>();
            public IInputProcessor[] InputProcessors { get; set; } = Array.Empty<IInputProcessor>();
            public double[] Ys { get; set; } = Array.Empty<double>();
            public double[][] Weights { get; set; } = Array.Empty<double[]>();
            public double[][] Biases { get; set; } = Array.Empty<double[]>();
            public double[]? Inputs { get; set; }
            public ILayer? PreviousLayer { get; set; }
            public ILayer? NextLayer { get; set; }
            public Func<double, double>? ActivationFunction { get; set; }

            public double[] ProcessInputs(double[] inputs)
            {
                return inputs;
            }

            public void SetInputs(double[] inputs)
            {
                Inputs = inputs;
            }

            public double[] Forward(double[] inputs)
            {
                return inputs;
            }
        }

        #endregion
    }
}
