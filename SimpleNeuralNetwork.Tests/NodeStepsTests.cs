using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class NodeStepsTests
    {
        private const double Tolerance = 1e-7;

        [Fact]
        public void NodeSteps_Constructor_InitializesWeightStepsArray()
        {
            // Arrange
            int weightCount = 3;

            // Act
            var nodeSteps = new Gradients(weightCount);

            // Assert
            Assert.Equal(weightCount, nodeSteps.WeightGradient.Length);
            Assert.All(nodeSteps.WeightGradient, step => Assert.Equal(0.0, step));
        }

        [Fact]
        public void NodeSteps_Constructor_InitializesBiasStepToZero()
        {
            // Arrange
            int weightCount = 5;

            // Act
            var nodeSteps = new Gradients(weightCount);

            // Assert
            Assert.Equal(0.0, nodeSteps.BiasGradient);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(5)]
        [InlineData(10)]
        public void NodeSteps_Constructor_WithVariousWeightCounts_CreatesCorrectArraySize(int weightCount)
        {
            // Act
            var nodeSteps = new Gradients(weightCount);

            // Assert
            Assert.Equal(weightCount, nodeSteps.WeightGradient.Length);
        }

        [Fact]
        public void NodeSteps_WeightSteps_CanBeModified()
        {
            // Arrange
            var nodeSteps = new Gradients(3);
            double[] newWeightSteps = { 0.1, 0.2, 0.3 };

            // Act
            for (int i = 0; i < newWeightSteps.Length; i++)
            {
                nodeSteps.WeightGradient[i] = newWeightSteps[i];
            }

            // Assert
            for (int i = 0; i < newWeightSteps.Length; i++)
            {
                Assert.Equal(newWeightSteps[i], nodeSteps.WeightGradient[i], 7);
            }
        }

        [Fact]
        public void NodeSteps_BiasStep_CanBeModified()
        {
            // Arrange
            var nodeSteps = new Gradients(2);
            double newBiasStep = 0.5;

            // Act
            nodeSteps.BiasGradient = newBiasStep;

            // Assert
            Assert.Equal(newBiasStep, nodeSteps.BiasGradient, 7);
        }

        [Fact]
        public void NodeSteps_WeightSteps_AccumulatesValues()
        {
            // Arrange
            var nodeSteps = new Gradients(2);

            // Act
            nodeSteps.WeightGradient[0] += 0.1;
            nodeSteps.WeightGradient[0] += 0.2;
            nodeSteps.WeightGradient[1] += 0.3;

            // Assert
            Assert.Equal(0.3, nodeSteps.WeightGradient[0], 7);
            Assert.Equal(0.3, nodeSteps.WeightGradient[1], 7);
        }

        [Fact]
        public void NodeSteps_BiasStep_AccumulatesValues()
        {
            // Arrange
            var nodeSteps = new Gradients(1);

            // Act
            nodeSteps.BiasGradient += 0.1;
            nodeSteps.BiasGradient += 0.2;

            // Assert
            Assert.Equal(0.3, nodeSteps.BiasGradient, 7);
        }

        [Fact]
        public void NodeSteps_Constructor_WithZeroWeights_CreatesEmptyArray()
        {
            // Act
            var nodeSteps = new Gradients(0);

            // Assert
            Assert.Empty(nodeSteps.WeightGradient);
            Assert.Equal(0.0, nodeSteps.BiasGradient);
        }

        [Fact]
        public void NodeSteps_WeightSteps_SupportsNegativeValues()
        {
            // Arrange
            var nodeSteps = new Gradients(2);

            // Act
            nodeSteps.WeightGradient[0] = -0.5;
            nodeSteps.WeightGradient[1] = -1.2;

            // Assert
            Assert.Equal(-0.5, nodeSteps.WeightGradient[0], 7);
            Assert.Equal(-1.2, nodeSteps.WeightGradient[1], 7);
        }

        [Fact]
        public void NodeSteps_BiasStep_SupportsNegativeValues()
        {
            // Arrange
            var nodeSteps = new Gradients(1);

            // Act
            nodeSteps.BiasGradient = -0.8;

            // Assert
            Assert.Equal(-0.8, nodeSteps.BiasGradient, 7);
        }

        [Fact]
        public void NodeSteps_WeightSteps_ArrayReference_IsIndependent()
        {
            // Arrange
            var nodeSteps1 = new Gradients(2);
            var nodeSteps2 = new Gradients(2);

            // Act
            nodeSteps1.WeightGradient[0] = 1.0;
            nodeSteps2.WeightGradient[0] = 2.0;

            // Assert
            Assert.Equal(1.0, nodeSteps1.WeightGradient[0], 7);
            Assert.Equal(2.0, nodeSteps2.WeightGradient[0], 7);
            Assert.NotSame(nodeSteps1.WeightGradient, nodeSteps2.WeightGradient);
        }

        [Fact]
        public void NodeSteps_MultiplePensionsteps_IndependentBiasSteps()
        {
            // Arrange
            var nodeSteps1 = new Gradients(1);
            var nodeSteps2 = new Gradients(1);

            // Act
            nodeSteps1.BiasGradient = 1.5;
            nodeSteps2.BiasGradient = 2.5;

            // Assert
            Assert.Equal(1.5, nodeSteps1.BiasGradient, 7);
            Assert.Equal(2.5, nodeSteps2.BiasGradient, 7);
        }
    }
}
