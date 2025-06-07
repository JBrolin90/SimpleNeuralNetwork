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
            var nodeSteps = new NodeSteps(weightCount);

            // Assert
            Assert.Equal(weightCount, nodeSteps.WeightSteps.Length);
            Assert.All(nodeSteps.WeightSteps, step => Assert.Equal(0.0, step));
        }

        [Fact]
        public void NodeSteps_Constructor_InitializesBiasStepToZero()
        {
            // Arrange
            int weightCount = 5;

            // Act
            var nodeSteps = new NodeSteps(weightCount);

            // Assert
            Assert.Equal(0.0, nodeSteps.BiasStep);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(5)]
        [InlineData(10)]
        public void NodeSteps_Constructor_WithVariousWeightCounts_CreatesCorrectArraySize(int weightCount)
        {
            // Act
            var nodeSteps = new NodeSteps(weightCount);

            // Assert
            Assert.Equal(weightCount, nodeSteps.WeightSteps.Length);
        }

        [Fact]
        public void NodeSteps_WeightSteps_CanBeModified()
        {
            // Arrange
            var nodeSteps = new NodeSteps(3);
            double[] newWeightSteps = { 0.1, 0.2, 0.3 };

            // Act
            for (int i = 0; i < newWeightSteps.Length; i++)
            {
                nodeSteps.WeightSteps[i] = newWeightSteps[i];
            }

            // Assert
            for (int i = 0; i < newWeightSteps.Length; i++)
            {
                Assert.Equal(newWeightSteps[i], nodeSteps.WeightSteps[i], 7);
            }
        }

        [Fact]
        public void NodeSteps_BiasStep_CanBeModified()
        {
            // Arrange
            var nodeSteps = new NodeSteps(2);
            double newBiasStep = 0.5;

            // Act
            nodeSteps.BiasStep = newBiasStep;

            // Assert
            Assert.Equal(newBiasStep, nodeSteps.BiasStep, 7);
        }

        [Fact]
        public void NodeSteps_WeightSteps_AccumulatesValues()
        {
            // Arrange
            var nodeSteps = new NodeSteps(2);

            // Act
            nodeSteps.WeightSteps[0] += 0.1;
            nodeSteps.WeightSteps[0] += 0.2;
            nodeSteps.WeightSteps[1] += 0.3;

            // Assert
            Assert.Equal(0.3, nodeSteps.WeightSteps[0], 7);
            Assert.Equal(0.3, nodeSteps.WeightSteps[1], 7);
        }

        [Fact]
        public void NodeSteps_BiasStep_AccumulatesValues()
        {
            // Arrange
            var nodeSteps = new NodeSteps(1);

            // Act
            nodeSteps.BiasStep += 0.1;
            nodeSteps.BiasStep += 0.2;

            // Assert
            Assert.Equal(0.3, nodeSteps.BiasStep, 7);
        }

        [Fact]
        public void NodeSteps_Constructor_WithZeroWeights_CreatesEmptyArray()
        {
            // Act
            var nodeSteps = new NodeSteps(0);

            // Assert
            Assert.Empty(nodeSteps.WeightSteps);
            Assert.Equal(0.0, nodeSteps.BiasStep);
        }

        [Fact]
        public void NodeSteps_WeightSteps_SupportsNegativeValues()
        {
            // Arrange
            var nodeSteps = new NodeSteps(2);

            // Act
            nodeSteps.WeightSteps[0] = -0.5;
            nodeSteps.WeightSteps[1] = -1.2;

            // Assert
            Assert.Equal(-0.5, nodeSteps.WeightSteps[0], 7);
            Assert.Equal(-1.2, nodeSteps.WeightSteps[1], 7);
        }

        [Fact]
        public void NodeSteps_BiasStep_SupportsNegativeValues()
        {
            // Arrange
            var nodeSteps = new NodeSteps(1);

            // Act
            nodeSteps.BiasStep = -0.8;

            // Assert
            Assert.Equal(-0.8, nodeSteps.BiasStep, 7);
        }

        [Fact]
        public void NodeSteps_WeightSteps_ArrayReference_IsIndependent()
        {
            // Arrange
            var nodeSteps1 = new NodeSteps(2);
            var nodeSteps2 = new NodeSteps(2);

            // Act
            nodeSteps1.WeightSteps[0] = 1.0;
            nodeSteps2.WeightSteps[0] = 2.0;

            // Assert
            Assert.Equal(1.0, nodeSteps1.WeightSteps[0], 7);
            Assert.Equal(2.0, nodeSteps2.WeightSteps[0], 7);
            Assert.NotSame(nodeSteps1.WeightSteps, nodeSteps2.WeightSteps);
        }

        [Fact]
        public void NodeSteps_MultiplePensionsteps_IndependentBiasSteps()
        {
            // Arrange
            var nodeSteps1 = new NodeSteps(1);
            var nodeSteps2 = new NodeSteps(1);

            // Act
            nodeSteps1.BiasStep = 1.5;
            nodeSteps2.BiasStep = 2.5;

            // Assert
            Assert.Equal(1.5, nodeSteps1.BiasStep, 7);
            Assert.Equal(2.5, nodeSteps2.BiasStep, 7);
        }
    }
}
