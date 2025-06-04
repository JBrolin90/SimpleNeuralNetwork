using Xunit;
using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class NodeStepsTests
{
    [Fact]
    public void Constructor_ShouldInitializeWeightSteps()
    {
        // Arrange
        int weightCount = 3;

        // Act
        var nodeSteps = new NodeSteps(weightCount);

        // Assert
        Assert.NotNull(nodeSteps.WeightSteps);
        Assert.Equal(weightCount, nodeSteps.WeightSteps.Length);
        Assert.Equal(0, nodeSteps.BiasStep);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    public void Constructor_WithDifferentWeightCounts_ShouldCreateCorrectArray(int weightCount)
    {
        // Act
        var nodeSteps = new NodeSteps(weightCount);

        // Assert
        Assert.Equal(weightCount, nodeSteps.WeightSteps.Length);
    }

    [Fact]
    public void WeightSteps_ShouldBeInitializedToZero()
    {
        // Arrange
        var nodeSteps = new NodeSteps(3);

        // Act & Assert
        for (int i = 0; i < nodeSteps.WeightSteps.Length; i++)
        {
            Assert.Equal(0, nodeSteps.WeightSteps[i]);
        }
    }
}
