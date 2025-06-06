using System;
using Xunit;
using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class SimpleNetworkCreatorTest
{
    [Fact]
    public void NetworkCreator_CanBeInstantiated()
    {
        // Arrange
        int[] layerSizes = [2, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.Unit
        ];

        // Act
        var creator = new NetworkCreator(layerSizes, activationFunctions);

        // Assert
        Assert.NotNull(creator);
    }
}
