using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class UnitTest1
{
    [Fact]
    public void TestProjectWorks()
    {
        // Simple test to verify the test framework is working
        Assert.True(true);
    }

    [Fact]
    public void CanAccessMainProject()
    {
        // Test that we can access classes from the main project
        var activationFunction = ActivationFunctions.Unit(0.5);
        Assert.Equal(0.5, activationFunction);
    }
}
