using Xunit;
using System;
using System.Linq;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class NeuronTests
    {
        private const double Tolerance = 1e-7;

        #region Constructor Tests

        [Fact]
        public void Node_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            var layer = new MockLayer();
            int index = 1;
            
            // Act
            var node = new Neuron(layer, index, ActivationFunctions.Sigmoid);
            
            // Assert
            Assert.Equal(layer, node.Layer);
            Assert.Equal(index, node.Index);
            Assert.Equal(ActivationFunctions.Sigmoid, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.SigmoidDerivative, node.ActivationDerivative);
            Assert.Equal(0.0, node.Y);
            Assert.Equal(0.0, node.X);
        }

        [Fact]
        public void Node_Constructor_WithNullLayer_ThrowsException()
        {
            // Arrange
            
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new Neuron(null!, 0, ActivationFunctions.Unit));
        }

        [Fact]
        public void Node_Constructor_WithNullActivationFunction_UsesDefaultSoftPlus()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            var node = new Neuron(layer, 0, null!);
            
            // Assert
            Assert.Equal(ActivationFunctions.SoftPlus, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.SoftPlusDerivative, node.ActivationDerivative);
        }

        [Fact]
        public void Node_Constructor_WithUnitActivation_SetsCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            var node = new Neuron(layer, 0, ActivationFunctions.Unit);
            
            // Assert
            Assert.Equal(ActivationFunctions.Unit, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.UnitDerivative, node.ActivationDerivative);
        }

        [Fact]
        public void Node_Constructor_WithSoftPlusActivation_SetsCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            var node = new Neuron(layer, 0, ActivationFunctions.SoftPlus);
            
            // Assert
            Assert.Equal(ActivationFunctions.SoftPlus, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.SoftPlusDerivative, node.ActivationDerivative);
        }

        [Fact]
        public void Node_Constructor_WithSigmoidActivation_SetsCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            var node = new Neuron(layer, 0, ActivationFunctions.Sigmoid);
            
            // Assert
            Assert.Equal(ActivationFunctions.Sigmoid, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.SigmoidDerivative, node.ActivationDerivative);
        }

        [Fact]
        public void Node_Constructor_WithTanhActivation_SetsCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            var node = new Neuron(layer, 0, ActivationFunctions.Tanh);
            
            // Assert
            Assert.Equal(ActivationFunctions.Tanh, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.TanhDerivative, node.ActivationDerivative);
        }

        [Fact]
        public void Node_Constructor_WithReLUActivation_SetsCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            var node = new Neuron(layer, 0, ActivationFunctions.ReLU);
            
            // Assert
            Assert.Equal(ActivationFunctions.ReLU, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.ReLUDerivative, node.ActivationDerivative);
        }

        [Fact]
        public void Node_Constructor_WithCustomActivation_UsesSoftPlusDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            Func<double, double> customActivation = x => x * x; // Custom quadratic function
            
            // Act
            var node = new Neuron(layer, 0, customActivation);
            
            // Assert
            Assert.Equal(customActivation, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.SoftPlusDerivative, node.ActivationDerivative);
        }

        [Theory]
        [InlineData(-1)]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(10)]
        [InlineData(100)]
        public void Node_Constructor_WithDifferentIndices_SetsIndexCorrectly(int index)
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            var node = new Neuron(layer, index, ActivationFunctions.Unit);
            
            // Assert
            Assert.Equal(index, node.Index);
        }

        #endregion

        #region Activate Method Tests

        [Fact]
        public void Activate_WithUnitFunction_ReturnsInput()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Unit);
            double input = 5.0;
            
            // Act
            double result = node.Activate(input);
            
            // Assert
            Assert.Equal(input, result, Tolerance);
            Assert.Equal(input, node.X, Tolerance);
        }

        [Fact]
        public void Activate_WithSigmoidFunction_ReturnsCorrectValue()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Sigmoid);
            double input = 0.0;
            
            // Act
            double result = node.Activate(input);
            
            // Assert
            Assert.Equal(0.5, result, Tolerance); // sigmoid(0) = 0.5
            Assert.Equal(input, node.X, Tolerance);
        }

        [Theory]
        [InlineData(0.0, 0.5)]
        [InlineData(1.0, 0.7310585786300049)]
        [InlineData(-1.0, 0.2689414213699951)]
        [InlineData(2.0, 0.8807970779778823)]
        public void Activate_WithSigmoidFunction_ReturnsExpectedValues(double input, double expected)
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Sigmoid);
            
            // Act
            double result = node.Activate(input);
            
            // Assert
            Assert.Equal(expected, result, Tolerance);
            Assert.Equal(input, node.X, Tolerance);
        }

        [Theory]
        [InlineData(-2.0, 0.0)]
        [InlineData(-1.0, 0.0)]
        [InlineData(0.0, 0.0)]
        [InlineData(1.0, 1.0)]
        [InlineData(2.0, 2.0)]
        public void Activate_WithReLUFunction_ReturnsExpectedValues(double input, double expected)
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.ReLU);
            
            // Act
            double result = node.Activate(input);
            
            // Assert
            Assert.Equal(expected, result, Tolerance);
            Assert.Equal(input, node.X, Tolerance);
        }

        [Theory]
        [InlineData(-1.0, -0.7615941559557649)]
        [InlineData(0.0, 0.0)]
        [InlineData(1.0, 0.7615941559557649)]
        [InlineData(2.0, 0.9640275800758169)]
        public void Activate_WithTanhFunction_ReturnsExpectedValues(double input, double expected)
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Tanh);
            
            // Act
            double result = node.Activate(input);
            
            // Assert
            Assert.Equal(expected, result, Tolerance);
            Assert.Equal(input, node.X, Tolerance);
        }

        [Fact]
        public void Activate_WithSoftPlusFunction_ReturnsPositiveValues()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.SoftPlus);
            
            // Act & Assert
            Assert.True(node.Activate(-10.0) > 0);
            Assert.True(node.Activate(0.0) > 0);
            Assert.True(node.Activate(10.0) > 0);
        }

        [Fact]
        public void Activate_StoreslastInputInX()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Unit);
            
            // Act
            node.Activate(3.14);
            Assert.Equal(3.14, node.X, Tolerance);
            
            node.Activate(-2.71);
            Assert.Equal(-2.71, node.X, Tolerance);
        }

        [Fact]
        public void Activate_WithExtremeValues_HandlesCorrectly()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Unit);
            
            // Act & Assert
            Assert.Equal(double.MaxValue, node.Activate(double.MaxValue));
            Assert.Equal(double.MinValue, node.Activate(double.MinValue));
            Assert.Equal(0.0, node.Activate(0.0));
            Assert.Equal(double.Epsilon, node.Activate(double.Epsilon));
        }

        #endregion

        #region Derivative Method Tests

        [Fact]
        public void Derivative_WithUnitFunction_ReturnsOne()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Unit);
            
            // Act & Assert
            Assert.Equal(1.0, node.Derivative(0.0), Tolerance);
            Assert.Equal(1.0, node.Derivative(5.0), Tolerance);
            Assert.Equal(1.0, node.Derivative(-5.0), Tolerance);
        }

        [Theory]
        [InlineData(-2.0, 0.0)]
        [InlineData(-1.0, 0.0)]
        [InlineData(0.0, 0.0)]
        [InlineData(1.0, 1.0)]
        [InlineData(2.0, 1.0)]
        public void Derivative_WithReLUFunction_ReturnsExpectedValues(double input, double expected)
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.ReLU);
            
            // Act
            double result = node.Derivative(input);
            
            // Assert
            Assert.Equal(expected, result, Tolerance);
        }

        [Theory]
        [InlineData(0.0, 0.25)] // sigmoid'(0) = 0.25
        [InlineData(1.0, 0.19661193324148185)] // sigmoid'(1) ≈ 0.1966
        public void Derivative_WithSigmoidFunction_ReturnsExpectedValues(double input, double expected)
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Sigmoid);
            
            // Act
            double derivative = node.Derivative(input);
            
            // Assert
            Assert.Equal(expected, derivative, Tolerance);
        }

        [Theory]
        [InlineData(0.0, 1.0)] // tanh'(0) = 1.0
        [InlineData(1.0, 0.4199743416140261)] // tanh'(1) ≈ 0.4200
        [InlineData(2.0, 0.07065082485316443)] // tanh'(2) ≈ 0.0707
        public void Derivative_WithTanhFunction_ReturnsExpectedValues(double input, double expected)
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Tanh);
            
            // Act
            double derivative = node.Derivative(input);
            
            // Assert
            Assert.Equal(expected, derivative, Tolerance);
        }

        [Fact]
        public void Derivative_WithSoftPlusFunction_UsesCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.SoftPlus);
            
            // Act
            double derivative = node.Derivative(0.0);
            
            // Assert
            // SoftPlus derivative is sigmoid, so derivative(0) should be 0.5
            Assert.Equal(0.5, derivative, Tolerance);
        }

        #endregion

        #region Property Tests

        [Fact]
        public void Properties_CanBeSetAndRetrieved()
        {
            // Arrange
            var layer1 = new MockLayer();
            var layer2 = new MockLayer();
            var node = new Neuron(layer1, 0, ActivationFunctions.Unit);
            
            // Act & Assert - Initial values
            Assert.Equal(layer1, node.Layer);
            Assert.Equal(0, node.Index);
            Assert.Equal(0.0, node.Y);
            Assert.Equal(0.0, node.X);
            
            // Act & Assert - Modified values
            node.Layer = layer2;
            node.Index = 5;
            node.Y = 3.14;
            node.X = 2.71;
            
            Assert.Equal(layer2, node.Layer);
            Assert.Equal(5, node.Index);
            Assert.Equal(3.14, node.Y, Tolerance);
            Assert.Equal(2.71, node.X, Tolerance);
        }

        [Fact]
        public void ActivationFunction_CanBeChanged()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Unit);
            
            // Act
            node.ActivationFunction = ActivationFunctions.Sigmoid;
            
            // Assert
            Assert.Equal(ActivationFunctions.Sigmoid, node.ActivationFunction);
            // Note: Changing activation function doesn't automatically update derivative
        }

        [Fact]
        public void ActivationDerivative_CanBeChanged()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Unit);
            
            // Act
            node.ActivationDerivative = ActivationFunctions.SigmoidDerivative;
            
            // Assert
            Assert.Equal(ActivationFunctions.SigmoidDerivative, node.ActivationDerivative);
        }

        #endregion

        #region Interface Implementation Tests

        [Fact]
        public void Neuron_ImplementsINeuron()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            INeuron neuron = new Neuron(layer, 0, ActivationFunctions.Unit);
            
            // Assert
            Assert.IsAssignableFrom<INeuron>(neuron);
            Assert.Equal(0, neuron.Index);
            Assert.Equal(ActivationFunctions.Unit, neuron.ActivationFunction);
            Assert.Equal(ActivationFunctions.UnitDerivative, neuron.ActivationDerivative);
        }

        [Fact]
        public void Neuron_InterfaceMethods_WorkCorrectly()
        {
            // Arrange
            var layer = new MockLayer();
            INeuron neuron = new Neuron(layer, 0, ActivationFunctions.Sigmoid);
            
            // Act
            double activated = neuron.Activate(0.0);
            double derivative = neuron.Derivative(0.0);  // Pass input, not output
            
            // Assert
            Assert.Equal(0.5, activated, Tolerance);
            Assert.Equal(0.25, derivative, Tolerance);
        }

        [Fact]
        public void Neuron_InterfaceProperties_CanBeModified()
        {
            // Arrange
            var layer = new MockLayer();
            INeuron neuron = new Neuron(layer, 0, ActivationFunctions.Unit);
            
            // Act
            neuron.Index = 10;
            neuron.ActivationFunction = ActivationFunctions.Tanh;
            neuron.ActivationDerivative = ActivationFunctions.TanhDerivative;
            
            // Assert
            Assert.Equal(10, neuron.Index);
            Assert.Equal(ActivationFunctions.Tanh, neuron.ActivationFunction);
            Assert.Equal(ActivationFunctions.TanhDerivative, neuron.ActivationDerivative);
        }

        #endregion

        #region Factory Tests

        [Fact]
        public void NeuronFactory_Create_ReturnsNeuronInstance()
        {
            // Arrange
            var factory = new NeuronFactory();
            var layer = new MockLayer();
            
            // Act
            var neuron = factory.Create(layer, 5, ActivationFunctions.Sigmoid);
            
            // Assert
            Assert.NotNull(neuron);
            Assert.IsType<Neuron>(neuron);
            Assert.Equal(layer, ((Neuron)neuron).Layer);
            Assert.Equal(5, neuron.Index);
            Assert.Equal(ActivationFunctions.Sigmoid, neuron.ActivationFunction);
        }

        [Fact]
        public void NeuronFactory_Create_WithNullActivation_UsesDefault()
        {
            // Arrange
            var factory = new NeuronFactory();
            var layer = new MockLayer();
            
            // Act
            var neuron = factory.Create(layer, 0, null!);
            
            // Assert
            Assert.NotNull(neuron);
            Assert.Equal(ActivationFunctions.SoftPlus, neuron.ActivationFunction);
        }

        [Fact]
        public void NeuronFactory_Create_WithNullLayer_ThrowsException()
        {
            // Arrange
            var factory = new NeuronFactory();
            
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => factory.Create(null!, 0, ActivationFunctions.Unit));
        }

        [Fact]
        public void NeuronFactory_ImplementsINeuronFactory()
        {
            // Arrange & Act
            INeuronFactory factory = new NeuronFactory();
            
            // Assert
            Assert.IsAssignableFrom<INeuronFactory>(factory);
        }

        #endregion

        #region Edge Cases and Performance Tests

        [Fact]
        public void Activate_MultipleCallsWithSameInput_ConsistentResults()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Sigmoid);
            double input = 1.5;
            
            // Act
            double result1 = node.Activate(input);
            double result2 = node.Activate(input);
            double result3 = node.Activate(input);
            
            // Assert
            Assert.Equal(result1, result2, Tolerance);
            Assert.Equal(result2, result3, Tolerance);
        }

        [Fact]
        public void Derivative_MultipleCallsWithSameInput_ConsistentResults()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Sigmoid);
            double input = 0.5;
            
            // Act
            double result1 = node.Derivative(input);
            double result2 = node.Derivative(input);
            double result3 = node.Derivative(input);
            
            // Assert
            Assert.Equal(result1, result2, Tolerance);
            Assert.Equal(result2, result3, Tolerance);
        }

        [Fact]
        public void Activate_WithLargeInputArray_ProcessesEfficiently()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Unit);
            var inputs = Enumerable.Range(0, 10000).Select(i => (double)i).ToArray();
            
            // Act & Assert - Should complete without issues
            foreach (var input in inputs)
            {
                double result = node.Activate(input);
                Assert.Equal(input, result, Tolerance);
            }
        }

        [Theory]
        [InlineData(double.NaN)]
        [InlineData(double.PositiveInfinity)]
        [InlineData(double.NegativeInfinity)]
        public void Activate_WithSpecialValues_HandlesGracefully(double specialValue)
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Unit);
            
            // Act
            double result = node.Activate(specialValue);
            
            // Assert
            Assert.Equal(specialValue, result);
            Assert.Equal(specialValue, node.X);
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void NeuronWorkflow_ActivateAndDerivative_WorksTogether()
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, ActivationFunctions.Sigmoid);
            double input = 2.0;
            
            // Act
            double activated = node.Activate(input);
            double derivative = node.Derivative(activated);
            
            // Assert
            Assert.True(activated >= 0 && activated <= 1); // Sigmoid output range
            Assert.True(derivative >= 0 && derivative <= 0.25); // Sigmoid derivative max is 0.25
            Assert.Equal(input, node.X, Tolerance);
        }

        [Fact]
        public void NeuronWorkflow_DifferentActivationFunctions_ProduceExpectedResults()
        {
            // Arrange
            var layer = new MockLayer();
            double input = 1.0;
            
            // Act & Assert - Unit
            var unitNode = new Neuron(layer, 0, ActivationFunctions.Unit);
            Assert.Equal(1.0, unitNode.Activate(input), Tolerance);
            Assert.Equal(1.0, unitNode.Derivative(input), Tolerance);
            
            // Act & Assert - ReLU
            var reluNode = new Neuron(layer, 1, ActivationFunctions.ReLU);
            Assert.Equal(1.0, reluNode.Activate(input), Tolerance);
            Assert.Equal(1.0, reluNode.Derivative(input), Tolerance);
            
            // Act & Assert - Sigmoid
            var sigmoidNode = new Neuron(layer, 2, ActivationFunctions.Sigmoid);
            double sigmoidResult = sigmoidNode.Activate(input);
            Assert.True(sigmoidResult > 0.5 && sigmoidResult < 1.0);
        }

        #endregion
    }

    // Mock layer class for testing
    public class MockLayer : ILayer
    {
        public INeuron[] Neurons { get; set; } = Array.Empty<INeuron>();
        public ILayer? PreviousLayer { get; set; }
        public ILayer? NextLayer { get; set; }
        public double[]? Inputs { get; set; }
        public int Index { get; set; }
        public IInputProcessor[] InputProcessors { get; set; } = Array.Empty<IInputProcessor>();

        public double[] Forward(double[] inputs)
        {
            return inputs;
        }

        public Gradients[] Backward(double dSSR, Gradients[] steps)
        {
            return steps;
        }

        public double CalculateLayerErrorRecursively(int index)
        {
            return 1.0; // Simplified for testing
        }

        public double GetBiasChainFactor()
        {
            return 1.0; // Simplified for testing
        }
    }
}