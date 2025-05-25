using System;

namespace BackPropagation.NNLib;

public class TestNeuralNetwork : NeuralNetwork
{
    public double SSR = 0;
    public double dSSR = 0;

    public TestNeuralNetwork(ILayerFactory LayerFactory, INodeFactory NodeFactory,
                        double[][][] weights, double[][] biases,
                        double learningRate = 0.01, Func<double, double>[]? activationFunctions = null)
                        : base(LayerFactory, NodeFactory, weights, biases, activationFunctions, learningRate)
    { }

    public void Test(double[] inputs, double[] expectedOutputs)
    {
        double[][] predictions = new double[inputs.Length][];
        SSR = 0; dSSR = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            predictions[i] = Predict([inputs[i]]);
            SSR += Math.Pow(expectedOutputs[i] - predictions[i][0], 2);
            dSSR = -2 * (expectedOutputs[i] - predictions[i][0]);
            db3 = dSSR * 1; // Assuming a single output node
        }
        Console.WriteLine($"Inputs: {string.Join(", ", inputs)}");
        Console.WriteLine($"Expected Outputs: {string.Join(", ", expectedOutputs)}");
        Console.WriteLine($"Predicted Outputs: {string.Join(", ", predictions.Select(arr => string.Join(";", arr)))}");
        Console.WriteLine();
    }

}
