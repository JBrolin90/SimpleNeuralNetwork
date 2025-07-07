using System;

namespace BackPropagation.NNLib;

public interface IInputProcessorFactory
{
    public IInputProcessor Build(ILayer layer, int index, double[] weights, double[] biase);
}
public class InputProcessorFactory: IInputProcessorFactory
{
    public IInputProcessor Build(ILayer layer, int index, double[] weights, double[] biase)
    {
        return new InputProcessor(layer, index, weights, biase);
    }
}
public interface IInputProcessor
{
    ILayer Layer { get; set; }
    int Index { get; set; }
    double[] Weights { get; set; }
    double[] Bias { get; set; }
    double[] I { get; set; }
    double Y { get; set; }

    double ProcessInputs(double[] xs);
}

public class InputProcessor : IInputProcessor
{
    public ILayer Layer { get; set; }
    public int Index { get; set; }
    public double[] Weights { get; set; }
    public double[] Bias { get; set; }

    public InputProcessor(ILayer layer, int index, double[] weights, double[] bias)
    {
        Layer = layer;
        Index = index;
        Weights = weights;
        Bias = bias;
    }

    public double[] I { get; set; } = [];
    public double Y { get; set; } = 0;
    public double ProcessInputs(double[] inputs)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (inputs.Length != Weights.Length)
        {
            throw new IndexOutOfRangeException("Input size does not match weights size.");
        }

        Y = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            double product = inputs[i] * Weights[i];
            if (double.IsInfinity(inputs[i]) && Weights[i] == 0)
            {
                product = 0; // Or some other sensible default
            }
            Y += product;
        }
        Y += Bias[0];
        return Y;
    }

}
