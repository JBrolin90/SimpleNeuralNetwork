using System.Globalization;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class MinimalTest
{
    const int epochs = 100;
    double[][] samples = [];
    NeuralNetworkTrainer? network;

    public void CreateTrainingData(int sampleCount = 3)
    {
        samples = [[0], [0.5], [1]];
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.SoftPlus, ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(1, [2, 1], af);
        network = networkCreator.CreateNetwork(0.01);
    }

    public void Train()
    {
        for (int i = 0; i < epochs; i++)
        {
            double[] loss = network!.TrainOneEpoch(samples, [[1], [0], [1]]);
            Console.WriteLine($"Loss = {loss[0]}\n");
        }
    }


    private double[][][] FixedWeights()
    {
        double[][][] Weights = network!.Weigths;
        for (int i = 0; i < Weights.Length; i++)
        {
            double[][] nodes = Weights[i];
            for (int j = 0; j < nodes.Length; j++)
            {
                double[] inputs = nodes[j];
                for (int k = 0; k < inputs.Length; k++)
                {
                    inputs[k] = 3;
                }
            }
        }
        return Weights;
    }
    public void DoIt()
    {
        CreateNetwork();

        Console.WriteLine("Before training:");

        CreateTrainingData();

        double[][][] Weights = network!.Weigths;
        Weights = [
            [[2.74], [-1.13]],
            [[0.36, 0.63]],
            []
            ];
        double[][][] biases = network.Biases = [[[0], [0]], [[0]]];




        Train();

        Console.WriteLine("After training:");
        var prediction = network.Predict([0.5]);
        Console.WriteLine($"0.5 => {prediction[0]}");
    }
}
