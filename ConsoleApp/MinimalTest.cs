using System.Globalization;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class MinimalTest
{
    const int epochs = 1;
    double[][] samples = [];
    NeuralNetworkTrainer? network;

    public void CreateTrainingData(int sampleCount = 3)
    {
        samples = [[0], [0.5], [1]];
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.SoftPlus, ActivationFunctions.Unit, ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(1, [2, 1], af);
        networkCreator.RandomizeWeights(-2, 2);
        network = networkCreator.CreateNetwork(0.001);
    }

    public void Train()
    {
        for (int i = 0; i < epochs; i++)
        {
            double[] loss = network!.TrainOneEpoch(samples, [[0], [1], [0]]);
            Console.WriteLine($"Loss = {loss[0]}\n");
        }
    }


    private double[][][] FixedWeights()
    {
        double FixedWeight(double _) => 3;
        NetworkCreator.ApplyOn3dArr(network!.Weigths, FixedWeight);
        return network.Weigths;
    }
    public void DoIt()
    {
        CreateNetwork();

        Console.WriteLine("Before training:");

        CreateTrainingData();

        
        double[][][] Weigths = [
            [[2.74], [-1.13]],
            [[0.36, 0.63]],
            ];


        int layerCount = Weigths.Length;
        for (int i = 0; i < layerCount; i++)
        {
            int nodeCount = Weigths[i].Length;
            for (int j = 0; j < nodeCount; j++)
            {
                int inputCount = Weigths[i][j].Length;
                for (int k = 0; k < inputCount; k++)
                {
                    network!.Weigths[i][j][k] = Weigths[i][j][k];
                }
            }
        }


            Train();

        Console.WriteLine("After training:");
        var prediction = network!.Predict([0]);
        Console.WriteLine($"0 => {prediction[0]}");
        prediction = network.Predict([0.5]);
        Console.WriteLine($"0.5 => {prediction[0]}");
        prediction = network.Predict([1]);
        Console.WriteLine($"1 => {prediction[0]}");
    }
}
