using System.Globalization;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class MinimalTest
{
    const int epochs = 900000;
    double[][] samples = [];
    INeuralNetwork? network;
    NeuralNetworkTrainer? trainer;

    public void CreateTrainingData(int sampleCount = 3)
    {
        samples = [[0], [0.5], [1]];
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.SoftPlus, ActivationFunctions.Unit, ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(1, [2, 1], af);
        networkCreator.RandomizeWeights(-2, 2);
        network = networkCreator.CreateNetwork();
        trainer = new(network, 0.01);
    }

    public void Train()
    {
        for (int i = 0; i < epochs; i++)
        {
            double[] loss = trainer!.TrainOneEpoch(samples, [[0], [1], [0]]);
            if (epochs % (epochs / 10) == 0)
            {
                //Console.WriteLine($"Epoch: {i}");
            }
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


        Train();
        static void print(double val)
        {
            Console.WriteLine($"Weight: {val}");    
        }
        NetworkCreator.ActOn3dArr(network!.Weigths, print);
        Console.WriteLine();
        NetworkCreator.ActOn3dArr(network!.Biases, print);

        Console.WriteLine("After training:");
        var prediction = network!.Predict([0]);
        Console.WriteLine($"0 => {prediction[0]}");
        prediction = network.Predict([0.5]);
        Console.WriteLine($"0.5 => {prediction[0]}");
        prediction = network.Predict([1]);
        Console.WriteLine($"1 => {prediction[0]}");
    }
}
