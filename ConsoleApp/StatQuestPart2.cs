using System;

using BackPropagation.NNLib;
namespace BackPropagation;


public class StatQuestPart2
{
    const int epochs = 50000;
    double[][] samples = [[0], [0.5], [1]];
    double[][] observed = [[0], [1], [0]];

    double[][][] weights = [
        [[2.74], [-1.13]],
    [[0.36, 0.63]],
    ];
    double[][][] biases = [[[0], [0]], [[0]]];
    double[][] ys = [[0, 0], [0, 0], [0, 0], [0, 0]];

    Func<double, double>[] activationFunctions = [
        ActivationFunctions.SoftPlus,
        ActivationFunctions.Unit
    ];

    public StatQuestPart2()
    {
        NeuralNetwork nn1 = new(new LayerFactory(), new NeuronFactory(), new InputProcessorFactory(),
    weights, biases, ys, activationFunctions);
        NeuralNetworkTrainer trainer = new(nn1, 0.1);

        double[] output0 = nn1.Predict([0]);
        double[] output1 = nn1.Predict([0.5]);
        double[] output2 = nn1.Predict([1.0]);
        Console.WriteLine($"Outputs: {output0[0]}, {output1[0]}, {output2[0]} Sum = {output0[0] + output1[0] + output2[0]}");



        for (int i = 0; i < epochs; i++)
        {
            trainer.TrainOneEpoch(samples, observed);
        }
        //        Console.WriteLine($"SSR nn1: {nn1.SSR[0]}");
    }

}
