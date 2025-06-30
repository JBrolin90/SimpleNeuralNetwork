using System;

using BackPropagation.NNLib;
namespace BackPropagation;

public class StatQuestPart2Random
{

    const int epochs = 50000;

    double[][] samples = [[0], [0.5], [1]];
    double[][] observed = [[0], [1], [0]];

    Func<double, double>[] activationFunctions = [
        ActivationFunctions.SoftPlus,
        ActivationFunctions.Unit
    ];

    public StatQuestPart2Random()
    {
        NetworkCreator creator = new(1, [2, 1], activationFunctions);
        creator.RandomizeWeights(-0.3, 0.3);
        INeuralNetwork nn2 = creator.CreateNetwork();
        NeuralNetworkTrainer trainer = new(nn2, 0.1);

        for (int i = 0; i < epochs; i++)
        {
            trainer.TrainOneEpoch(samples, observed);
        }
        //        Console.WriteLine($"SSR nn2: {nn2.SSR[0]}");
    }

}
