using BackPropagation.NNLib;
internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, BackPropagation learners!");

        double[] inputs = [0, 0.5, 1];
        double[] observed = [0, 1, 0];

        double[][][] weights = [
            [],
            [[2.74], [-1.13]],
            [[0.36, 0.63]],
            []
            ];
        double[][][] biases = [[], [[0], [0]], [[0]], []];
        double[][] ys = [[0, 0], [0, 0], [0, 0], [0, 0]];


        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit, // Activation function for the second layer
    ActivationFunctions.SoftPlus, // Activation function for the first layer
    ActivationFunctions.Unit, // Activation function for the second layer
    ActivationFunctions.Unit // Activation function for the second layer
        ];


        NeuralNetworkTrainer nn = new(new LayerFactory(), new NodeFactory(),
             weights, biases, ys, 0.01, activationFunctions);
        NetworkCreator creator = new NetworkCreator([1, 2, 1, 1], activationFunctions);
        creator.RandomizeWeights();
        //creator.Weights = [[], [[2.74], [-1.13]], [[0.36, 0.63]], []];
        //creator.Biases = [[], [[0], [0]], [[0]], []]; 
        //creator.Ys = ys;
        //creator.ActivationFunctions = activationFunctions;
        for (int i = 0; i < 500; i++)
        {
            nn.Train(inputs, observed);
        }
        Console.WriteLine($"SSR: {nn.SSR}");

        NeuralNetworkTrainer nn2 = creator.CreateNetwork();
        for (int i = 0; i < 500; i++)
        {
            nn2.Train(inputs, observed);
        }
        Console.WriteLine($"SSR2: {nn2.SSR}");
        Console.WriteLine($"SSR: ");

    }
}