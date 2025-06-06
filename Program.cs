using BackPropagation.NNLib;
internal class Program
{
    const int epochs = 3000000;
    private static void Main(string[] args)
    {
        // Debug NetworkCreator structure
        var inputs = 1;
        var layerSizes = new int[] { 2, 1 };
        var activationFunctions = new Func<double, double>[] {
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        };

        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

        Console.WriteLine($"Inputs: {inputs}");
        Console.WriteLine($"Layer sizes: [{string.Join(", ", layerSizes)}]");
        Console.WriteLine($"Weights structure:");

        for (int i = 0; i < creator.Weights.Length; i++)
        {
            Console.WriteLine($"  Layer {i}: {creator.Weights[i].Length} nodes");
            for (int j = 0; j < creator.Weights[i].Length; j++)
            {
                Console.WriteLine($"    Node {j}: {creator.Weights[i][j].Length} weights");
            }
        }

        Console.WriteLine($"\nBiases structure:");
        for (int i = 0; i < creator.Biases.Length; i++)
        {
            Console.WriteLine($"  Layer {i}: {creator.Biases[i].Length} nodes");
            for (int j = 0; j < creator.Biases[i].Length; j++)
            {
                Console.WriteLine($"    Node {j}: {creator.Biases[i][j].Length} biases");
            }
        }

        Console.WriteLine($"\nTrying to access creator.Weights[1][1][0]...");
        try
        {
            var value = creator.Weights[1][1][0];
            Console.WriteLine($"Successfully accessed: {value}");
        }
        catch (IndexOutOfRangeException ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine($"Layer 1 has {creator.Weights[1].Length} nodes, but trying to access node 1 (0-based)");
        }
    }
}