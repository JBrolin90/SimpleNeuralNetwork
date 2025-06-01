using System;

namespace BackPropagation.NNLib;

public class BackPropagator
{
    public INeuralNetwork NN { get; set; }
    public double LearningRate { get; set; } = 0.01;
    public BackPropagator(INeuralNetwork nn, double learningRate = 0.01)
    {
        NN = nn;
        LearningRate = learningRate;
    }
    public void BackPropagate(double[] inputs, double[] expectedOutputs)
    {
        // Forward pass
        double[] outputs = NN.Predict(inputs);

        // Calculate the error
        double[] errors = new double[outputs.Length];
        for (int i = 0; i < outputs.Length; i++)
        {
            errors[i] = expectedOutputs[i] - outputs[i];
        }

        // Backward pass
        double[][][] weightUpdates = NN.BackwardPass(inputs, errors);

        // Update weights and biases
        // Iterate through layers in reverse order to apply weight updates        
        for (int i = NN.Layers.Length - 1; i >= 0; i--)
        {
            var layer = NN.Layers[i];
            for (int j = 0; j < layer.Nodes.Length; j++)
            {
                var node = layer.Nodes[j];
                for (int k = 0; k < node.Weights.Length; k++)
                {
                    node.Weights[k] += weightUpdates[i][j][k];
                }
                for (int k = 0; k < node.Bias.Length; k++)
                {
                    node.Bias[k] += LearningRate * node.BiasDerivative();
                }
            }
        }
    }
    public double[] Predict(double[] inputs)
    {
        return NN.Predict(inputs);
    }
    public double[] Train(double[] inputs, double[] expectedOutputs)
    {
        BackPropagate(inputs, expectedOutputs);
        return Predict(inputs);
    }
    public void TrainBatch(double[][] inputs, double[][] expectedOutputs)
    {
        for (int i = 0; i < inputs.Length; i++)
        {
            Train(inputs[i], expectedOutputs[i]);
        }
    }
    public void SetLearningRate(double learningRate)
    {
        LearningRate = learningRate;
    }
    public double GetLearningRate()
    {
        return LearningRate;
    }
    public void Reset()
    {
        // Reset the neural network state if needed
        foreach (var layer in NN.Layers)
        {
            foreach (var node in layer.Nodes)
            {
                //node.Reset();
            }
        }
    }
}
