using System;

namespace BackPropagation.NNLib;

public static class LossFunctions
{
    public static double[] SquaredError(double[] predicted, double[] observed)
    {
        if (predicted == null)
            throw new ArgumentNullException(nameof(predicted));
        if (observed == null)
            throw new ArgumentNullException(nameof(observed));
        if (predicted.Length != observed.Length)
            throw new ArgumentException("Arrays must be of the same length");

        int count = predicted.Length;
        double[] errors = new double[count];
        for (int i = 0; i < count; i++)
        {
            double diff = (predicted[i] - observed[i]);
            errors[i] = diff * diff;
        }
        return errors;
    }
    public static double[] SquaredErrorDerivative(double[] predicted, double[] observed)
    {
        if (predicted.Length != observed.Length)
            throw new ArgumentException("Arrays must be of the same length");

        int count = predicted.Length;
        double[] errors = new double[count];
        for (int i = 0; i < count; i++)
        {
            double diff = (predicted[i] - observed[i]);
            errors[i] = 2 * diff;
        }
        return errors;
    }
    public static double[] SumSquaredError(double[][] predicted, double[][] observed)
    {
        if (predicted.Length != observed.Length)
            throw new ArgumentException("Arrays must be of the same length");

        int outputCount = predicted[0].Length;
        double[] sum = new double[outputCount];
        for (int predictionIndex = 0; predictionIndex < predicted.Length; predictionIndex++)
        {
            for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
            {
                double[] diff = new double[outputCount];
                diff[outputIndex] = predicted[predictionIndex][outputIndex] - observed[predictionIndex][outputIndex];
                sum[outputIndex] += diff[outputIndex] * diff[outputIndex];
            }
        }
        return sum;
    }
    public static double[] SumSquaredErrorDerivative(double[][] predicted, double[][] observed)
    {
        if (predicted.Length != observed.Length)
            throw new ArgumentException("Arrays must be of the same length");

        int outputCount = predicted[0].Length;
        double[] sum = new double[outputCount];
        for (int predictionIndex = 0; predictionIndex < predicted.Length; predictionIndex++)
        {
            for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
            {
                double[] diff = new double[outputCount];
                diff[outputIndex] = predicted[predictionIndex][outputIndex] - observed[predictionIndex][outputIndex];
                sum[outputIndex] += 2 * diff[outputIndex];
            }
        }
        return sum;
    }
    public static double[] SumMeanSquaredError(double[][] predicted, double[][] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Arrays must be of the same length");

        int outputCunt = predicted[0].Length;
        double[] sum = new double[outputCunt];
        for (int predictionIndex = 0; predictionIndex < predicted.Length; predictionIndex++)
        {
            for (int outputIndex = 0; outputIndex < outputCunt; outputIndex++)
            {
                double[] diff = new double[outputCunt];
                diff[outputIndex] = predicted[predictionIndex][outputIndex] - actual[predictionIndex][outputIndex];
                sum[outputIndex] += diff[outputIndex] * diff[outputIndex];
            }
        }
        for (int outputIndex = 0; outputIndex < outputCunt; outputIndex++)
        {
            sum[outputIndex] = sum[outputIndex] / predicted.Length;
        }
        return sum;
    }

    public static double[] SumMeanSquaredErrorDerivative(double[][] predicted, double[][] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Arrays must be of the same length");

        int outputCunt = predicted[0].Length;
        double[] sum = new double[outputCunt];
        for (int predictionIndex = 0; predictionIndex < predicted.Length; predictionIndex++)
        {
            for (int outputIndex = 0; outputIndex < outputCunt; outputIndex++)
            {
                double[] diff = new double[outputCunt];
                diff[outputIndex] = predicted[predictionIndex][outputIndex] - actual[predictionIndex][outputIndex];
                sum[outputIndex] += 2 * diff[outputIndex];
            }
        }
        for (int outputIndex = 0; outputIndex < outputCunt; outputIndex++)
        {
            sum[outputIndex] = sum[outputIndex] / predicted.Length;
        }
        return sum;
    }

    public static double CrossEntropyLoss(double[] predicted, double[] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Arrays must be of the same length");

        double loss = 0.0;
        for (int i = 0; i < predicted.Length; i++)
        {
            // Clamp the predicted values to avoid log(0)
            double p = Math.Max(predicted[i], 1e-15);
            loss -= actual[i] * Math.Log(p);
        }
        return loss / predicted.Length;
    }

    public static double HingeLoss(double[] predicted, double[] actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Arrays must be of the same length");

        double loss = 0.0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double margin = actual[i] * predicted[i];
            loss += Math.Max(0, 1 - margin);
        }
        return loss / predicted.Length;
    }
    public static double HuberLoss(double[] predicted, double[] actual, double delta = 1.0)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Arrays must be of the same length");

        double loss = 0.0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double diff = predicted[i] - actual[i];
            if (Math.Abs(diff) <= delta)
            {
                loss += 0.5 * diff * diff;
            }
            else
            {
                loss += delta * (Math.Abs(diff) - 0.5 * delta);
            }
        }
        return loss / predicted.Length;
    }
}
