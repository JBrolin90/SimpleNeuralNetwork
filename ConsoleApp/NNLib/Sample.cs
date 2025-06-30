using System;

namespace BackPropagation.NNLib;

public enum Operation { add, hypot }
public class Sample
{
    private double[] _sample = new double[4];
    private double[] observed = [];
    public double[] Xample => _sample;
    public double[] Observed => observed;

    public Sample(double a, double b, Operation op, double normalizer)
    {
        int i = 0;
        _sample[i++] = a;
        _sample[i++] = b;
        if (op == Operation.add)
        {
            _sample[i++] = 0;
            _sample[i++] = 1;
            observed[0] = a + b;
        }
        else if (op == Operation.hypot)
        {
            _sample[i++] = 1;
            _sample[i++] = 0;
            double A = a * normalizer;
            double B = b * normalizer;
            observed[0] = Math.Sqrt(A * A + B * B);
        }
    }
}
