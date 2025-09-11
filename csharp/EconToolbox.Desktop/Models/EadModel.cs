using System;

namespace EconToolbox.Desktop.Models
{
    public static class EadModel
    {
        public static double Compute(double[] probabilities, double[] damages)
        {
            if (probabilities.Length != damages.Length)
                throw new ArgumentException("Probability and damage counts must match");
            double sum = 0.0;
            for (int i = 0; i < probabilities.Length - 1; i++)
            {
                sum += 0.5 * (damages[i] + damages[i + 1]) * (probabilities[i] - probabilities[i + 1]);
            }
            return sum;
        }
    }
}
