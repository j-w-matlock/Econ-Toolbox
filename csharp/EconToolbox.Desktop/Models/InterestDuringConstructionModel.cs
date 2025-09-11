using System.Linq;

namespace EconToolbox.Desktop.Models
{
    public static class InterestDuringConstructionModel
    {
        public static double Compute(double totalInitialCost, double rate, int months, double[]? costs = null, string[]? timings = null)
        {
            if (months <= 0) return 0.0;
            double monthlyRate = rate / 12.0;
            if (costs == null)
            {
                double monthlyCost = totalInitialCost / months;
                costs = Enumerable.Repeat(monthlyCost, months).ToArray();
                timings = new string[months];
                timings[0] = "beginning";
                for (int i = 1; i < months; i++) timings[i] = "middle";
            }
            else if (timings == null)
            {
                timings = Enumerable.Repeat("middle", costs.Length).ToArray();
            }
            double idc = 0.0;
            for (int i = 0; i < costs.Length; i++)
            {
                string timing = timings![i];
                double remaining;
                if (timing == "beginning") remaining = months - i;
                else if (timing == "end") remaining = months - i - 1;
                else remaining = months - i - 0.5;
                idc += costs[i] * monthlyRate * remaining;
            }
            return idc;
        }
    }
}
