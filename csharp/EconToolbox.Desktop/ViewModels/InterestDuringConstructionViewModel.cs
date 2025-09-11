using System;
using System.Linq;
using System.Windows.Input;
using EconToolbox.Desktop.Models;

namespace EconToolbox.Desktop.ViewModels
{
    public class InterestDuringConstructionViewModel : BaseViewModel
    {
        private double _totalInitialCost;
        private double _rate = 5.0;
        private int _months = 12;
        private string _costs = string.Empty;
        private string _timings = string.Empty;
        private string _result = string.Empty;

        public double TotalInitialCost
        {
            get => _totalInitialCost;
            set { _totalInitialCost = value; OnPropertyChanged(); }
        }

        public double Rate
        {
            get => _rate;
            set { _rate = value; OnPropertyChanged(); }
        }

        public int Months
        {
            get => _months;
            set { _months = value; OnPropertyChanged(); }
        }

        public string Costs
        {
            get => _costs;
            set { _costs = value; OnPropertyChanged(); }
        }

        public string Timings
        {
            get => _timings;
            set { _timings = value; OnPropertyChanged(); }
        }

        public string Result
        {
            get => _result;
            set { _result = value; OnPropertyChanged(); }
        }

        public ICommand ComputeCommand { get; }

        public InterestDuringConstructionViewModel()
        {
            ComputeCommand = new RelayCommand(Compute);
        }

        private void Compute()
        {
            try
            {
                double[]? costArr = null;
                string[]? timingArr = null;
                if (!string.IsNullOrWhiteSpace(Costs))
                    costArr = Costs.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s.Trim())).ToArray();
                if (!string.IsNullOrWhiteSpace(Timings))
                    timingArr = Timings.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => s.Trim()).ToArray();
                double idc = InterestDuringConstructionModel.Compute(TotalInitialCost, Rate / 100.0, Months, costArr, timingArr);
                Result = $"Interest during construction: {idc:F2}";
            }
            catch (Exception ex)
            {
                Result = $"Error: {ex.Message}";
            }
        }
    }
}
