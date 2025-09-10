using EconToolbox.Desktop.Models;
using System.Windows.Input;

namespace EconToolbox.Desktop.ViewModels
{
    public class StorageCostViewModel : BaseViewModel
    {
        private double _totalCost;
        private double _storagePrice;
        private double _storageReallocated;
        private double _totalStorage;
        private string _result = string.Empty;

        public double TotalCost
        {
            get => _totalCost;
            set { _totalCost = value; OnPropertyChanged(); }
        }

        public double StoragePrice
        {
            get => _storagePrice;
            set { _storagePrice = value; OnPropertyChanged(); }
        }

        public double StorageReallocated
        {
            get => _storageReallocated;
            set { _storageReallocated = value; OnPropertyChanged(); }
        }

        public double TotalStorage
        {
            get => _totalStorage;
            set { _totalStorage = value; OnPropertyChanged(); }
        }

        public string Result
        {
            get => _result;
            set { _result = value; OnPropertyChanged(); }
        }

        public ICommand ComputeCommand { get; }

        public StorageCostViewModel()
        {
            ComputeCommand = new RelayCommand(Compute);
        }

        private void Compute()
        {
            double updated = EconomicCalculator.UpdatedStorageCost(TotalCost, StoragePrice, StorageReallocated, TotalStorage);
            Result = $"Updated cost: {updated:F2}";
        }
    }
}
