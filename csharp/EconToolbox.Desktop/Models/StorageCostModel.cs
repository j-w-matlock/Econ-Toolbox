namespace EconToolbox.Desktop.Models
{
    public static class StorageCostModel
    {
        public static double Compute(double totalCost, double storagePrice, double storageReallocated, double totalUsableStorage)
        {
            return (totalCost - storagePrice) * storageReallocated / totalUsableStorage;
        }
    }
}
