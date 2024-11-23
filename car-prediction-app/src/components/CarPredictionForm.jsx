import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';

const CarPredictionForm = () => {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    brand: '',
    model: '',
    milage: '',
    exteriorColor: '',
    interiorColor: '',
    beenInAccident: false,
    cleanTitle: false,
    hp: '',
    engineSize: '',
    cylCount: '',
    electric: false,
    hasTurbo: false,
    transmissionSpeeds: '',
    manual: false,
    automatic: false,
    yearOfCar: ''
  });

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    // Placeholder for API call
    try {
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulated API delay
      // Add your API call here
      console.log('Form data submitted:', formData);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h1 className="text-2xl font-bold mb-6">Car Price Prediction</h1>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Text/Number Inputs */}
          <div className="space-y-2">
            <label className="block text-sm font-medium">Brand</label>
            <input
              type="text"
              name="brand"
              value={formData.brand}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Model</label>
            <input
              type="text"
              name="model"
              value={formData.model}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Mileage</label>
            <input
              type="number"
              name="milage"
              value={formData.milage}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Exterior Color</label>
            <input
              type="text"
              name="exteriorColor"
              value={formData.exteriorColor}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Interior Color</label>
            <input
              type="text"
              name="interiorColor"
              value={formData.interiorColor}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Horsepower</label>
            <input
              type="number"
              name="hp"
              value={formData.hp}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Engine Size (L)</label>
            <input
              type="number"
              name="engineSize"
              value={formData.engineSize}
              onChange={handleInputChange}
              step="0.1"
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Cylinder Count</label>
            <input
              type="number"
              name="cylCount"
              value={formData.cylCount}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Transmission Speeds</label>
            <input
              type="number"
              name="transmissionSpeeds"
              value={formData.transmissionSpeeds}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Year of Car</label>
            <input
              type="number"
              name="yearOfCar"
              value={formData.yearOfCar}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
            />
          </div>
        </div>

        {/* Checkboxes */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              name="beenInAccident"
              checked={formData.beenInAccident}
              onChange={handleInputChange}
              className="rounded"
            />
            <span className="text-sm">Been in Accident</span>
          </label>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              name="cleanTitle"
              checked={formData.cleanTitle}
              onChange={handleInputChange}
              className="rounded"
            />
            <span className="text-sm">Clean Title</span>
          </label>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              name="electric"
              checked={formData.electric}
              onChange={handleInputChange}
              className="rounded"
            />
            <span className="text-sm">Electric</span>
          </label>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              name="hasTurbo"
              checked={formData.hasTurbo}
              onChange={handleInputChange}
              className="rounded"
            />
            <span className="text-sm">Has Turbo</span>
          </label>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              name="manual"
              checked={formData.manual}
              onChange={handleInputChange}
              className="rounded"
            />
            <span className="text-sm">Manual</span>
          </label>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              name="automatic"
              checked={formData.automatic}
              onChange={handleInputChange}
              className="rounded"
            />
            <span className="text-sm">Automatic</span>
          </label>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full mt-6 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-blue-400 flex items-center justify-center space-x-2"
        >
          {loading ? (
            <>
              <Loader2 className="animate-spin" size={20} />
              <span>Predicting...</span>
            </>
          ) : (
            <span>Predict Price</span>
          )}
        </button>
      </form>
    </div>
  );
};

export default CarPredictionForm;