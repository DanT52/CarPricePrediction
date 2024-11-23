import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';
import axios from 'axios';

const CarPredictionForm = () => {
  const [loading, setLoading] = useState(false);
  const [predictedPrice, setPredictedPrice] = useState(null);
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
    setPredictedPrice(null);

    const payload = {
      brand: formData.brand,
      model: formData.model,
      milage: parseFloat(formData.milage),
      ext_col: formData.exteriorColor,
      int_col: formData.interiorColor,
      accident: formData.beenInAccident ? 1 : 0,
      clean_title: formData.cleanTitle ? 1 : 0,
      hp: parseFloat(formData.hp),
      L: parseFloat(formData.engineSize),
      cyl_count: parseInt(formData.cylCount, 10),
      electric: formData.electric ? 1 : 0,
      turbo: formData.hasTurbo ? 1 : 0,
      trans_speed: parseInt(formData.transmissionSpeeds, 10),
      manual: formData.manual ? 1 : 0,
      automatic: formData.automatic ? 1 : 0,
      model_year: parseInt(formData.yearOfCar, 10)
    };

    try {
      const response = await axios.post('http://localhost:8000/predict', payload);
      setPredictedPrice(response.data.predicted_price);
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

      {predictedPrice && (
        <div className="mt-6 p-4 bg-green-100 text-green-800 rounded-lg">
          <h2 className="text-xl font-bold">Predicted Price: {predictedPrice}</h2>
        </div>
      )}
    </div>
  );
};

export default CarPredictionForm;