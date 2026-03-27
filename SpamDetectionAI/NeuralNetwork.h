#pragma once 
#include <vector>
#include <memory>
#include <random>


namespace SpamDetectionAI 
{
	class ActivationFunction
	{
		public:
			static double Sigmoid(double x) {
				return 1.0 / (1.0 + std::exp(-x)); // Sigmoid fonksiyonu, genellikle çýktý katmanýnda kullanýlýr ve çýktýyý 0 ile 1 arasýnda sýkýþtýrýr.
			}
			static double SigmoidDerivative(double x) { return Sigmoid(x) * (1 - Sigmoid(x)); } // Sigmoid fonksiyonunun türevi
			static double ReLU(double x) { return std::max(0.0, x); } // ReLU fonksiyonu, genellikle gizli katmanlarda kullanýlýr ve negatif deðerleri sýfýrlar.
			static double ReLUDerivative(double x) { return x > 0 ? 1.0 : 0.0; } // ReLU fonksiyonunun türevi

	};


	//  Sinir aðý GÝZLÝ KATMANLARINDA ReLU, ÇIKTI KATMANINDA Sigmoid kullanacaðýz.
	class HiddenLayer
	{
		private:
			std::vector<std::vector<double>> m_weights; 
			std::vector<double>m_biases;
			std::vector<double>m_outputs;
			std::vector<double>m_inputs;

		public:
			HiddenLayer(int inputSize, int outputSize);
			std::vector<double> Forward(const std::vector<double>& input);
			std::vector<double>Backward(const std::vector<double>& gradOutput, double learningRate);

			std::vector<std::vector<double>>& GetWeights() { return m_weights; }
			std::vector<double>& GetBiases() { return m_biases; }

			
	};

	class  OutputLayer
	{
	private:
		std::vector<std::vector<double>> m_weights;
		double m_biases;
		double m_outputs;
		std::vector<double>m_inputs;

	};
	
	// MLP (Multi-Layer Perceptron) 

	class NeuralNetwork
	{
		private:
			std::vector<std::unique_ptr<HiddenLayer>> m_hiddenLayers;
			std::vector<double> m_outputWeights;
			double m_outputBias;
			double m_learningRate;

			void InitializeWeights(int inputSize, int outputSize, std::vector<std::vector<double>>& weights, std::vector<double>& biases); // Aðýrlýklarý ve biaslarý rastgele baþlatmak için yardýmcý fonksiyon

		public:
			NeuralNetwork(const std::vector<int>& arch, double learningRate = 0.01);

			double Predict(const std::vector<double>& features) const; 
			void Train(const std::vector<std::vector<double>>& X,
				const std::vector<double>& y, 
				int epochs, 
				bool verbose=true);

			double Evulate(const std::vector<std::vector<double>>& X,
				const std::vector<double>& y) const;


			void SaveModel(const std::string& filename) const;
			void LoadModel(const std::string& filename);


	};
}
 