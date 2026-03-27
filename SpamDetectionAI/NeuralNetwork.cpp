#include "NeuralNetwork.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <stdexcept> 
#include <numeric> 

using namespace std;
namespace SpamDetectionAI
{
	HiddenLayer:: HiddenLayer(int inputSize, int outputSize)
	{
		m_weights.resize(outputSize, vector<double>(inputSize));
		m_biases.resize(outputSize, 0.0);
		m_outputs.resize(outputSize, 0.0);
		

		// Xavier Initialization (Glorot Initialization) kullanarak aðýrlýklarý baþlat Bu yöntem, aðýrlýklarýn baþlangýçta çok büyük veya çok küçük olmamasýný saðlar ve genellikle derin sinir aðlarýnda daha iyi performans gösterir.

		mt19937 rng(random_device{}());

		double limit = sqrt(6.0 / (inputSize + outputSize)); // Xavier Initialization için limit hesaplama
		uniform_real_distribution<double> dist(-limit, limit); 
		
		for (auto& row: m_weights)
		{
			for(auto& w: row)
			{
				w = dist(rng); // Aðýrlýklarý -limit ile limit arasýnda rastgele baþlat
			}

		}
	}

	vector<double> HiddenLayer::Forward(const vector<double>& input)
	{
		m_inputs = input; 
		int outputSize = m_weights.size();
		int inputSize = m_weights[0].size();

		m_outputs.resize(outputSize);

		for (int i = 0; i < outputSize; ++i)
		{
			double sum = m_biases[i];
			for (int j = 0; j < inputSize; ++j) {

				sum += m_weights[i][j] * input[j]; // Aðýrlýklý toplam hesaplama

				m_outputs[i] = ActivationFunction::ReLU(sum); // ReLU aktivasyon 
			} 
		}return m_outputs; 
	}

	vector<double> HiddenLayer::Backward(const vector<double>& gradOutput, double learningRate)
	{
		int outputSize = static_cast<int>(m_weights.size());
		int inputSize = static_cast<int>(m_inputs.size());


		vector<double>  gradInput(inputSize, 0.0);

		for (int i = 0; i < outputSize; i++)
		{
			// ReLU'nun türev: output > 0 ise 1, deðilse 0

			double delta = gradOutput[i] * ActivationFunction::ReLUDerivative(m_outputs[i]); // Çýkýþ hatasý ile ReLU'nun türevi çarpýlýr

			// Aðýrlýk güncellemesi

			for (int j = 0; j < inputSize; j++)
			{
				gradInput[j] += m_weights[i][j] * delta;
				m_weights[i][j] -= learningRate * delta * m_inputs[j];

			}

			m_biases[i] -= learningRate * delta;


		}return gradInput;

	};

	void NeuralNetwork::InitializeWeights(int inputSize
	,int outputSize ,vector<vector<double>>& weights,vector<double>&biases)
	{
		weights.resize(outputSize, vector<double>(inputSize));
		biases.resize(outputSize, 0.0);

		mt19937 rng(random_device{}());
		double limit = sqrt(6.0 / (inputSize + outputSize));
		uniform_real_distribution<double> dist(-limit, limit);


		for (auto& row : weights)
			for (auto& weight : row)
				weight = dist(rng);
	}


	NeuralNetwork::NeuralNetwork(const vector<int>& arch, double learningRate) : m_learningRate(learningRate)
	{
		for (int  i = 0; i < static_cast<int>(arch.size())-2; i++)
		{
			m_hiddenLayers.push_back(
				make_unique<HiddenLayer>(arch[i], arch[i + 1]
			));

		}

		int lastHidden = arch[arch.size() - 2];
		m_outputWeights.resize(lastHidden, 0.0);
		m_outputBias = 0.0;


		mt19937 rng(random_device{}());
		double limit = sqrt(6.0 / (lastHidden+1));
		uniform_real_distribution<double> dist(-limit, limit);
	
		for (auto& weight : m_outputWeights)
					weight = dist(rng);
	}


	double NeuralNetwork::Predict(const vector<double>& features) const
	{
		vector<double> current = features;
		for (const auto& layer : m_hiddenLayers)
			current = layer->Forward(current);


		//Sigmoid Fonksiyonu

		double sum = m_outputBias;
		for (int i = 0; i < static_cast<int>(m_outputWeights.size()); i++)
			sum += m_outputWeights[i] * current[i];

		return ActivationFunction::Sigmoid(sum);

		
	}

	void NeuralNetwork::Train(const vector<vector<double>>& X,
		const vector<double>& y,
		int epochs,
		bool verbose)

	{
		for (int  epoch = 0; epoch < epochs; epoch++)
		{
			double totalLoss = 0.0;
			
			//Forward Pass
			for (size_t idx = 0; idx < X.size(); idx++)
			{
				vector<double> current = X[idx];
				for (auto& layer : m_hiddenLayers)
					current = layer->Forward(current);
			
			// Output Layer
			
				double rawOutput = m_outputBias;
				for (int i = 0; i < static_cast<int>(m_outputWeights.size()); i++)
				{rawOutput += m_outputWeights[i] * current[i];}
				

				double predicted = ActivationFunction::Sigmoid(rawOutput);
				double actual = y[idx];



				// Binary cross-entropy loss

				double eps = 1e-7; 

				totalLoss += -(actual * log(predicted + eps) + (1 - actual) * log(1 - predicted + eps));



				double outputGrad = predicted - actual;

				for (int i = 0; i <static_cast<int>(m_outputWeights.size()); i++)
				{
					double grad = outputGrad * current[i];
					m_outputWeights[i] -= m_learningRate * grad;

				}
				m_outputBias -= m_learningRate * outputGrad;


				vector<double> gradBack(m_outputWeights.size());
				for (int  i = 0; i < static_cast<int>(m_outputWeights.size()); i++)
				{
					gradBack[i] = outputGrad * m_outputWeights[i];

				}
				for (int i = static_cast<int>(m_hiddenLayers.size()) - 1; i >= 0; i--)
					gradBack = m_hiddenLayers[i]->Backward(gradBack, m_learningRate);
			}

			if (verbose && (epoch + 1) % 10 == 0)
			{
				double avgLoss = totalLoss / X.size();

				cout << "Epochs:" << epoch + 1 << "/" << epochs << "Loss:" << fixed << setprecision(4) << avgLoss << endl;
			}

		}

		

	}

	double NeuralNetwork::Evulate(const vector<vector<double>>& X, const vector<double>& y) const
	{
		int correct = 0;
		for (size_t i = 0; i < X.size(); i++)
		{
			double pred = Predict(X[i]);
			int predLabel = pred >= 0.5 ? 1 : 0;
			if (predLabel == static_cast<int>(y[i])) ++correct;

		} return static_cast<double>(correct) / X.size();
		
	}

	void NeuralNetwork::SaveModel(const string& filename) const
	{
		ofstream file(filename, ios::binary);

		if (!file.is_open())
		{
			throw runtime_error("Cannot open file for saving" + filename);
		}


		int owSize = static_cast<int>(m_outputWeights.size());
		file.write(reinterpret_cast<const char*>(&owSize), sizeof(int));
		file.write(reinterpret_cast<const char*>(m_outputWeights.data()),
			owSize * sizeof(double));

		file.write(reinterpret_cast<const char*>(&m_outputBias), sizeof(double));



		//Hidden 

		int layerCount = static_cast<int>(m_hiddenLayers.size());
		file.write(reinterpret_cast<const char*>(&layerCount), sizeof(int));


		for (const auto& layer : m_hiddenLayers)
		{
			const auto& weights = layer->GetWeights();
			const auto& biases = layer->GetBiases();

			int rows = static_cast<int>(weights[0].size());
			int cols = rows > 0 ? static_cast<int>(weights[0].size()) : 0;

			file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
			file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

			for (const auto& row : weights)
				file.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(double));
			file.write(reinterpret_cast<const char*>(biases.data()), rows * sizeof(double));

		} cout << "Model saved to:" << filename << endl;

	}
	void NeuralNetwork::LoadModel(const string& filename)
	{
		ifstream file(filename, ios::binary);

		if (!file.is_open())
		{
			throw runtime_error("Cannot open file for loading" + filename);
		}

		int owSize;
		file.read(reinterpret_cast<char*>(&owSize), sizeof(int));
		m_outputWeights.resize(owSize);
		file.read(reinterpret_cast<char*>(m_outputWeights.data()), owSize * sizeof(double));
		file.read(reinterpret_cast<char*>(&m_outputBias), sizeof(double));

		int layerCount;
		file.read(reinterpret_cast<char*>(&layerCount), sizeof(int));
		m_hiddenLayers.clear();
		for (int i = 0; i < layerCount; i++)
		{
			int rows, cols;
			file.read(reinterpret_cast<char*>(&rows), sizeof(int));
			file.read(reinterpret_cast<char*>(&cols), sizeof(int));
			auto layer = std::make_unique<HiddenLayer>(cols, rows);
			auto& weights = layer->GetWeights();
			auto& biases = layer->GetBiases();
			for (auto& row : weights)
				file.read(reinterpret_cast<char*>(row.data()), cols * sizeof(double));
			file.read(reinterpret_cast<char*>(biases.data()), rows * sizeof(double));
			m_hiddenLayers.push_back(std::move(layer));

		} cout << "Model Loaded from:" << filename << endl;
	}

	
}
