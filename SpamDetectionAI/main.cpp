#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include "FeatureExtractor.h"
#include "NeuralNetwork.h"
#include <filesystem>

using namespace SpamDetectionAI;
using namespace std;


struct  SMSData
{
	vector<string> texts;
	vector<int> labels; // 0: ham, 1: spam

};

class  SMSDataLoader
{
	public:
		static SMSData LoadSpamData(const string& filename)
		{
			SMSData data;
			ifstream file(filename);
			if (!file.is_open())
			{
				cerr << "Error opening file: " << filename << endl;
				return data;

			}

			string line;
			getline(file, line);

			while (getline(file, line))
			{
				stringstream stringStream(line);
				string label, text;

				getline(stringStream, label, ',');
				getline(stringStream, text);

				if (!text.empty() && text.front()=='"')
						text = text.substr(1, text.size() - 2);



				data.texts.push_back(text);
				data.labels.push_back(label == "spam" ? 1 : 0);
						
			}

			file.close();
			return data;
		}

		static void PrintDataStatics(const SMSData& data)
		{
			// __int64 hatasý alýrsan  static cast ekle 67 line 
			int spamCount = count(data.labels.begin(), data.labels.end(), 1);
			int hamCount = static_cast<int>(data.labels.size()) - spamCount;
			cout << "**** Dataset Statistics****" << endl;
			cout << "Total messages: " << data.labels.size() << endl;
			cout << "Spam messages: " << spamCount << endl;
			cout << "Ham messages: " << hamCount << endl;
			cout << fixed << setprecision(2) << "Spam ratio: " << (data.labels.empty() ? 0.0 : (static_cast<double>(spamCount) / data.labels.size()) * 100) << "%" << endl;
		}
		

};


class  ModelTrain
{
	private:

		static void CalculateMatrics(const NeuralNetwork& model, const vector<vector<double>>& X, const vector<double>& y)
		{
			int TP = 0, TN = 0, FP = 0, FN = 0; // TP: True Positives, TN: True Negatives, FP: False Positives, FN: False Negatives

			for (size_t i = 0; i < X.size(); i++)
			{
				double prediction = model.Predict(X[i]);
				int predictedLabel = prediction >= 0.5 ? 1 : 0; // 0.5 eþik deðeri
				int actualLabel = static_cast<int>(y[i]);

				if (predictedLabel == 1 && actualLabel == 1) ++TP;
				else if (predictedLabel == 0 && actualLabel == 0) ++TN;
				else if (predictedLabel == 1 && actualLabel == 0) ++FP;
				else if (predictedLabel == 0 && actualLabel == 1) ++FN;
			}
			double accuracy = (TP + TN) / static_cast<double>(X.size()); // Doðruluk : Doðru tahminlerin toplam tahminlere oraný

			double precision = TP + FP == 0 ? 0 : static_cast<double>(TP) / (TP + FP); // Kesinlik : Gerçek pozitiflerin ne kadarýnýn doðru tahmin edildiði

			double recall = TP + FN == 0 ? 0 : static_cast<double>(TP) / (TP + FN); // Duyarlýlýk : Gerçek pozitiflerin ne kadarýnýn doðru tahmin edildiði

			double f1Score = precision + recall == 0 ? 0 : 2 * (precision * recall) / (precision + recall) ; // F1 Skoru: Kesinlik ve duyarlýlýðýn harmonik ortalamasý

			cout << "**** Model Evaluation Metrics ****" << endl;

			cout << "Accuracy:" << fixed << setprecision(4) << accuracy * 100 << "%" << endl; // Accuracy: Doðru tahminlerin toplam tahminlere oraný
			cout << "Precision:" << fixed << setprecision(4) << precision * 100 << "%" << endl; // Precision: Gerçek pozitiflerin ne kadarýnýn doðru tahmin edildiði
			cout << "Recall:" << fixed << setprecision(4) << recall * 100 << "%" << endl; // Recall: Gerçek pozitiflerin ne kadarýnýn doðru tahmin edildiði: *100: Yüzde olarak ifade etmek için

			cout << "F1 Score:" << fixed << setprecision(4) << f1Score * 100 << "%" << endl; // F1 Score: Kesinlik ve duyarlýlýðýn harmonik ortalamasý

			cout << "Confusion Matrix:" <<" True Positive= "<<TP <<" True Negative= "<< TN <<" False Positive= " <<FP <<" False Negative= "<< FN << endl;
 		}

		static void  ExampleTests(const NeuralNetwork& neuralNetwork, const FeatureExtractor& extractor)
		{
			cout << "**** Example Predictions ****" << endl;

			vector<pair<string,int>> examples = {
				{"Congratulations! You've won a free ticket to the Bahamas! Call now!", 1},
				{"Hey, are we still on for dinner tonight?", 0},
				{"URGENT: Your account has been compromised. Click here to reset your password.", 1},
				{"Can you send me the report by tomorrow?", 0},
				{"Win a brand new car by entering our sweepstakes! Text WIN to 12345.", 1},
				{"Don't forget about the meeting at 3 PM today.", 0}
			};

			for (const auto& [text,actual] : examples)
			{
				auto features = extractor.ExtractAdvancedFeatures(text); // Metni özellik vektörüne dönüþtür bu özellikler, modelin tahmin yaparken kullanacaðý sayýsal temsili saðlar. Örneðin, kelime sýklýðý, mesaj uzunluðu, büyük harf kullanýmý gibi özellikler içerebilir.
				double prediction = neuralNetwork.Predict(features); // 0.5 eþik deðeri bu kod ile tahmin yaparken kullanýlýr. Eðer tahmin deðeri 0.5 veya daha yüksekse, mesajýn spam olduðu varsayýlýr; aksi takdirde ham (normal) olarak kabul edilir.
				string predictedLabel = prediction >= 0.5 ? "Spam" : "Ham"; 

				string actualLabel = actual == 1 ? "Spam" : "Ham"; // Gerçek etiket, 1 ise "Spam", 0 ise "Ham" olarak ifade edilir.

				cout << "\nMessages: " << text << endl;
				cout << "Spam Probability: " << fixed << setprecision(4) << prediction * 100 << "%" << endl; // Tahmin edilen spam olasýlýðý, yüzde olarak ifade edilir.
				cout << "Predicted Label: " << predictedLabel << " [!] Actual Label: " << actualLabel << endl;
				 
				if (predictedLabel==actualLabel)  cout<< "Prediction is correct." << endl;
				else cout << "Prediction is incorrect." << endl; // Tahminin doðruluðu, modelin tahmin ettiði etiket ile gerçek etiket karþýlaþtýrýlarak deðerlendirilir. Eðer tahmin edilen etiket gerçek etikete eþitse, tahmin doðru olarak kabul edilir; aksi takdirde yanlýþ olarak deðerlendirilir.
				cout << "-----------------------------------" << endl;
				
			}
		}

	public:

		static void TrainAndEvulate()
		{
			cout << "**** Spam Detection AI on working ****" << endl;
			cout << string(40, '-') << endl; // 40 adet tire karakteri ile görsel bir ayýrýcý oluþturulur.

			std::cout << "Working Directory: "
				<< std::filesystem::current_path() << std::endl;

			//1. Load and Analyze Data
			auto data = SMSDataLoader::LoadSpamData("*\\DATA\\spam.csv");
			if (data.texts.empty())
			{
				cerr << "Failed Load DATA!";
				return;
			}

			SMSDataLoader::PrintDataStatics(data);

			//2. Feature Extraction

			FeatureExtractor extractor; 
			extractor.BuildVocabulary(data.texts); 

			vector<vector<double>> features;

			for (const auto& text: data.texts)
			{
				features.push_back(extractor.ExtractAdvancedFeatures(text)); 
			}
			int featureSize = extractor.GetFeatureSize();
			cout << "Feature size: " << featureSize << endl;
			cout << "Vocabulary size: " << featureSize - 5 << endl; // 5 ek özellik olduðu varsayýlarak	

			//3. Train-Test Split

			random_device rd; 
			mt19937 g(rd()); // Rastgele sayý üreteci, verilerin karýþtýrýlmasý için kullanýlýr. mt19937, Mersenne Twister algoritmasýný kullanan bir rastgele sayý üreteci sýnýfýdýr. rd() ile tohumlanýr, böylece her çalýþtýrmada farklý bir sýralama elde edilir.
			vector<size_t> indices(data.texts.size());
			for (size_t i = 0; i < indices.size(); ++i) indices[i] = i; 
			shuffle(indices.begin(), indices.end(), g); // Verilerin rastgele karýþtýrýlmasý için shuffle algoritmasý kullanýlýr. Bu, modelin eðitim ve test verilerini rastgele seçmesini saðlar, böylece modelin genelleme yeteneði artýrýlýr.

			int splitIndex = static_cast<int>(features.size() * 0.8); // Verilerin %80'i eðitim, %20'si test için ayrýlýr.


			vector<vector<double>> X_train, X_test;
			vector<double> y_train, y_test;

			// Eðitim verileri için ilk %80'lik kýsmý kullanýlýr.
			for (int  i = 0; i < splitIndex; i++)
			{
				X_train.push_back(features[indices[i]]);
				y_train.push_back(data.labels[indices[i]]);
			}
			// Test verileri için kalan %20'lik kýsmý kullanýlýr.
			for (int i = splitIndex; i < features.size(); i++)
			{
				X_test.push_back(features[indices[i]]);
				y_test.push_back(data.labels[indices[i]]); 
			}


			cout << "Training samples: " << X_train.size() << endl;
			cout << "Testing samples: " << X_test.size() << endl;

			//4. Train Neural Network



			vector<int> architecture = { featureSize, 16, 8, 1 }; // Giriþ katmaný, iki gizli katman ve çýkýþ katmaný

			NeuralNetwork neuralNetwork(architecture, 0.01); // Öðrenme oraný 0.01 olarak belirle 

			 

			auto start = chrono::high_resolution_clock::now(); // Eðitim süresini ölçmek için baþlangýç zamanýný kaydet

			cout <<  "Training the model..." << endl;
			cout << string(40, '-') << endl;
			neuralNetwork.Train(X_train, y_train, 50); 


			auto end = chrono::high_resolution_clock::now(); // Eðitim süresini ölçmek için bitiþ zamanýný kaydet
			auto  duration = chrono::duration_cast<chrono::seconds>(end - start);



			cout << string(40, '-') << endl;
			cout << "Training completed in " << duration.count() << " seconds." << endl;


			// Deðerlendirme metriklerini hesapla ve yazdýr

			double testAccuracy = neuralNetwork.Evulate(X_test, y_test);
			double trainAccuracy = neuralNetwork.Evulate(X_train, y_train);


			cout << " Training Results: " << endl;	

			cout << "Train Accuracy: " << fixed << setprecision(4) << trainAccuracy * 100 << "%" << endl; // Eðitim doðruluðu, modelin eðitim verileri üzerindeki performansýný gösterir. Yüzde olarak ifade edilir.

			cout << "Test Accuracy: " << fixed << setprecision(4) << testAccuracy * 100 << "%" << endl; // Test doðruluðu, modelin test verileri üzerindeki performansýný gösterir. Yüzde olarak ifade edilir.


			CalculateMatrics(neuralNetwork, X_test, y_test); // Modelin test verileri üzerindeki performansýný deðerlendirmek için çeþitli metrikler hesaplanýr ve yazdýrýlýr. Bu metrikler arasýnda doðruluk (accuracy), kesinlik (precision), duyarlýlýk (recall), F1 skoru ve karýþýklýk matrisi (confusion matrix) bulunur.

			// Example tests with new messages

			ExampleTests(neuralNetwork, extractor); // Modelin yeni mesajlar üzerindeki tahmin performansýný göstermek için örnek testler yapýlýr. Bu testlerde, modelin tahmin ettiði etiketler (spam veya ham) gerçek etiketlerle karþýlaþtýrýlýr ve tahminlerin doðruluðu deðerlendirilir.


			// Save 
			neuralNetwork.SaveModel("spam_detection_model.dat"); // Eðitilmiþ modelin aðýrlýklarýný ve yapýlandýrmasýný "spam_detection_model.dat" adlý bir dosyaya kaydeder. Bu, modelin daha sonra yüklenip kullanýlabilmesi için önemlidir.
			
		}
		


};


int main()
{
	try
	{
		ModelTrain::TrainAndEvulate(); // Modelin eðitilmesi ve deðerlendirilmesi iþlemini baþlatýr. Bu fonksiyon, veri yükleme, özellik çýkarma, model eðitimi ve deðerlendirme gibi tüm adýmlarý içerir.
	}
	catch (const  exception& ex)
	{
		cerr << "An error occurred: " << ex.what() << endl; // Herhangi bir istisna durumunda, hata mesajýný yakalar ve ekrana yazdýrýr.
	}

	cout << "**** Program finished ****" << endl;
	cin.get(); 
}
 

 