#pragma once

#include <iostream>
#ifdef _MSC_VER
#include <string>
#else
#include <cstring>
#endif
#include <vector>
#include <map>
#include <memory>
#include "deepeye_util.h"

using namespace std;

#ifdef _MSC_VER
#ifdef deepeye_predictor_gpu_EXPORTS
#define DLL_CLASS __declspec(dllexport)
#else
#define DLL_CLASS __declspec(dllimport)
#endif
#else
#define DLL_CLASS
#endif

namespace deepeye {

    class DLL_CLASS ImageClassifier{
    public:
        enum Arch{
            GoogLeNet,
            VGG16,
			MobileNet,
			ResNet50,
        };
    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    
    public:
        ImageClassifier(int n_classes, int channel, int batch_size, Arch arch, float alpha=1.0);
        ~ImageClassifier();

        void load_weights(string path);
        void predict_prob(const float *x, float *pred_prob, bool normalization=true);
        int predict(const float *x, bool normalization=true);
		void cout_info();

        void set_weights_pathes(std::map<std::string, std::string> weight_pathes);
        std::map<std::string, std::string> get_weights_pathes();

        int get_n_classes();
        int get_channel();
        int get_batch_size();
        const int *get_input_size();
    };
}

