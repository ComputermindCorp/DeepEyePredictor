#pragma once

#include <iostream>
#include <cstring>
#include <vector>
#include <map>
#include <memory>

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

struct SsdResult {
    int _class;
    float _score;
    float _x;
    float _y;
    float _w;
    float _h;
};

namespace deepeye {

    class DLL_CLASS ObjectDetector {
    public:
        enum Arch{
			SSD512,
        };
    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    
    public:
		ObjectDetector(int n_classes, int channel, int batch_size, Arch arch);
        ~ObjectDetector();

        void load_weights(string path);
        void predict(const float *x, std::vector<SsdResult> *y, int img_height, int img_widht, vector<float> conf_list, float nms);

        void set_weights_pathes(std::map<std::string, std::string> weight_pathes);
        std::map<std::string, std::string> get_weights_pathes();

        int get_n_classes();
        int get_channel();
        int get_batch_size();
        const int *get_input_size();
    };
}
