#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include "deepeye_image_classifier_gpu.h"
#include "deepeye_util.h"
#include "cm_image.h"

using namespace deepeye;

void show_result(float *pred_prob, int n_classes, int pred)
{
    std::cout << std::fixed;
    std::cout << "[0] NG ... " << std::setprecision(5) << pred_prob[0] << std::endl;
    std::cout << "[1] OK ... " << std::setprecision(5) <<pred_prob[1] << std::endl;
    std::cout << "pred: [" << pred << ": ";
    if(pred == 0){
        std::cout << "NG]";
    }
    else if(pred==1){
        std::cout << "OK]";
    }

    std::cout << endl;
}

void classification()
{
    static const int n_classes = 2;
    static const int channel = 3;
    static const int batch_size = 1;

    auto *googlenet = new ImageClassifier(n_classes, channel, batch_size, ImageClassifier::Arch::GoogLeNet);

    try
    {        
        googlenet->load_weights("../../../models/googlenet_lego.deep");
    }
    catch(...)
    {
        cout << "failed to load weight file." << endl;
        exit(1);
    }

    std::ifstream ifs("../../../img/classification/image.txt");
    std::string img_path;
   
    if (ifs.fail())
    {
        std::cerr << "Failed to open image.txt" << std::endl;
    }
    float pred_prob[n_classes];
    int pred;

    while (getline(ifs, img_path))
    {
        Image *x = Image::load(img_path, googlenet->get_input_size());

        if(x == nullptr){
            cout << "failed to open image file. [" << img_path << "]" << endl;
            continue;
        }

        //----- predict OK data
        std::cout << "---- " << img_path << " ----" << std::endl;
        //-- GoogLeNet
        std::cout << "[GoogLeNet]" << std::endl;
        googlenet->predict_prob(x->get_data(), pred_prob);
        pred = googlenet->predict(x->get_data()) ;
        show_result(pred_prob, n_classes, pred);
        std::cout << std::endl;

        delete x;
    }

    delete googlenet;
}

int main()
{
    // classification
    std::cout << "[CLASSIFICATION]" << std::endl;
    classification();

#ifdef _MSC_VER
	system("pause");
#endif

    return 0;
}
