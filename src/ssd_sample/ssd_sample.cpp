#include <iostream>
#include <iomanip>

#include "deepeye_object_detector_gpu.h"
#include "deepeye_util.h"
#include "cm_image.h"

using namespace deepeye;

int main()
{
	static const int n_classes = 10;
	static const int channel = 3;
    static const int batch_size = 1;
	static const float nms = 0.45;

    auto *net = new ObjectDetector(n_classes, channel, batch_size, ObjectDetector::Arch::SSD512);

    try
	{
		net->load_weights("../../../models/ssd_10classes.deep");
	}
	catch (...)
	{
		cout << "[Error] failed to open file." << endl;
#ifdef _MSC_VER
		system("pause");
#endif
		exit(1);
	}

	// load image
	Image *x = Image::load_ssd_image("../../../img/ssd/002796.jpg", net->get_input_size());

    if (x == nullptr) {
		cout << "failed to open image file." << endl;
#ifdef _MSC_VER
		system("pause");
#endif
		exit(1);
	}

	// setting confidence threshold per classes
	std::vector<float> conf_list;
	for (int i = 0; i < n_classes; ++i) {
		conf_list.push_back(0.01);
	}

    // predict
    try
	{
		std::vector<SsdResult> y_ssd;
		net->predict(x->get_data(), &y_ssd, x->get_org_height(), x->get_org_width(), conf_list, nms);
		
		cout << "----------" << endl;
		for (SsdResult result : y_ssd) {
			cout << result._class << ", " << result._score << ", (" << (int)(result._x) << ", " << (int)(result._y) << ", " << (int)(result._w + result._x ) << ", " << (int)(result._h+ result._y) << ")" << endl;
		}
	}
	catch (...)
	{
		cout << "[Error] failed to open file." << endl;
		exit(1);
	}

	delete x;
	delete net;

#ifdef _MSC_VER
	system("pause");
#endif

    return 0;
}
