#include <opencv2/opencv.hpp>
#include "cm_image.h"
#include <opencv2/highgui.hpp>

using namespace deepeye;

Image *Image::load(string path, const int size[2])
{
    cv::Mat im = cv::imread(path);
    if(im.data == nullptr){
        return nullptr;
    }

    if(im.channels() == 1){
	    cv::cvtColor(im, im, cv::COLOR_GRAY2RGB);
	}
	else{
	    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
	}

    if(size != nullptr){
        cv::resize(im, im, cv::Size(size[1], size[0]), 0, 0, cv::INTER_CUBIC);
    }

    Image *obj = new Image();

    obj->width = im.cols;
    obj->height = im.rows;
    obj->channel = im.channels();
    obj->length = obj->width * obj->height * obj->channel;
    
	// HWC -> CHW
	obj->data = new float[obj->length];
	for (int hei = 0; hei < obj->height; hei++) {
		for (int wid = 0; wid < obj->width; wid++) {
			for (int ch = 0; ch < obj->channel; ++ch) {
				obj->data[ch*obj->height*obj->width +hei*obj->width +wid] = (float)im.at<cv::Vec3b>(hei, wid)[ch];
			}
		}
	}
    
    return obj;
}

Image *Image::load_ssd_image(string path, const int size[2])
{
	cv::Mat im = cv::imread(path);
	if (im.data == nullptr) {
		return nullptr;
	}

	if(im.channels() == 1){
	    cv::cvtColor(im, im, cv::COLOR_GRAY2RGB);
	}
	else{
	    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
	}

	Image *obj = new Image();
	obj->org_width = im.cols;
	obj->org_height = im.rows;

	if (size != nullptr) {
		cv::resize(im, im, cv::Size(size[1], size[0]), 0, 0, cv::INTER_CUBIC);
	}
	

	obj->width = im.cols;
	obj->height = im.rows;
	obj->channel = im.channels();
	obj->length = obj->width * obj->height * obj->channel;

	obj->data = new float[obj->length];

	float mean[3] = { 103.939, 116.779, 123.68 };

	// HWC -> CHW
	for (int hei = 0; hei < obj->height; hei++) {
		for (int wid = 0; wid < obj->width; wid++) {
			for (int ch = 0; ch < obj->channel; ++ch) {
				obj->data[ch*obj->height*obj->width +hei*obj->width +wid] = (float)im.at<cv::Vec3b>(hei, wid)[ch] - mean[ch];
			}
		}
	}


	return obj;
}

float *Image::get_data()
{
    return this->data;
}