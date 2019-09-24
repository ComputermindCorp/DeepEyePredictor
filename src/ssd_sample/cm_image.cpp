#include <opencv2/opencv.hpp>
#include "cm_image.h"
#include<opencv2/highgui.hpp>

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

	std::stringstream ss;
	ss << "test" << ".png";
	cv::imwrite(ss.str(), im);

    Image *obj = new Image();

    obj->width = im.cols;
    obj->height = im.rows;
    obj->channel = im.channels();
    obj->length = obj->width * obj->height * obj->channel;
    
	// Chainer�p�̃��C���֓���邽�߁AChainer�p�ɓǂݍ��݉摜��ϊ�����BHWC��CHW
	// VGG16�́�255����K�v���Ȃ��B
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
	// Pillow�ł�RGB�œǂݍ���
	// OpenCV �� imread() ��BGR�œǂݍ���
	if(im.channels() == 1){
	    cv::cvtColor(im, im, cv::COLOR_GRAY2RGB);
	}
	else{
	    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
	}
	// Keras Application imagenet_utils.py _preprocess_numpy_input() ���A���̒��̏����ł�RGB��BGR���������Ă���
	// �܂�A���̂܂ܕϊ��������������ɁABGR�Ő��_���s���B
	// ...�Ǝv�������A�w�K����RGB�Ŋw�K���Ă���悤�Ȃ̂ŁARGB�ɕϊ����Đ��_���s��

	//std::cout << "load img size(col,row)=" << im.cols << "," << im.rows << std::endl;

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

	float mean[3] = { 103.939, 116.779, 123.68 };	// Keras Applications ���A�摜�̕��ϒl�����Z���Ă���

	// [H][W][C] �̏��Ŋi�[����Ă���B[C][H][W]�ɕϊ�����
	// SSD�͐��K���i��255�j�͂��Ȃ��Ă���
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