#pragma once

#ifdef _MSC_VER
#include <string>
#else
#include <cstring>
#endif

using namespace std;

namespace deepeye {
    class Image{
    private:
        int width;
        int height;
        int channel;
        int length;
        //unsigned char *data;
        float *data = nullptr;

		int org_width;
		int org_height;
    public:
        Image() {};
        ~Image()
        {
            delete[] this->data;
        };

        float *get_data();

		int get_org_width() { return org_width; }
		int get_org_height() { return org_height; }

        static Image *load(string path, const int size[2]);
		static Image *load_ssd_image(string path, const int size[2]);
    };
}