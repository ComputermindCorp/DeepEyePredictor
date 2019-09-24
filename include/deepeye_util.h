#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <float.h>

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

namespace deepeye
{
    class DLL_CLASS util {
    public:
        static int argmax(const float *x, int length)
        {
            float max = FLT_MIN;
            int max_index = 0;

            for(int i=0; i<length; i++)
            {
                if(x[i] > max){
                    max = x[i];
                    max_index = i;                    
                }
            }

            return max_index;
        }
    };

    class CertificationException : std::exception
    {
    private:
        std::string m_error;
    public:
        CertificationException( const std::string &str ) : m_error(str){}
        ~CertificationException(){}

        const char* what( void ) const noexcept { return m_error.c_str(); }
    };

    class DeepFileIOException : std::exception
    {
    private:
        std::string m_error;
    public:
        DeepFileIOException( const std::string &str ) : m_error(str){}
        ~DeepFileIOException(){}

        const char* what( void ) const noexcept { return m_error.c_str(); }
    };

    class DeepFileFormatException : std::exception
    {
    private:
        std::string m_error;
    public:
        DeepFileFormatException( const std::string &str ) : m_error(str){}
        ~DeepFileFormatException(){}

        const char* what( void ) const noexcept { return m_error.c_str(); }
    };
}
