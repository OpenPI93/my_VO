#pragma once

#include <Eigen/Core>
#include <exception>
#include <string>
#include <vector>
#include <iostream>
//#include <opencv.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <thread>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using Sophus::SE3d;

#define __in__
#define __out__
#define __inout__

#define THREADING_POOL


/*
�ڱ�ϵͳ�У�point2d�ĸ�ʽΪ(x, y)��point3d�ĸ�ʽΪ(x, y, z)
*/

typedef Eigen::Vector2d point2d;
typedef Eigen::Vector3d point3d;
typedef vector<point3d> line3d;

class param_exception : public std::exception {
public:
    param_exception(__in__ const string& info): exception(info.c_str()) {}
private:
    param_exception() = delete;
};

/*
* it is use for algorithms that use OpenCV's cv::Mat to process image, which will be throw when the point is out of the boundary.
*/
class point_out_of_boundary : public std::exception {
public:
    point_out_of_boundary(double x, double y, int col = 0, int row = 0) {
        string message;
        if (x < 0 || x > col) {
            message += "x = " + std::to_string(x) + "is out of boundary!  ";
        }
        if (y < 0 || y > row) {
            message += "y = " + std::to_string(y) + "is out of boundary!  ";
        }
        message += "\n";
        __std_exception_data _InitData = { message.c_str(), true };
        __std_exception_copy(&_InitData, &_Data);
    }
    virtual char const* what() const
    {
        return _Data._What ? _Data._What : "Unknown exception";
    }

private:

    __std_exception_data _Data;
};

namespace clvo {

    class mappoint {
        mappoint() {}
        /*
        Video-based, Real-Time Multi View Stereo
        param:
        x       ��ͼ������
        y       ��ͼ������
        z       ��ͼ���
        sigma   ���Э���������������˹�ֲ�
        max      ��������Ҳ࣬�������������ȷֲ�
        min    ���������࣬�������������ȷֲ�
        mu      ��˹-���ȷֲ���
        */
        mappoint(double x, double y, double z, double sigma, double min, double max, double mu = 1);
    };

    class keypoint {
    public:
        keypoint() { next = head = nullptr; }
        keypoint(int x, int y) { pt = point2d(x, y); harrisScore = 0; next = head = nullptr; }
        keypoint(point2d& _pt) :pt(_pt) {}
        keypoint& operator = (keypoint& kp) { 
            pt = kp.pt; angle = kp.angle; harrisScore = kp.harrisScore;
            for (int i = 0; i < 8; ++i)
                descripter[i] = kp.descripter[i];
            return *this;
        }
        point2d pt;
        //��¼����һ֡��ƥ����Ϣ
        keypoint* next;
        //��¼�Ĵ�������ĳ�ʼ��
        keypoint* head;
        //��¼������������֡�����
        long long index;
        
        unsigned long descripter[8];
        //����洢���ǻ��ȣ����ڼ���cos��sin
        double angle;
        double harrisScore;
    };

    //��ά����
    class line2d {
    public:
        line2d() {}
        void push_back(point2d& pt) { points.push_back(pt); }
        keypoint* addKeypoint(keypoint& kp) { kps.push_back(kp); return &kp; }
        int size()const { return points.size(); }
        vector<Eigen::Vector2d> points;
        vector<keypoint> kps;

    };

    /*
    ���ݰ�����
    camera: �۲��Ӧ�����
    point:  �۲��Ӧ�ĵ�ͼ�㣨�ؼ�֡��ǰһ֡�����������ά���꣩
    obs:    ��ǰ֡�����ά����
    */
    class observation {
    public:
        observation() {}
        observation(int _camera, int _point, point3d& _obs) : camera(_camera), point(_point), obs(_obs){}
        int camera;
        int point;
        point3d obs;
    };

    class usefulTool {
    public:
        /*
        * ��Ŀ��ͼ����н�����
        * @param    src ����ͼ��
        * @param    size ����������
        * @return   ���������
        * @throws   param_exception if the type of src is not CV_8UC1 or size is lower than 1.0
        */
        static cv::Mat DownSampling(__in__ const cv::Mat& src, __in__ const double size) {
            const int row = (int)(src.rows / size);
            const int col = (int)(src.cols / size);

            if (size == 1.0)return src.clone();

            if (src.type() != CV_8UC1) {
                cout << "the type of sec is not CV_8UC1\n";
                throw param_exception("the type of src is not CV_8UC1");
            }
            if (size < 1.0) {
                cout << "the parameter size is illegal\n";
                throw param_exception("size is lower than 1.0");
            }

            cv::Mat des(row, col, CV_8UC1);
            for (int i = 0; i < row; ++i) {
                for (int j = 0; j < col; ++j) {
                    int cur_row = (int)(i * size);
                    int cur_col = (int)(j * size);

                    int temp = 0;
                    if (src.cols == cur_col - 1 || src.rows == cur_row - 1)des.data[i * col + j] = src.at<uchar>(cur_row, cur_col);
                    else temp = src.at<uchar>(cur_row, cur_col) + src.at<uchar>(cur_row + 1, cur_col) + src.at<uchar>(cur_row, cur_col + 1) + src.at<uchar>(cur_row + 1, cur_col + 1);
                    des.data[i * col + j] = (uchar)(temp >> 2);
                }
            }

            return des;
        }
        /*
        * �������ز�ֵ
        * @param    img ����ͼ��
        * @param    x ���غ�����
        * @param    y ����������
        * @return   ��ֵ�������ֵ
        * @throw    param_exception if the coordinate is out of boundary
        */
        static inline float getPixelValue(const cv::Mat& img, double x, double y) {

            if (img.rows == 0)throw("bad input of image\n ");

            if (std::isnan(x) || std::isnan(y)) {
                throw param_exception("the param x or y is not a number");
            }

            if (x < 0 || x > img.cols || y < 0 || y > img.rows)
                throw param_exception("the param x or y is out of boundary");

            uchar *data = &img.data[int(y) * img.step + int(x)];
            float xx = x - floor(x);
            float yy = y - floor(y);
            return float(
                (1 - xx) * (1 - yy) * data[0] +
                xx * (1 - yy) * data[1] +
                (1 - xx) * yy * data[img.step] +
                xx * yy * data[img.step + 1]
                );
        }
        /*
        * ���ڶ��ֱ�ӷ���������ĳ��Ϊ���ĵ����ؿ�ĸ�˹��Ȩ��ֵ
        * @param    img ����ͼ��
        * @param    x ���غ�����
        * @param    y ����������
        * @return   ��ֵ�������ֵ
        * @throw    param_exception if the coordinate is out of boundary
        */
        static inline float getGaussPitchValue(const cv::Mat& img, double x, double y) {
            if (x < 3 || x > img.cols - 3 || y < 3 || y > img.rows - 3)
                return getPixelValue(img, x, y);

            double sum = 3 * getPixelValue(img, x, y);
            for (int i = 0; i < 9; ++i) {
                sum += (2 >> (i % 2)) * getPixelValue(img, x - 1 + i / 3, y - 1 + i % 3);
            }

            return sum / 16.0;
        }
    };

    class groundTruth {
    public:
        groundTruth(long double time, point3d&& pos) :timestamp(time), position(pos){}
        long double timestamp;
        point3d position;
    };
}

