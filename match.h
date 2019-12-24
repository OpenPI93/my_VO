#pragma once
#include "basetype.h"
#include "frame.h"
#include "epnp.h"
#include <list>
using std::list;


#ifdef THREADING_POOL

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <memory>
#include "pqueue.h"
union maininfo{
    maininfo() { pkp = nullptr; }
    void* pkp;
    point2d position;
};

struct threadinfo {
    threadinfo() {}
    threadinfo(const threadinfo& ti) { 
        mission = ti.mission; 
        information.position = ti.information.position; 
        frame1 = ti.frame1; 
        frame2 = ti.frame2;
    }
    threadinfo& operator=(const threadinfo& ti) {
        mission = ti.mission;
        information.position = ti.information.position;
        frame1 = ti.frame1;
        frame2 = ti.frame2;
        return *this;
    }
    int mission;
    maininfo information;
    void* frame1;
    void* frame2;
};

#endif

namespace clvo {

    extern class system;

    class LucasKanade {
    public:
        LucasKanade(cv::Mat& img1, cv::Mat& img2) : src(img1), des(img2) {}
        ~LucasKanade() {}
        vector<line2d> alignTrans();
        void getLines(int shorter, int longer);
    protected:
        void getImgEdge(double threshold = 0.96);
        //����ƽ��������
        inline ushort sqrtForSimpleEdge(__in__ const float data);
        void getLine(line2d& aim_line, point2d _point, const int longest = 150, const int edge = 50);
        cv::Mat edge;
        cv::Mat src;
        cv::Mat des;
        vector<line2d> lines;
        cv::Mat showLines();
        cv::Mat downSample(cv::Mat& src, double size);
    };

    class matcher2d {
    public:
        matcher2d(point2d& _pt1, point2d& _pt2) :pt1(_pt1), pt2(_pt2){}
        point2d pt1;
        point2d pt2;
    };

    class matcher3d {
    public:
        matcher3d(point3d& _pt1, point3d& _pt2) :pt1(_pt1), pt2(_pt2) {}
        point3d pt1;
        point3d pt2;
    };

    class cornerMatcher {
    public:
        cornerMatcher(system* _system) :mpSystem(_system) {
#ifdef THREADING_POOL
            done = false;
            mission = 0;
            maxthreadsnumber = 3;
            Hs.resize(maxthreadsnumber);
            costs.resize(maxthreadsnumber);
            bs.resize(maxthreadsnumber);
            inBoundaryCounts.resize(maxthreadsnumber);
            halfmatches.resize(maxthreadsnumber);
            readymatchs.resize(maxthreadsnumber);
            try {
                for(int i = 0 ; i < maxthreadsnumber; ++i)
                    threads.push_back(std::thread(&cornerMatcher::thread_work, this));
            }
            catch (...) {
                done = true;
                throw;
            }
#endif
        }
        ~cornerMatcher() {
#ifdef THREADING_POOL
            for (int i = 0; i < threads.size(); ++i) {
                threads[i].join();
            }
            done = true;
#endif
        }
        //һ��ͳһ�Ľӿڣ����ܰ�������ƥ�� + λ����� + ֱ������׷���Ż�
        Sophus::SE3d computePose(frame* frame1, frame* frame2, frame* frame0 = nullptr);
        /*������ƥ�䣬��λ��frame1��cell[i][j]�е���������frame2��cell[i ,i - 1, i + i][j, j - 1, j + 1]���Ÿ�cell�е����������ƥ��
        ����ֵΪ���в���RANSACɸѡ����ڵ㣬RANSAC����ICP
        ��ʵ�ֹ����У�match����Ϊͳһ��ڣ�RGBD��ʵ�ּ�matchRGBD, ˫Ŀ��ʱ���matchSTEREO��
        */
        vector<matcher3d> match(frame* frame1, frame* frame2);
        //���ICP
        Sophus::SE3d ICPSolver(vector<matcher3d>& data, bool is_RANSAC = true);
        //���PnP
        Sophus::SE3d EPnPSolver(vector<matcher3d>& data, frame* pframe1, bool& isOK);
    protected:
        system* mpSystem;
        vector<matcher3d> matchRGBD(frame* frame1, frame* frame2);
        vector<matcher3d> matchSTEREO(frame* frame1, frame* frame2);
        //��������������ĺ�������
        inline int destance(const keypoint& kp1, const keypoint& kp2, int max_distance = 20);
        //�����ж�άƥ����Ϣת��Ϊ��άƥ����Ϣ, ����frame������Ϊ��ȡdepth�е����ֵ
        void match2d_to_3d(frame* frame1, frame* frame2);

        void match2d_to_3d_Stereo(frame* frame1, frame* frame2);
        //��������άƥ����Ϣ��Ϊ��άƥ����Ϣ
        void match3d_to_2d();
        //ICP�޳���ƥ��
        vector<matcher3d> RANSACICP(double p = 0.5, double z = 0.9999, int k = 4, double threshold = 1.6);
        vector<matcher2d> RANSAC(double p = 0.8, double z = 0.99, int k = 8, double threshold = 1e-2);
        Eigen::Matrix3d estimateEssentialEightPoints(vector<matcher2d>& data);
        Eigen::Matrix3d fundamentalEssentialEightPoints(vector<matcher2d>& data);


        //����ֱ�ӷ���λ�˽����Ż�
        Sophus::SE3d directPostOptimize(frame* frame1, frame* frame2, Sophus::SE3d&);
        //���������ͳһ�Ż���������ֱ�ӷ�λ�˹���
        Sophus::SE3d pointDirectPostOptimize(frame* frame1, frame* frame2, Sophus::SE3d&);

        vector<matcher2d> mvMatcher2d;
        vector<matcher3d> mvMatcher3d;
    private:
        //����ƥ����
        cv::Mat showMatch(frame* frame1, frame* frame2);

        //////�̳߳�////////
#ifdef THREADING_POOL
        int maxthreadsnumber;
        std::atomic_bool done;
        std::atomic_int mission;
        clmt::threadSafeQueue<threadinfo> work_queue;
        std::vector<std::thread> threads;
        std::vector<std::vector<keypoint*>> halfmatches;
        std::vector<std::vector<matcher2d>> readymatchs;
        mutable std::mutex mut;

        Sophus::SE3d tpT;

        vector<Eigen::Matrix<double, 6, 6>> Hs;
        vector<Eigen::Matrix<double, 6, 1>> bs;
        vector<int> inBoundaryCounts;
        vector<double> costs;
        double tpFx, tpFy, tpCx, tpCy;
        
        vector<clvo::matcher3d> threadingPoolMatchRGBD(frame* frame1, frame* frame2);
        void threadingPoolMatch(frame* frame1, frame* frame2, int start, int end, int threadnumber);
        void threadingPoolInverseMatch(frame* frame1, frame* frame2, keypoint* kp);
        Sophus::SE3d threadingPoolDriect(frame* frame1, frame* frame2, Sophus::SE3d& T);
        
        void threadingPoolDirectHcompute(const cv::Mat* img1, const cv::Mat* img2, int start, vector<point3d>* pvpts);

        void thread_work() {
            while (!done) {
                threadinfo task;
                if (work_queue.tryPop(task)) {
                    switch (task.mission & 0xff) {
                    case 1: {
                        threadingPoolMatch((clvo::frame*)task.frame1, (clvo::frame*)task.frame2, task.information.position[0], task.information.position[1], task.mission >> 8);
                        break;
                    }
                    case 2: {
                        threadingPoolInverseMatch((clvo::frame*)task.frame1, (clvo::frame*)task.frame2, (clvo::keypoint*)task.information.pkp);
                        break;
                    }
                    case 3: {
                        threadingPoolDirectHcompute((cv::Mat*)task.frame1, (cv::Mat*)task.frame2, task.mission >> 8, (vector<point3d>*)task.information.pkp);
                        break;
                    }
                    }   //end of switch                
                }
                else {
                    std::this_thread::yield();
                }
            }
        }

        void submit(threadinfo ti) {
            work_queue.push(ti);
        }
#endif
    };

}