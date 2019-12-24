#pragma once
#include "basetype.h"
#include "frame.h"

namespace clvo {

    typedef Eigen::Matrix<double, 6, 6> H_cam;
    typedef Eigen::Matrix<double, 6, 3> H_Edge;
    typedef Eigen::Matrix<double, 3, 3> H_pt;

    class directPoseOnlyBackEnd {
    public:
        directPoseOnlyBackEnd(const vector<double>& K) : fx(K[0]), fy(K[2]), cx(K[1]), cy(K[3]), baseline(K[4]) {}
        ~directPoseOnlyBackEnd() {}
        void run(vector<std::shared_ptr<frame>>& frames);
    protected:
        double fx;
        double fy;
        double cx;
        double cy;
        double baseline;
    };

    class optimizer {
    public:
        optimizer() {}
        //用于局部地图的后端优化
        void runLocalMap(vector<Sophus::SE3d>& poses, vector<point3d>& points, vector<clvo::observation>& observations);
        
        /*
        * 用来进行局部地图优化
        * @param    vpframes 全部帧
        * @param    start 起始帧，即以该帧为初试位姿
        * @param    end 结束帧，通常情况下为刚刚求解完位姿的帧
        * @throw    param_exception if the end - start < 1
        * @throw    param_exception if the vpframes is empty
        */
        void runPostOnly(vector< std::shared_ptr<frame> >& vpframes, int start, int end);
    protected:
        void computeJacobian(vector<Sophus::SE3d>& poses, vector<point3d>& points, vector<clvo::observation>& observations);

        /*
        | B E |
        | E C |       
        */
        vector<H_cam> B;
        vector<H_Edge> E;
        vector<H_pt> C;
    };
}