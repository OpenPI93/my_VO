#pragma once
//#include <opencv.hpp>
#include <opencv2/opencv.hpp>
#include "basetype.h"
#include "frame.h"

namespace clvo
{

    class PnPsolver {
    public:
        
        PnPsolver(const vector<point2d>& vpPoint2ds, const vector<point3d> &vpPoint3ds, const vector<double>& K);

        ~PnPsolver();
        //设置参数，详见cpp文件里面的解释
        void SetRansacParameters(double probability = 0.99, int minInliers = 8, int maxIterations = 300, int minSet = 4, float epsilon = 0.4,
            float th2 = 5.991);
        
        //tracking里面使用，返回值为相机位姿
        Sophus::SE3d iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers);

    private:
        //检验有多少点是内点，所有的匹配都要参与到内点检测中。阈值用到的是mvMaxError
        void CheckInliers();
        /*精化模型，具体操作为：
        先根据内点的个数调整一下pws，alphas这些容器的尺寸，使得尺寸不小于内点个数
        用所有筛选的内点一起建立一个模型（在iterate函数里面只用4个点建立模型）
        检查有多少点是内点，如果内点个数足够多，那么返回true，否则返回false
        多说一下，如果返回值为true，那么iterate函数会直接返回精化后的模型所计算出来的T
        */
        bool Refine();

        // Functions from the original EPnP code
        //修改maximum_number_of_correspondences的值，如果n > maximum_number_of_correspondences，则相应的更改pws、us、alphas以及pcs的size
        void set_maximum_number_of_correspondences(const int n);//√
        //number_of_correspondences置为0，iterate函数中RANSAC使用
        void reset_correspondences(void);//√
        //向pws和us中添加一对点，同时number_of_correspondences++，它的作用是把用来建立模型的点保存起来，通常情况下加四个点就可以，Refine函数里面是加全部内点
        void add_correspondence(const double X, const double Y, const double Z,
            const double u, const double v);//√

        /*
         该函数用在compute_pose函数中，用来计算当前模型的位姿并返回当前模型的误差。在compute_pose中总共使用了三次，每一次find_betas_approx后都要记录一次模型误差并保存该模型记录的R 和 t，其功能是计算控制点坐标、计算alpha（α用来填充M矩阵）、计算β，就、求解位姿
          输入值为一个空的旋转矩阵和平移变量
        */
        double compute_pose(Sophus::SE3d& T);//√
       
         //在函数compute_R_and_t的最后一步中调用，目的为计算所有地图点使用当前模型的重投影误差，返回值为平均误差
        double reprojection_error(Sophus::SE3d T);//√

        //计算四个虚拟控制点的坐标
        void choose_control_points(void);//√
        //计算每一个地图点对应的alpha
        void compute_barycentric_coordinates(void);//√
        //计算M矩阵，就是那个n(地图点数量) * 12 的矩阵
        void fill_M(CvMat * M, const int row, const double * alphas, const double u, const double v);//√
        //计算控制点的相机坐标下的坐标，计算公式为c_c = ΣβUT，就是论文里面的公式(8)
        void compute_ccs(const double * betas, const double * ut);//√
         //根据控制点的相机坐标计算投影点的三维坐标，用到的公式为论文里面的公式(2)，即利用c_c和alpha求解
        void compute_pcs(void);//√
        //在利用compute_pcs计算地图点的相机坐标后，如果地图点在相机坐标系下的深度为负，则将地图点在相机坐标系下的坐标取负
        void solve_for_sign(void);//√

        //根据N的数量来求解对应的β，三个函数分别对应N = 4， 2， 3
        void find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho, double * betas);//√
        void find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho, double * betas);//√
        void find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho, double * betas);//√


        //计算向量的点乘
        double dot(const double * v1, const double * v2);//√
        //计算两个点的距离的平方，切记是平方
        double dist2(const double * p1, const double * p2);//√

        //论文中公式(13)用到了一个方程 Lβ=ρ
        //计算公式中的ρ
        void compute_rho(double * rho);//√
        //计算公式中的L
        void compute_L_6x10(const double * ut, double * l_6x10);//√

        //这个类只提供一个迭代的接口，具体的工作是compute_A_and_b_gauss_newton来完成一阶导和误差的计算以及qr_solve来解方程
        void gauss_newton(const CvMat * L_6x10, const CvMat * Rho, double current_betas[4]);//√
        //配合上面的gauss_newton函数使用
        void compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
            double cb[4], CvMat * A, CvMat * b);//√

        //这个函数提供了一个接口，其作用为：调用compute_ccs求控制点的相机坐标，调用compute_pcs求地图点的相机坐标，调用solve_for_sign检查并纠正深度
        //最后调用estimate_R_and_t求解位姿，然后返回reprojection_error，也就是这个模型的重投影误差
        double compute_R_and_t(const double * ut, const double * betas, Sophus::SE3d& T);//√
        //根据ICP计算相机的运动，详细算法可以参考高博的slam十四讲第七章中的ICP的SVD方法
        void estimate_R_and_t(Sophus::SE3d& T);//√

         //一个简单的赋值矩阵和向量的函数，在compute_pose里面用的
        void copy_R_and_t(const Sophus::SE3d& T_src,
            Sophus::SE3d& T_des);

        void qr_solve(CvMat * A, CvMat * b, CvMat * X);

        //旋转矩阵转四元数
        void mat_to_quat(const double R[3][3], double q[4]);

        double fx, fy, cx, cy;

        //alphas的具体数量由每一次ransac中最大点个数确定
        //pws为地图点的世界坐标集合、us为二维坐标集合、alphas为每个地图点对应的四个权重的集合、pcs为地图点的相机坐标集合
        //这四个数组的容量从最开始的4个会逐渐增加，增加容量是在精化阶段进行，因为需要将当前模型的全部内点都参与模型的建立
        double * pws, *us, *alphas, *pcs;
        //记录可以容纳的最大点对以及相应信息的条数
        int maximum_number_of_correspondences;
        //这个变量很重要，它表示每次用多少点对信息求解EPnP问题，比如，普通的筛选模型阶段只需要4个点对，但是在精化阶段则需要用全部的初步筛选的内点建立模型
        int number_of_correspondences;

        //世界坐标系和相机坐标系下的四个控制点w表示world，c表示camera
        double cws[4][3], ccs[4][3];

        //一个没有被用到的变量，看上去应该是四个控制点在世界坐标系下构成矩阵的特征值？
        double cws_determinant;

        //记录传进来的地图点信息，主要用来保存那个size，因为3D-2D匹配信息在构造函数中就已经用输入进来的信息和对应的帧信息保存下来了，后面用这个容器都是取size
        vector<point3d> mvpMapPointMatches;

        // 2D Points
        //地图点对应的每一个投影点的2D坐标
        vector<point2d> mvP2D;
        
        // 3D Points
        //地图点的世界坐标
        vector<point3d> mvP3Dw;

        // Index in Frame
        //记录地图点对应的Frame中的特征点序号，由于有些地图点isBad，所以这里面存储的序号不是连续的
        vector<size_t> mvKeyPointIndices;

        // Current Estimation
        Sophus::SE3d mTi;
        Sophus::SE3d mTcwi;
        //记录当前模型中的内点，或者说是将内点对应的序号的元素置为true，其尺寸为N，即匹配点对的个数，在构造函数中resize
        vector<bool> mvbInliersi;
        //记录当前模型中内点的个数，数值在CheckInliers()中置为0并开始累加，然后在iterate中判断该模型是否为最佳模型
        int mnInliersi;

        // 记录当前RANSAC的迭代次数，用在iterate函数中
        int mnIterations;
        //用来记录当前最好模型的内点，是内点则对应的位置为true
        vector<bool> mvbBestInliers;
        //记录最佳模型里面有几个内点
        int mnBestInliers;
        Sophus::SE3d mBestTcw;

        // Refined
        //精化后的位姿，通过精化的位姿直接被打包返回了
        Sophus::SE3d mRefinedTcw;
        //精化后的内点
        vector<bool> mvbRefinedInliers;
        //精化后的内点个数
        int mnRefinedInliers;

        // Number of Correspondences
        //所有二维特征点个数, 或者说是匹配的个数
        int N;

        // Indices for random selection [0 .. N-1]
        //记录所有合格的地图点的索引，和mvP3Dw配合使用，其序号是必然连续的
        vector<size_t> mvAllIndices;

        // RANSAC probability
        //只在SetRansacParameters函数中用到，计算最终需要迭代多少次的时候使用
        double mRansacProb;

        // RANSAC最终至少需要让多少个内点符合模型
        int mRansacMinInliers;

        // RANSAC最多迭代多少次
        int mRansacMaxIts;

        // RANSAC expected inliers/total ratio
        // 在SetRansacParameters函数中计算，用来记录（内点个数/总点对数），这里的内点个数不一定是用户最开始输入的那个内点个数，详见构造函数中的解释
        float mRansacEpsilon;

        // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
        
        float mRansacTh;

        // RANSAC Minimun Set used at each iteration
        // 构建模型需要的最少点对数，在构造函数中设置，在EPnP里面这个数一般就是4
        int mRansacMinSet;

       

    };

} //namespace ORB_SLAM

