#include "epnp.h"

namespace clvo{
PnPsolver::PnPsolver(const vector<point2d>& vpPoint2ds, const vector<point3d> &vpPoint3ds, const vector<double>& K) :
    pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
    mnIterations(0), mnBestInliers(0), N(0)
{
    // 根据点数初始化容器的大小
    mvpMapPointMatches = vpPoint3ds;
    mvP2D.reserve(vpPoint2ds.size());
   
    mvP3Dw.reserve(vpPoint3ds.size());
    
    mvAllIndices.reserve(vpPoint3ds.size());

    int idx = 0;
    for (size_t i = 0, iend = vpPoint3ds.size(); i<iend; i++)
    {
        point3d pMP = vpPoint3ds[i];//依次获取一个MapPoint

         mvP2D.push_back(vpPoint2ds[i]);//存放到mvP2D容器
                
         mvP3Dw.push_back(pMP);

         mvKeyPointIndices.push_back(i);//记录被使用特征点在原始特征点容器中的索引, mvKeyPointIndices是跳跃的
         mvAllIndices.push_back(idx);//记录被使用特征点的索引, mvAllIndices是连续的

         idx++;       
    }

    // Set camera calibration parameters
    fx = K[0];
    cx = K[1];
    fy = K[2];
    cy = K[3];

    SetRansacParameters();
}

PnPsolver::~PnPsolver()
{
    delete[] pws;
    delete[] us;
    delete[] alphas;
    delete[] pcs;
}

// 设置RANSAC迭代的参数
/*
* @param probability = 0.99, 即中RANSAC中的z（可以得到一个理想模型的概率）
* @param minInliers = 8,
* @param maxIterations = 300, RANSAC最大迭代次数？
* @param minSet = 4, 需要的最少数据个数，也就是每次至少有四对匹配信息
* @param epsilon = 0.4, 即RANSAC中的p, 表示随机选择一个点是内点的概率
* @param th2 = 5.991
*/
void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;
    mRansacEpsilon = epsilon;
    mRansacMinSet = minSet;

    N = mvP2D.size(); // 所有二维特征点个数

    mvbInliersi.resize(N);// inlier index, mvbInliersi记录每次迭代inlier的点

                          // Adjust Parameters according to number of correspondences（不是我写的哈）
                          /*这里nMinInliers表示模型最终需要得到多少个内点，用N乘以内点的百分比就可以得到满足输入值的内点个数。但是这个个数不见得和输入的minInliers相同，所以就需要从输入的minInliers，N*mRansacEpsilon得到的nMinInliers，还有minSet（构建模型的最少内点数）中找一个最大值作为最终的“所有输入中内点个数的最小值”。这个时候我们在一开始输入的那个内点概率epsilon和真正的（最少内点数/总点对数）很可能会不一样，于是我们取那个两者的最大值作为最终的概率*/
    int nMinInliers = N*mRansacEpsilon;
    if (nMinInliers<mRansacMinInliers)
        nMinInliers = mRansacMinInliers;
    if (nMinInliers<minSet)
        nMinInliers = minSet;
    //mRansacMinInliers记录的就是RANSAC最终要得到最少多少个内点
    mRansacMinInliers = nMinInliers;

    if (mRansacEpsilon<(float)mRansacMinInliers / N)
        mRansacEpsilon = (float)mRansacMinInliers / N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    //如果点对的个数恰巧是构建模型需要的最少点对个数，则迭代次数为1
    if (mRansacMinInliers == N)
        nIterations = 1;
    else {
        //RANSAC公式
        nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(mRansacEpsilon, 4)));
    }

    //这个地方的意思是：如果经过计算得到RANSAC根本用不了迭代maxIterations那么多次，则迭代nIterations次，如果经过计算需要迭代的次数太多，那么只迭代nIterations次。当然，如果不小心把nIterations计算成一个小于1的数，那么还是要迭代一次的嘛。
    //mRansacMaxIts就是记录RANSAC最终要迭代多少次
    //mRansacMaxIts = std::max(1, std::min(nIterations, mRansacMaxIts));
    mRansacMaxIts = std::max(1, std::max(nIterations, mRansacMaxIts));

    mRansacTh = th2;
}

Sophus::SE3d PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers.clear();
    nInliers = 0;

    // mRansacMinSet为每次RANSAC需要的特征点数，默认为4组3D-2D对应点，这里设置构建模型需要的最少点数是为了给pws那四个容器比较合理的size，这个size只能越来越大
    set_maximum_number_of_correspondences(mRansacMinSet);

    // N为所有2D点的个数, mRansacMinInliers为RANSAC迭代过程中最少的inlier数，如果参与迭代的总匹配数都不够建立模型的，表明跟丢了
    if (N<mRansacMinInliers)
    {
        bNoMore = true;
        return Sophus::SE3d();
    }

    
    // mvAllIndices为所有参与PnP的2D点的索引
    // vAvailableIndices为每次从mvAllIndices中随机挑选mRansacMinSet组3D-2D对应点进行一次RANSAC，然后将选入模型的点删除
    vector<size_t> vAvailableIndices;

    int nCurrentIterations = 0;

    std::random_device rd;
    std::mt19937 mt(rd());
    
    while (mnIterations<mRansacMaxIts || nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;
        reset_correspondences();

        vAvailableIndices = mvAllIndices;
        std::uniform_real_distribution<double> dist(0, vAvailableIndices.size() - 1);
        // Get min set of points
        for (short i = 0; i < mRansacMinSet; ++i)
        {
            int randi = dist(mt);

            int idx = vAvailableIndices[randi];

            // 将对应的3D-2D压入到pws和us
            add_correspondence(mvP3Dw[idx][0], mvP3Dw[idx][1], mvP3Dw[idx][2], mvP2D[idx][0], mvP2D[idx][1]);

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // Compute camera pose
        // 这个地方就是EPnP的入口了
        compute_pose(mTi);

        // Check inliers
        CheckInliers();
        //这个模型选出来的内点数比最低要求要多的情况下
        if (mnInliersi >= mRansacMinInliers)
        {
            // If it is the best solution so far, save it
            //把当前模型设置为最佳模型
            if (mnInliersi>mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                mBestTcw = mTi;
            }

            //对模型进行精化，如果精化后发现这个模型依然很优秀（emmm，就是剩下的内点数依然比最低要求高），就把精化后的模型返回，精化后的内点也计再vbInliers里面
            if (Refine())
            {
                nInliers = mnRefinedInliers;
                vbInliers = vector<bool>(mvpMapPointMatches.size(), false);
                for (int i = 0; i<N; i++)
                {
                    if (mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                return mRefinedTcw;
            }

        }//end of if(mnInliersi>=mRansacMinInliers)
    }//end of while

     //迭代完以后，如果迭代次数不少于迭代的最大次数，那么有两种可能，一种是迭代结束前找到了最佳模型但是这个模型经过精化发现它并不是那么好，另一种是干脆没找到
    if (mnIterations >= mRansacMaxIts)
    {
        bNoMore = true;
        //如果找到了一个最佳模型，那么把最佳模型的内点记录下来，然后把位姿返回，否则就算跟丢了
        if (mnBestInliers >= mRansacMinInliers)
        {
            nInliers = mnBestInliers;
            vbInliers = vector<bool>(mvpMapPointMatches.size(), false);
            for (int i = 0; i<N; i++)
            {
                if (mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            return mBestTcw;
        }
    }
    //这是跟丢了
    return Sophus::SE3d();
}

bool PnPsolver::Refine()
{
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    //提取出所有内点
    for (size_t i = 0; i<mvbBestInliers.size(); i++)
    {
        if (mvbBestInliers[i])
        {
            vIndices.push_back(i);
        }
    }

    set_maximum_number_of_correspondences(vIndices.size());

    reset_correspondences();

    for (size_t i = 0; i<vIndices.size(); i++)
    {
        int idx = vIndices[i];
        add_correspondence(mvP3Dw[idx][0], mvP3Dw[idx][1], mvP3Dw[idx][2], mvP2D[idx][0], mvP2D[idx][1]);
    }

    // Compute camera pose
    compute_pose(mTi);

    // Check inliers
    CheckInliers();

    // 通过CheckInliers函数得到那些inlier点用来提纯
    mnRefinedInliers = mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    //如果提纯得到的内点数依然多于最少内点个数，则表示这个模型还不错，那么便把这个模型作为该帧的运动返回
    if (mnInliersi>mRansacMinInliers)
    {
        mRefinedTcw = mTi;
        return true;
    }

    return false;
}

// 通过之前求解的(R t)检查哪些3D-2D点对属于inliers
void PnPsolver::CheckInliers()
{
    mnInliersi = 0;

    for (int i = 0; i<N; i++)
    {
        point3d P3Dw = mvP3Dw[i];
        point2d P2D = mvP2D[i];

        // 将3D点由世界坐标系旋转到相机坐标系
        point3d P3DC = mTi * P3Dw;
        P3DC /= P3DC[2];
        float Xc = P3DC[0];
        float Yc = P3DC[1];

        // 将相机坐标系下的3D进行针孔投影
        double ue = cx + fx * Xc;
        double ve = cy + fy * Yc;

        // 计算残差大小
        float distX = P2D[0] - ue;
        float distY = P2D[1] - ve;

        float error2 = distX*distX + distY*distY;

        if (error2 < mRansacTh)
        {
            mvbInliersi[i] = true;
            mnInliersi++;
        }
        else
        {
            mvbInliersi[i] = false;
        }
    }
}

// number_of_correspondences为RANSAC每次PnP求解时时3D点和2D点匹配对数
// RANSAC需要很多次，maximum_number_of_correspondences为匹配对数最大值
// 这个变量用于决定pws us alphas pcs容器的大小，因此只能逐渐变大不能减小
// 如果maximum_number_of_correspondences之前设置的过小，则重新设置，并重新初始化pws us alphas pcs的大小
// set_maximum_number_of_correspondences总共被调用过两次，一次为Refine中使用，另一次为iterate函数中
void PnPsolver::set_maximum_number_of_correspondences(int n)
{
    if (maximum_number_of_correspondences < n) {
        if (pws != 0) delete[] pws;
        if (us != 0) delete[] us;
        if (alphas != 0) delete[] alphas;
        if (pcs != 0) delete[] pcs;

        maximum_number_of_correspondences = n;
        pws = new double[3 * maximum_number_of_correspondences];// 每个3D点有(X Y Z)三个值
        us = new double[2 * maximum_number_of_correspondences];// 每个图像2D点有(u v)两个值
        alphas = new double[4 * maximum_number_of_correspondences];// 每个3D点由四个控制点拟合，有四个系数
        pcs = new double[3 * maximum_number_of_correspondences];// 每个3D点有(X Y Z)三个值
    }
}

void PnPsolver::reset_correspondences(void)
{
    number_of_correspondences = 0;
}

void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v)
{
    pws[3 * number_of_correspondences] = X;
    pws[3 * number_of_correspondences + 1] = Y;
    pws[3 * number_of_correspondences + 2] = Z;

    us[2 * number_of_correspondences] = u;
    us[2 * number_of_correspondences + 1] = v;

    number_of_correspondences++;
}

void PnPsolver::choose_control_points(void)
{
    // Take C0 as the reference points centroid:
    // 步骤1：第一个控制点：参与PnP计算的参考3D点的几何中心
    cws[0][0] = cws[0][1] = cws[0][2] = 0;
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            cws[0][j] += pws[3 * i + j];

    for (int j = 0; j < 3; j++)
        cws[0][j] /= number_of_correspondences;


    // Take C1, C2, and C3 from PCA on the reference points:
    // 步骤2：计算其它三个控制点，C1, C2, C3通过PCA分解得到
    // 将所有的3D参考点写成矩阵，(number_of_correspondences *　３)的矩阵
    CvMat * PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);

    double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
    CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
    CvMat DC = cvMat(3, 1, CV_64F, dc);
    CvMat UCt = cvMat(3, 3, CV_64F, uct);

    // 步骤2.1：将存在pws中的参考3D点减去第一个控制点的坐标（相当于把第一个控制点作为原点）, 并存入PW0
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];

    // 步骤2.2：利用SVD分解P'P可以获得P的主分量
    // 类似于齐次线性最小二乘求解的过程，
    // PW0的转置乘以PW0
    cvMulTransposed(PW0, &PW0tPW0, 1);
    cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

    cvReleaseMat(&PW0);

    // 步骤2.3：得到C1, C2, C3三个3D控制点，最后加上之前减掉的第一个控制点这个偏移量
    for (int i = 1; i < 4; i++) {
        double k = sqrt(dc[i - 1] / number_of_correspondences);
        for (int j = 0; j < 3; j++)
            cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
    }
}

// 求解四个控制点的系数alphas
// (a2 a3 a4)' = inverse(cws2-cws1 cws3-cws1 cws4-cws1)*(pws-cws1)，a1 = 1-a2-a3-a4
// 每一个3D控制点，都有一组alphas与之对应
// cws1 cws2 cws3 cws4为四个控制点的坐标
// pws为3D参考点的坐标
void PnPsolver::compute_barycentric_coordinates(void)
{
    double cc[3 * 3], cc_inv[3 * 3];
    CvMat CC = cvMat(3, 3, CV_64F, cc);
    CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

    // 第一个控制点在质心的位置，后面三个控制点减去第一个控制点的坐标（以第一个控制点为原点）
    // 步骤1：减去质心后得到x y z轴
    // 
    // cws的排列 |cws1_x cws1_y cws1_z|  ---> |cws1|
    //          |cws2_x cws2_y cws2_z|       |cws2|
    //          |cws3_x cws3_y cws3_z|       |cws3|
    //          |cws4_x cws4_y cws4_z|       |cws4|
    //          
    // cc的排列  |cc2_x cc3_x cc4_x|  --->|cc2 cc3 cc4|
    //          |cc2_y cc3_y cc4_y|
    //          |cc2_z cc3_z cc4_z|
    for (int i = 0; i < 3; i++)
        for (int j = 1; j < 4; j++) {
            //计算控制点的去质心坐标
            cc[3 * i + j - 1] = cws[j][i] - cws[0][i];
        }

    //求那个C逆矩阵
    cvInvert(&CC, &CC_inv, CV_SVD);
    double * ci = cc_inv;
    //对每一个点对应的alpha系数进行计算
    for (int i = 0; i < number_of_correspondences; i++) {
        double * pi = pws + 3 * i;// pi指向第i个3D点的首地址
        double * a = alphas + 4 * i;// a指向第i个控制点系数alphas的首地址

                                    // pi[]-cws[0][]表示将pi和步骤1进行相同的平移
        for (int j = 0; j < 3; j++) {
            a[1 + j] = ci[3 * j] * (pi[0] - cws[0][0]) +
                ci[3 * j + 1] * (pi[1] - cws[0][1]) +
                ci[3 * j + 2] * (pi[2] - cws[0][2]);
        }
        //四个系数的和为1
        a[0] = 1.0f - a[1] - a[2] - a[3];
    }
}

// 填充最小二乘的M矩阵
// 对每一个3D参考点：
// |ai1 0    -ai1*ui, ai2  0    -ai2*ui, ai3 0   -ai3*ui, ai4 0   -ai4*ui|
// |0   ai1  -ai1*vi, 0    ai2  -ai2*vi, 0   ai3 -ai3*vi, 0   ai4 -ai4*vi|
// 其中i从0到4
void PnPsolver::fill_M(CvMat * M,
    const int row, const double * as, const double u, const double v)
{
    double * M1 = M->data.db + row * 12;
    double * M2 = M1 + 12;

    for (int i = 0; i < 4; i++) {
        M1[3 * i] = as[i] * fx;
        M1[3 * i + 1] = 0.0;
        M1[3 * i + 2] = as[i] * (cx - u);

        M2[3 * i] = 0.0;
        M2[3 * i + 1] = as[i] * fy;
        M2[3 * i + 2] = as[i] * (cy - v);
    }
}

// 每一个控制点在相机坐标系下都表示为特征向量乘以beta的形式，EPnP论文的公式16
void PnPsolver::compute_ccs(const double * betas, const double * ut)
{
    for (int i = 0; i < 4; i++)
        ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

    for (int i = 0; i < 4; i++) {
        const double * v = ut + 12 * (11 - i);
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
                ccs[j][k] += betas[i] * v[3 * j + k];
    }
}

// 用四个控制点作为单位向量表示下的世界坐标系下3D点的坐标
void PnPsolver::compute_pcs(void)
{
    for (int i = 0; i < number_of_correspondences; i++) {
        double * a = alphas + 4 * i;
        double * pc = pcs + 3 * i;

        for (int j = 0; j < 3; j++)
            pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
    }
}

double PnPsolver::compute_pose(Sophus::SE3d& T)
{
    // 步骤1：获得EPnP算法中的四个控制点
    choose_control_points();
    // 步骤2：计算世界坐标系下每个3D点用4个控制点线性表达时的系数alphas，公式1
    compute_barycentric_coordinates();

    // 步骤3：构造M矩阵，公式(3)(4)-->(5)(6)(7)
    CvMat * M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);

    for (int i = 0; i < number_of_correspondences; i++)
        fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

    double mtm[12 * 12], d[12], ut[12 * 12];
    CvMat MtM = cvMat(12, 12, CV_64F, mtm);
    CvMat D = cvMat(12, 1, CV_64F, d);
    CvMat Ut = cvMat(12, 12, CV_64F, ut);

    // 步骤3：求解Mx = 0
    // SVD分解M'M
    cvMulTransposed(M, &MtM, 1);

    //这里MtM是对称阵，所有左右奇异矩阵是相同的，所以可以看作  xT * M = 0, 这个方程的零空间为UT
    cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);//得到向量ut
    cvReleaseMat(&M);

    double l_6x10[6 * 10], rho[6];
    CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
    CvMat Rho = cvMat(6, 1, CV_64F, rho);

    compute_L_6x10(ut, l_6x10);
    compute_rho(rho);

    //为什么这里的Beta是[4][4]? Beta[0]从头到尾就没用过呀？而且对应的rep_errors, Rs, ts都只用了三个
    double Betas[4][4], rep_errors[4];
    Sophus::SE3d Ts[4];

    // 不管什么情况，都假设论文中N=4，并求解部分betas（如果全求解出来会有冲突）
    // 通过优化得到剩下的betas
    // 最后计算R t

    //后面的三步find_betas_approx是分别求解β，每一步对应的应该是N = 4， 2， 3的情况

    // EPnP论文公式10 15（此处为吴博的注释，我感觉用到的不是公式10，因为这段代码直接copy的OpenCV\calib3D\epnp.cpp里面的代码，这里的1可能表示的仅仅是一个序号？）
    //我大胆猜测这是假设论文里面N的值为4，即Beta有4个值，gauss_newton函数是为了迭代得到最小误差，但是实际代码里面的迭代次数是固定的，可能仅仅是为了得到一个比较合理的误差
    find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
    gauss_newton(&L_6x10, &Rho, Betas[1]);
    rep_errors[1] = compute_R_and_t(ut, Betas[1], Ts[1]);

    // EPnP论文公式11 15
    // 这是假设论文里面N的值为2，即Beta有2个值
    find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
    gauss_newton(&L_6x10, &Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(ut, Betas[2], Ts[2]);

    // 这是假设论文里面N的值为3，即Beta有3个值
    find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
    gauss_newton(&L_6x10, &Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(ut, Betas[3], Ts[3]);

    //寻找误差最小的情况
    int N = 1;
    if (rep_errors[2] < rep_errors[1]) N = 2;
    if (rep_errors[3] < rep_errors[N]) N = 3;

    //把误差最小的情况设置为本模型的相机外参
    copy_R_and_t(Ts[N], T);

    //没有哪个函数需要用这个返回值，为什么要返回？
    return rep_errors[N];
}

void PnPsolver::copy_R_and_t(const Sophus::SE3d& T_src,
    Sophus::SE3d& T_des)
{
    T_des = T_src;
}

double PnPsolver::dist2(const double * p1, const double * p2)
{
    return
        (p1[0] - p2[0]) * (p1[0] - p2[0]) +
        (p1[1] - p2[1]) * (p1[1] - p2[1]) +
        (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double PnPsolver::dot(const double * v1, const double * v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

//计算重投影误差
double PnPsolver::reprojection_error(Sophus::SE3d T)
{
    double sum2 = 0.0;

    for (int i = 0; i < number_of_correspondences; i++) {
        double * pw = pws + 3 * i;
        //计算地图点在相机坐标系下的x, y以及z.inverse

        point3d P3DC = mTi * point3d(pw[0], pw[1], pw[2]);
        P3DC /= P3DC[2];
        float Xc = P3DC[0];
        float Yc = P3DC[1];
        
        
        double ue = cx + fx * Xc;
        double ve = cy + fy * Yc;
        double u = us[2 * i], v = us[2 * i + 1];

        sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
    }
    //返回一个平均误差
    return sum2 / number_of_correspondences;
}

// 根据世界坐标系下的四个控制点与机体坐标下对应的四个控制点（和世界坐标系下四个控制点相同尺度），求取R t，也就是ICP问题
void PnPsolver::estimate_R_and_t(Sophus::SE3d& T)
{
    //所有地图点的质心（pc0是相机坐标系下的质心，pw0是世界坐标系下的质心）
    double pc0[3], pw0[3];

    pc0[0] = pc0[1] = pc0[2] = 0.0;
    pw0[0] = pw0[1] = pw0[2] = 0.0;

    for (int i = 0; i < number_of_correspondences; i++) {
        const double * pc = pcs + 3 * i;
        const double * pw = pws + 3 * i;

        for (int j = 0; j < 3; j++) {
            pc0[j] += pc[j];
            pw0[j] += pw[j];
        }
    }//end of for

    for (int j = 0; j < 3; j++) {
        pc0[j] /= number_of_correspondences;
        pw0[j] /= number_of_correspondences;
    }

    double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
    CvMat ABt = cvMat(3, 3, CV_64F, abt);
    CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
    CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
    CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

    cvSetZero(&ABt);
    for (int i = 0; i < number_of_correspondences; i++) {
        double * pc = pcs + 3 * i;
        double * pw = pws + 3 * i;
        //abt为
        for (int j = 0; j < 3; j++) {
            abt[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
            abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
            abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
        }
    }

    double R[3][3];
    point3d t;

    cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);

    const double det =
        R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
        R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

    if (det < 0) {
        R[2][0] = -R[2][0];
        R[2][1] = -R[2][1];
        R[2][2] = -R[2][2];
    }

    t[0] = pc0[0] - dot(R[0], pw0);
    t[1] = pc0[1] - dot(R[1], pw0);
    t[2] = pc0[2] - dot(R[2], pw0);

    Eigen::Matrix<double, 3, 3> Rot;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Rot(i, j) = R[i][j];

    T = Sophus::SE3d(Rot, t);
}

//这个函数的功能看上去应该是，如果相机坐标系下的深度为负数，则把点坐标取个负
void PnPsolver::solve_for_sign(void)
{
    if (pcs[2] < 0.0) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                ccs[i][j] = -ccs[i][j];

        for (int i = 0; i < number_of_correspondences; i++) {
            pcs[3 * i] = -pcs[3 * i];
            pcs[3 * i + 1] = -pcs[3 * i + 1];
            pcs[3 * i + 2] = -pcs[3 * i + 2];
        }
    }//end of if
}

double PnPsolver::compute_R_and_t(const double * ut, const double * betas,
    Sophus::SE3d& T)
{
    //c_c = ΣβUT
    compute_ccs(betas, ut);
    compute_pcs();

    solve_for_sign();

    estimate_R_and_t(T);

    return reprojection_error(T);
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void PnPsolver::find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho,
    double * betas)
{
    double l_6x4[6 * 4], b4[4];
    CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
    CvMat B4 = cvMat(4, 1, CV_64F, b4);
    //这个for就是把数据Copy出来嘛
    for (int i = 0; i < 6; i++) {
        cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
        cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
    }

    cvSolve(&L_6x4, Rho, &B4, CV_SVD);
    //这里的b4[0]就是beta1 * beta1，后面求解的是beta2，beta3和beta4
    if (b4[0] < 0) {
        betas[0] = sqrt(-b4[0]);
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    }
    else {
        betas[0] = sqrt(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void PnPsolver::find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho,
    double * betas)
{
    double l_6x3[6 * 3], b3[3];
    CvMat L_6x3 = cvMat(6, 3, CV_64F, l_6x3);
    CvMat B3 = cvMat(3, 1, CV_64F, b3);

    for (int i = 0; i < 6; i++) {
        cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
    }

    cvSolve(&L_6x3, Rho, &B3, CV_SVD);

    if (b3[0] < 0) {
        betas[0] = sqrt(-b3[0]);
        betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
    }
    else {
        betas[0] = sqrt(b3[0]);
        betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0) betas[0] = -betas[0];

    betas[2] = 0.0;
    betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void PnPsolver::find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho,
    double * betas)
{
    double l_6x5[6 * 5], b5[5];
    CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
    CvMat B5 = cvMat(5, 1, CV_64F, b5);

    for (int i = 0; i < 6; i++) {
        cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
        cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
        cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
    }

    cvSolve(&L_6x5, Rho, &B5, CV_SVD);

    if (b5[0] < 0) {
        betas[0] = sqrt(-b5[0]);
        betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
    }
    else {
        betas[0] = sqrt(b5[0]);
        betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
    }
    if (b5[1] < 0) betas[0] = -betas[0];
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}

// 计算并填充矩阵L，如果想知道结构，可以按照论文里面的公式(12)，把β的数量变成4，然后展开，就可以知道最后那个for循环是怎么得到的了
void PnPsolver::compute_L_6x10(const double * ut, double * l_6x10)
{
    const double * v[4];

    v[0] = ut + 12 * 11;
    v[1] = ut + 12 * 10;
    v[2] = ut + 12 * 9;
    v[3] = ut + 12 * 8;

    /*
    4对应着β有4个值；6对应着有两两相乘有6种情况，就是论文里面的i，j；3对应的是三维子向量
    */
    double dv[4][6][3];

    for (int i = 0; i < 4; i++) {
        //这里的a代表论文中的i，b代表论文中的j
        int a = 0, b = 1;
        for (int j = 0; j < 6; j++) {
            dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
            dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
            dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

            b++;              //
            if (b > 3) {      //  这四行就是计算那6种组合方式喽
                a++;            //
                b = a + 1;      //
            }
        }
    }

    for (int i = 0; i < 6; i++) {
        double * row = l_6x10 + 10 * i;

        row[0] = dot(dv[0][i], dv[0][i]);
        row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
        row[2] = dot(dv[1][i], dv[1][i]);
        row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
        row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
        row[5] = dot(dv[2][i], dv[2][i]);
        row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
        row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
        row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
        row[9] = dot(dv[3][i], dv[3][i]);
    }
}

// 计算四个控制点任意两点间的距离，总共6个距离，对应论文里面的公式(10)的等式右面那一坨
void PnPsolver::compute_rho(double * rho)
{
    rho[0] = dist2(cws[0], cws[1]);
    rho[1] = dist2(cws[0], cws[2]);
    rho[2] = dist2(cws[0], cws[3]);
    rho[3] = dist2(cws[1], cws[2]);
    rho[4] = dist2(cws[1], cws[3]);
    rho[5] = dist2(cws[2], cws[3]);
}

//这个函数的作用是为它后面那个函数gauss_newton做辅助的，目的是计算误差 CvMat * b 以及计算L矩阵对β的导数 CvMat * A
void PnPsolver::compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
    double betas[4], CvMat * A, CvMat * b)
{
    for (int i = 0; i < 6; i++) {
        const double * rowL = l_6x10 + i * 10;
        //这里的rowA是L_6x10对betas的一阶导
        double * rowA = A->data.db + i * 4;

        rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
        rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
        rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
        rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

        /*这个地方为了加深一下记忆，复习一下吧：
        rho表示的是控制点 i 和 j 在世界坐标系下的距离，由于经过变换后（相机外参嘛）对应控制点的距离是不变的，故这里设置error为世界坐标系下的点的距离减去
        我们估算的十种β和对应L乘积 Lβ 的差，对，就是那个 Lβ = ρ。然后for循环里的6就是i和j的六种组合状态。具体的可以看论文里面的公式(15)
        */
        cvmSet(b, i, 0, rho[i] -
            (
                rowL[0] * betas[0] * betas[0] +
                rowL[1] * betas[0] * betas[1] +
                rowL[2] * betas[1] * betas[1] +
                rowL[3] * betas[0] * betas[2] +
                rowL[4] * betas[1] * betas[2] +
                rowL[5] * betas[2] * betas[2] +
                rowL[6] * betas[0] * betas[3] +
                rowL[7] * betas[1] * betas[3] +
                rowL[8] * betas[2] * betas[3] +
                rowL[9] * betas[3] * betas[3]
                ));
    }
}

//这是高斯牛顿法？高斯牛顿法不应该是 JTJ x = JT e吗？
void PnPsolver::gauss_newton(const CvMat * L_6x10, const CvMat * Rho,
    double betas[4])
{
    const int iterations_number = 5;

    double a[6 * 4], b[6], x[4];
    CvMat A = cvMat(6, 4, CV_64F, a);
    CvMat B = cvMat(6, 1, CV_64F, b);
    CvMat X = cvMat(4, 1, CV_64F, x);

    for (int k = 0; k < iterations_number; k++) {
        //这个地方很巧妙的将数据的指针按照double来读取，data.db就是一个double指针，而data是一个联合体。
        compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db,
            betas, &A, &B);
        qr_solve(&A, &B, &X);

        for (int i = 0; i < 4; i++)
            betas[i] += x[i];
    }
}

void PnPsolver::qr_solve(CvMat * A, CvMat * b, CvMat * X)
{
    static int max_nr = 0;
    static double * A1, *A2;

    const int nr = A->rows;
    const int nc = A->cols;

    if (max_nr != 0 && max_nr < nr) {
        delete[] A1;
        delete[] A2;
    }
    if (max_nr < nr) {
        max_nr = nr;
        A1 = new double[nr];
        A2 = new double[nr];
    }

    double * pA = A->data.db, *ppAkk = pA;
    for (int k = 0; k < nc; k++) {
        double * ppAik = ppAkk, eta = fabs(*ppAik);
        for (int i = k + 1; i < nr; i++) {
            double elt = fabs(*ppAik);
            if (eta < elt) eta = elt;
            ppAik += nc;
        }

        if (eta == 0) {
            A1[k] = A2[k] = 0.0;
            std::cerr << "God damnit, A is singular, this shouldn't happen." << endl;
            return;
        }
        else {
            double * ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
            for (int i = k; i < nr; i++) {
                *ppAik *= inv_eta;
                sum += *ppAik * *ppAik;
                ppAik += nc;
            }
            double sigma = sqrt(sum);
            if (*ppAkk < 0)
                sigma = -sigma;
            *ppAkk += sigma;
            A1[k] = sigma * *ppAkk;
            A2[k] = -eta * sigma;
            for (int j = k + 1; j < nc; j++) {
                double * ppAik = ppAkk, sum = 0;
                for (int i = k; i < nr; i++) {
                    sum += *ppAik * ppAik[j - k];
                    ppAik += nc;
                }
                double tau = sum / A1[k];
                ppAik = ppAkk;
                for (int i = k; i < nr; i++) {
                    ppAik[j - k] -= tau * *ppAik;
                    ppAik += nc;
                }
            }
        }
        ppAkk += nc + 1;
    }

    // b <- Qt b
    double * ppAjj = pA, *pb = b->data.db;
    for (int j = 0; j < nc; j++) {
        double * ppAij = ppAjj, tau = 0;
        for (int i = j; i < nr; i++) {
            tau += *ppAij * pb[i];
            ppAij += nc;
        }
        tau /= A1[j];
        ppAij = ppAjj;
        for (int i = j; i < nr; i++) {
            pb[i] -= tau * *ppAij;
            ppAij += nc;
        }
        ppAjj += nc + 1;
    }

    // X = R-1 b
    double * pX = X->data.db;
    pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
    for (int i = nc - 2; i >= 0; i--) {
        double * ppAij = pA + i * nc + (i + 1), sum = 0;

        for (int j = i + 1; j < nc; j++) {
            sum += *ppAij * pX[j];
            ppAij++;
        }
        pX[i] = (pb[i] - sum) / A2[i];
    }
}




void PnPsolver::mat_to_quat(const double R[3][3], double q[4])
{
    double tr = R[0][0] + R[1][1] + R[2][2];
    double n4;

    if (tr > 0.0f) {
        q[0] = R[1][2] - R[2][1];
        q[1] = R[2][0] - R[0][2];
        q[2] = R[0][1] - R[1][0];
        q[3] = tr + 1.0f;
        n4 = q[3];
    }
    else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
        q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
        q[1] = R[1][0] + R[0][1];
        q[2] = R[2][0] + R[0][2];
        q[3] = R[1][2] - R[2][1];
        n4 = q[0];
    }
    else if (R[1][1] > R[2][2]) {
        q[0] = R[1][0] + R[0][1];
        q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
        q[2] = R[2][1] + R[1][2];
        q[3] = R[2][0] - R[0][2];
        n4 = q[1];
    }
    else {
        q[0] = R[2][0] + R[0][2];
        q[1] = R[2][1] + R[1][2];
        q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
        q[3] = R[0][1] - R[1][0];
        n4 = q[2];
    }
    double scale = 0.5f / double(sqrt(n4));

    q[0] *= scale;
    q[1] *= scale;
    q[2] *= scale;
    q[3] *= scale;
}

} //namespace ORB_SLAM
