#include "epnp.h"

namespace clvo{
PnPsolver::PnPsolver(const vector<point2d>& vpPoint2ds, const vector<point3d> &vpPoint3ds, const vector<double>& K) :
    pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
    mnIterations(0), mnBestInliers(0), N(0)
{
    // ���ݵ�����ʼ�������Ĵ�С
    mvpMapPointMatches = vpPoint3ds;
    mvP2D.reserve(vpPoint2ds.size());
   
    mvP3Dw.reserve(vpPoint3ds.size());
    
    mvAllIndices.reserve(vpPoint3ds.size());

    int idx = 0;
    for (size_t i = 0, iend = vpPoint3ds.size(); i<iend; i++)
    {
        point3d pMP = vpPoint3ds[i];//���λ�ȡһ��MapPoint

         mvP2D.push_back(vpPoint2ds[i]);//��ŵ�mvP2D����
                
         mvP3Dw.push_back(pMP);

         mvKeyPointIndices.push_back(i);//��¼��ʹ����������ԭʼ�����������е�����, mvKeyPointIndices����Ծ��
         mvAllIndices.push_back(idx);//��¼��ʹ�������������, mvAllIndices��������

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

// ����RANSAC�����Ĳ���
/*
* @param probability = 0.99, ����RANSAC�е�z�����Եõ�һ������ģ�͵ĸ��ʣ�
* @param minInliers = 8,
* @param maxIterations = 300, RANSAC������������
* @param minSet = 4, ��Ҫ���������ݸ�����Ҳ����ÿ���������Ķ�ƥ����Ϣ
* @param epsilon = 0.4, ��RANSAC�е�p, ��ʾ���ѡ��һ�������ڵ�ĸ���
* @param th2 = 5.991
*/
void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;
    mRansacEpsilon = epsilon;
    mRansacMinSet = minSet;

    N = mvP2D.size(); // ���ж�ά���������

    mvbInliersi.resize(N);// inlier index, mvbInliersi��¼ÿ�ε���inlier�ĵ�

                          // Adjust Parameters according to number of correspondences��������д�Ĺ���
                          /*����nMinInliers��ʾģ��������Ҫ�õ����ٸ��ڵ㣬��N�����ڵ�İٷֱȾͿ��Եõ���������ֵ���ڵ����������������������ú������minInliers��ͬ�����Ծ���Ҫ�������minInliers��N*mRansacEpsilon�õ���nMinInliers������minSet������ģ�͵������ڵ���������һ�����ֵ��Ϊ���յġ������������ڵ��������Сֵ�������ʱ��������һ��ʼ������Ǹ��ڵ����epsilon�������ģ������ڵ���/�ܵ�������ܿ��ܻ᲻һ������������ȡ�Ǹ����ߵ����ֵ��Ϊ���յĸ���*/
    int nMinInliers = N*mRansacEpsilon;
    if (nMinInliers<mRansacMinInliers)
        nMinInliers = mRansacMinInliers;
    if (nMinInliers<minSet)
        nMinInliers = minSet;
    //mRansacMinInliers��¼�ľ���RANSAC����Ҫ�õ����ٶ��ٸ��ڵ�
    mRansacMinInliers = nMinInliers;

    if (mRansacEpsilon<(float)mRansacMinInliers / N)
        mRansacEpsilon = (float)mRansacMinInliers / N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    //�����Եĸ���ǡ���ǹ���ģ����Ҫ�����ٵ�Ը��������������Ϊ1
    if (mRansacMinInliers == N)
        nIterations = 1;
    else {
        //RANSAC��ʽ
        nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(mRansacEpsilon, 4)));
    }

    //����ط�����˼�ǣ������������õ�RANSAC�����ò��˵���maxIterations��ô��Σ������nIterations�Σ��������������Ҫ�����Ĵ���̫�࣬��ôֻ����nIterations�Ρ���Ȼ�������С�İ�nIterations�����һ��С��1��������ô����Ҫ����һ�ε��
    //mRansacMaxIts���Ǽ�¼RANSAC����Ҫ�������ٴ�
    //mRansacMaxIts = std::max(1, std::min(nIterations, mRansacMaxIts));
    mRansacMaxIts = std::max(1, std::max(nIterations, mRansacMaxIts));

    mRansacTh = th2;
}

Sophus::SE3d PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers.clear();
    nInliers = 0;

    // mRansacMinSetΪÿ��RANSAC��Ҫ������������Ĭ��Ϊ4��3D-2D��Ӧ�㣬�������ù���ģ����Ҫ�����ٵ�����Ϊ�˸�pws���ĸ������ȽϺ����size�����sizeֻ��Խ��Խ��
    set_maximum_number_of_correspondences(mRansacMinSet);

    // NΪ����2D��ĸ���, mRansacMinInliersΪRANSAC�������������ٵ�inlier������������������ƥ��������������ģ�͵ģ�����������
    if (N<mRansacMinInliers)
    {
        bNoMore = true;
        return Sophus::SE3d();
    }

    
    // mvAllIndicesΪ���в���PnP��2D�������
    // vAvailableIndicesΪÿ�δ�mvAllIndices�������ѡmRansacMinSet��3D-2D��Ӧ�����һ��RANSAC��Ȼ��ѡ��ģ�͵ĵ�ɾ��
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

            // ����Ӧ��3D-2Dѹ�뵽pws��us
            add_correspondence(mvP3Dw[idx][0], mvP3Dw[idx][1], mvP3Dw[idx][2], mvP2D[idx][0], mvP2D[idx][1]);

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // Compute camera pose
        // ����ط�����EPnP�������
        compute_pose(mTi);

        // Check inliers
        CheckInliers();
        //���ģ��ѡ�������ڵ��������Ҫ��Ҫ��������
        if (mnInliersi >= mRansacMinInliers)
        {
            // If it is the best solution so far, save it
            //�ѵ�ǰģ������Ϊ���ģ��
            if (mnInliersi>mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                mBestTcw = mTi;
            }

            //��ģ�ͽ��о�������������������ģ����Ȼ�����㣨emmm������ʣ�µ��ڵ�����Ȼ�����Ҫ��ߣ����ͰѾ������ģ�ͷ��أ���������ڵ�Ҳ����vbInliers����
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

     //�������Ժ�����������������ڵ���������������ô�����ֿ��ܣ�һ���ǵ�������ǰ�ҵ������ģ�͵������ģ�;���������������������ô�ã���һ���Ǹɴ�û�ҵ�
    if (mnIterations >= mRansacMaxIts)
    {
        bNoMore = true;
        //����ҵ���һ�����ģ�ͣ���ô�����ģ�͵��ڵ��¼������Ȼ���λ�˷��أ�������������
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
    //���Ǹ�����
    return Sophus::SE3d();
}

bool PnPsolver::Refine()
{
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    //��ȡ�������ڵ�
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

    // ͨ��CheckInliers�����õ���Щinlier�������ᴿ
    mnRefinedInliers = mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    //����ᴿ�õ����ڵ�����Ȼ���������ڵ���������ʾ���ģ�ͻ�������ô������ģ����Ϊ��֡���˶�����
    if (mnInliersi>mRansacMinInliers)
    {
        mRefinedTcw = mTi;
        return true;
    }

    return false;
}

// ͨ��֮ǰ����(R t)�����Щ3D-2D�������inliers
void PnPsolver::CheckInliers()
{
    mnInliersi = 0;

    for (int i = 0; i<N; i++)
    {
        point3d P3Dw = mvP3Dw[i];
        point2d P2D = mvP2D[i];

        // ��3D������������ϵ��ת���������ϵ
        point3d P3DC = mTi * P3Dw;
        P3DC /= P3DC[2];
        float Xc = P3DC[0];
        float Yc = P3DC[1];

        // ���������ϵ�µ�3D�������ͶӰ
        double ue = cx + fx * Xc;
        double ve = cy + fy * Yc;

        // ����в��С
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

// number_of_correspondencesΪRANSACÿ��PnP���ʱʱ3D���2D��ƥ�����
// RANSAC��Ҫ�ܶ�Σ�maximum_number_of_correspondencesΪƥ��������ֵ
// ����������ھ���pws us alphas pcs�����Ĵ�С�����ֻ���𽥱���ܼ�С
// ���maximum_number_of_correspondences֮ǰ���õĹ�С�����������ã������³�ʼ��pws us alphas pcs�Ĵ�С
// set_maximum_number_of_correspondences�ܹ������ù����Σ�һ��ΪRefine��ʹ�ã���һ��Ϊiterate������
void PnPsolver::set_maximum_number_of_correspondences(int n)
{
    if (maximum_number_of_correspondences < n) {
        if (pws != 0) delete[] pws;
        if (us != 0) delete[] us;
        if (alphas != 0) delete[] alphas;
        if (pcs != 0) delete[] pcs;

        maximum_number_of_correspondences = n;
        pws = new double[3 * maximum_number_of_correspondences];// ÿ��3D����(X Y Z)����ֵ
        us = new double[2 * maximum_number_of_correspondences];// ÿ��ͼ��2D����(u v)����ֵ
        alphas = new double[4 * maximum_number_of_correspondences];// ÿ��3D�����ĸ����Ƶ���ϣ����ĸ�ϵ��
        pcs = new double[3 * maximum_number_of_correspondences];// ÿ��3D����(X Y Z)����ֵ
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
    // ����1����һ�����Ƶ㣺����PnP����Ĳο�3D��ļ�������
    cws[0][0] = cws[0][1] = cws[0][2] = 0;
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            cws[0][j] += pws[3 * i + j];

    for (int j = 0; j < 3; j++)
        cws[0][j] /= number_of_correspondences;


    // Take C1, C2, and C3 from PCA on the reference points:
    // ����2�����������������Ƶ㣬C1, C2, C3ͨ��PCA�ֽ�õ�
    // �����е�3D�ο���д�ɾ���(number_of_correspondences *����)�ľ���
    CvMat * PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);

    double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
    CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
    CvMat DC = cvMat(3, 1, CV_64F, dc);
    CvMat UCt = cvMat(3, 3, CV_64F, uct);

    // ����2.1��������pws�еĲο�3D���ȥ��һ�����Ƶ�����꣨�൱�ڰѵ�һ�����Ƶ���Ϊԭ�㣩, ������PW0
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];

    // ����2.2������SVD�ֽ�P'P���Ի��P��������
    // ���������������С�������Ĺ��̣�
    // PW0��ת�ó���PW0
    cvMulTransposed(PW0, &PW0tPW0, 1);
    cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);

    cvReleaseMat(&PW0);

    // ����2.3���õ�C1, C2, C3����3D���Ƶ㣬������֮ǰ�����ĵ�һ�����Ƶ����ƫ����
    for (int i = 1; i < 4; i++) {
        double k = sqrt(dc[i - 1] / number_of_correspondences);
        for (int j = 0; j < 3; j++)
            cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
    }
}

// ����ĸ����Ƶ��ϵ��alphas
// (a2 a3 a4)' = inverse(cws2-cws1 cws3-cws1 cws4-cws1)*(pws-cws1)��a1 = 1-a2-a3-a4
// ÿһ��3D���Ƶ㣬����һ��alphas��֮��Ӧ
// cws1 cws2 cws3 cws4Ϊ�ĸ����Ƶ������
// pwsΪ3D�ο��������
void PnPsolver::compute_barycentric_coordinates(void)
{
    double cc[3 * 3], cc_inv[3 * 3];
    CvMat CC = cvMat(3, 3, CV_64F, cc);
    CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

    // ��һ�����Ƶ������ĵ�λ�ã������������Ƶ��ȥ��һ�����Ƶ�����꣨�Ե�һ�����Ƶ�Ϊԭ�㣩
    // ����1����ȥ���ĺ�õ�x y z��
    // 
    // cws������ |cws1_x cws1_y cws1_z|  ---> |cws1|
    //          |cws2_x cws2_y cws2_z|       |cws2|
    //          |cws3_x cws3_y cws3_z|       |cws3|
    //          |cws4_x cws4_y cws4_z|       |cws4|
    //          
    // cc������  |cc2_x cc3_x cc4_x|  --->|cc2 cc3 cc4|
    //          |cc2_y cc3_y cc4_y|
    //          |cc2_z cc3_z cc4_z|
    for (int i = 0; i < 3; i++)
        for (int j = 1; j < 4; j++) {
            //������Ƶ��ȥ��������
            cc[3 * i + j - 1] = cws[j][i] - cws[0][i];
        }

    //���Ǹ�C�����
    cvInvert(&CC, &CC_inv, CV_SVD);
    double * ci = cc_inv;
    //��ÿһ�����Ӧ��alphaϵ�����м���
    for (int i = 0; i < number_of_correspondences; i++) {
        double * pi = pws + 3 * i;// piָ���i��3D����׵�ַ
        double * a = alphas + 4 * i;// aָ���i�����Ƶ�ϵ��alphas���׵�ַ

                                    // pi[]-cws[0][]��ʾ��pi�Ͳ���1������ͬ��ƽ��
        for (int j = 0; j < 3; j++) {
            a[1 + j] = ci[3 * j] * (pi[0] - cws[0][0]) +
                ci[3 * j + 1] * (pi[1] - cws[0][1]) +
                ci[3 * j + 2] * (pi[2] - cws[0][2]);
        }
        //�ĸ�ϵ���ĺ�Ϊ1
        a[0] = 1.0f - a[1] - a[2] - a[3];
    }
}

// �����С���˵�M����
// ��ÿһ��3D�ο��㣺
// |ai1 0    -ai1*ui, ai2  0    -ai2*ui, ai3 0   -ai3*ui, ai4 0   -ai4*ui|
// |0   ai1  -ai1*vi, 0    ai2  -ai2*vi, 0   ai3 -ai3*vi, 0   ai4 -ai4*vi|
// ����i��0��4
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

// ÿһ�����Ƶ����������ϵ�¶���ʾΪ������������beta����ʽ��EPnP���ĵĹ�ʽ16
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

// ���ĸ����Ƶ���Ϊ��λ������ʾ�µ���������ϵ��3D�������
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
    // ����1�����EPnP�㷨�е��ĸ����Ƶ�
    choose_control_points();
    // ����2��������������ϵ��ÿ��3D����4�����Ƶ����Ա��ʱ��ϵ��alphas����ʽ1
    compute_barycentric_coordinates();

    // ����3������M���󣬹�ʽ(3)(4)-->(5)(6)(7)
    CvMat * M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);

    for (int i = 0; i < number_of_correspondences; i++)
        fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

    double mtm[12 * 12], d[12], ut[12 * 12];
    CvMat MtM = cvMat(12, 12, CV_64F, mtm);
    CvMat D = cvMat(12, 1, CV_64F, d);
    CvMat Ut = cvMat(12, 12, CV_64F, ut);

    // ����3�����Mx = 0
    // SVD�ֽ�M'M
    cvMulTransposed(M, &MtM, 1);

    //����MtM�ǶԳ����������������������ͬ�ģ����Կ��Կ���  xT * M = 0, ������̵���ռ�ΪUT
    cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);//�õ�����ut
    cvReleaseMat(&M);

    double l_6x10[6 * 10], rho[6];
    CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
    CvMat Rho = cvMat(6, 1, CV_64F, rho);

    compute_L_6x10(ut, l_6x10);
    compute_rho(rho);

    //Ϊʲô�����Beta��[4][4]? Beta[0]��ͷ��β��û�ù�ѽ�����Ҷ�Ӧ��rep_errors, Rs, ts��ֻ��������
    double Betas[4][4], rep_errors[4];
    Sophus::SE3d Ts[4];

    // ����ʲô�����������������N=4������ⲿ��betas�����ȫ���������г�ͻ��
    // ͨ���Ż��õ�ʣ�µ�betas
    // ������R t

    //���������find_betas_approx�Ƿֱ����£�ÿһ����Ӧ��Ӧ����N = 4�� 2�� 3�����

    // EPnP���Ĺ�ʽ10 15���˴�Ϊ�ⲩ��ע�ͣ��Ҹо��õ��Ĳ��ǹ�ʽ10����Ϊ��δ���ֱ��copy��OpenCV\calib3D\epnp.cpp����Ĵ��룬�����1���ܱ�ʾ�Ľ�����һ����ţ���
    //�Ҵ󵨲²����Ǽ�����������N��ֵΪ4����Beta��4��ֵ��gauss_newton������Ϊ�˵����õ���С������ʵ�ʴ�������ĵ��������ǹ̶��ģ����ܽ�����Ϊ�˵õ�һ���ȽϺ�������
    find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
    gauss_newton(&L_6x10, &Rho, Betas[1]);
    rep_errors[1] = compute_R_and_t(ut, Betas[1], Ts[1]);

    // EPnP���Ĺ�ʽ11 15
    // ���Ǽ�����������N��ֵΪ2����Beta��2��ֵ
    find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
    gauss_newton(&L_6x10, &Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(ut, Betas[2], Ts[2]);

    // ���Ǽ�����������N��ֵΪ3����Beta��3��ֵ
    find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
    gauss_newton(&L_6x10, &Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(ut, Betas[3], Ts[3]);

    //Ѱ�������С�����
    int N = 1;
    if (rep_errors[2] < rep_errors[1]) N = 2;
    if (rep_errors[3] < rep_errors[N]) N = 3;

    //�������С���������Ϊ��ģ�͵�������
    copy_R_and_t(Ts[N], T);

    //û���ĸ�������Ҫ���������ֵ��ΪʲôҪ���أ�
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

//������ͶӰ���
double PnPsolver::reprojection_error(Sophus::SE3d T)
{
    double sum2 = 0.0;

    for (int i = 0; i < number_of_correspondences; i++) {
        double * pw = pws + 3 * i;
        //�����ͼ�����������ϵ�µ�x, y�Լ�z.inverse

        point3d P3DC = mTi * point3d(pw[0], pw[1], pw[2]);
        P3DC /= P3DC[2];
        float Xc = P3DC[0];
        float Yc = P3DC[1];
        
        
        double ue = cx + fx * Xc;
        double ve = cy + fy * Yc;
        double u = us[2 * i], v = us[2 * i + 1];

        sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
    }
    //����һ��ƽ�����
    return sum2 / number_of_correspondences;
}

// ������������ϵ�µ��ĸ����Ƶ�����������¶�Ӧ���ĸ����Ƶ㣨����������ϵ���ĸ����Ƶ���ͬ�߶ȣ�����ȡR t��Ҳ����ICP����
void PnPsolver::estimate_R_and_t(Sophus::SE3d& T)
{
    //���е�ͼ������ģ�pc0���������ϵ�µ����ģ�pw0����������ϵ�µ����ģ�
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
        //abtΪ
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

//��������Ĺ��ܿ���ȥӦ���ǣ�����������ϵ�µ����Ϊ��������ѵ�����ȡ����
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
    //c_c = ����UT
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
    //���for���ǰ�����Copy������
    for (int i = 0; i < 6; i++) {
        cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
        cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
        cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
        cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
    }

    cvSolve(&L_6x4, Rho, &B4, CV_SVD);
    //�����b4[0]����beta1 * beta1������������beta2��beta3��beta4
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

// ���㲢������L�������֪���ṹ�����԰�����������Ĺ�ʽ(12)���Ѧµ��������4��Ȼ��չ�����Ϳ���֪������Ǹ�forѭ������ô�õ�����
void PnPsolver::compute_L_6x10(const double * ut, double * l_6x10)
{
    const double * v[4];

    v[0] = ut + 12 * 11;
    v[1] = ut + 12 * 10;
    v[2] = ut + 12 * 9;
    v[3] = ut + 12 * 8;

    /*
    4��Ӧ�Ŧ���4��ֵ��6��Ӧ�������������6��������������������i��j��3��Ӧ������ά������
    */
    double dv[4][6][3];

    for (int i = 0; i < 4; i++) {
        //�����a���������е�i��b���������е�j
        int a = 0, b = 1;
        for (int j = 0; j < 6; j++) {
            dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
            dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
            dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

            b++;              //
            if (b > 3) {      //  �����о��Ǽ�����6����Ϸ�ʽ�
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

// �����ĸ����Ƶ����������ľ��룬�ܹ�6�����룬��Ӧ��������Ĺ�ʽ(10)�ĵ�ʽ������һ��
void PnPsolver::compute_rho(double * rho)
{
    rho[0] = dist2(cws[0], cws[1]);
    rho[1] = dist2(cws[0], cws[2]);
    rho[2] = dist2(cws[0], cws[3]);
    rho[3] = dist2(cws[1], cws[2]);
    rho[4] = dist2(cws[1], cws[3]);
    rho[5] = dist2(cws[2], cws[3]);
}

//���������������Ϊ�������Ǹ�����gauss_newton�������ģ�Ŀ���Ǽ������ CvMat * b �Լ�����L����Ԧµĵ��� CvMat * A
void PnPsolver::compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
    double betas[4], CvMat * A, CvMat * b)
{
    for (int i = 0; i < 6; i++) {
        const double * rowL = l_6x10 + i * 10;
        //�����rowA��L_6x10��betas��һ�׵�
        double * rowA = A->data.db + i * 4;

        rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
        rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
        rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
        rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

        /*����ط�Ϊ�˼���һ�¼��䣬��ϰһ�°ɣ�
        rho��ʾ���ǿ��Ƶ� i �� j ����������ϵ�µľ��룬���ھ����任�����������Ӧ���Ƶ�ľ����ǲ���ģ�����������errorΪ��������ϵ�µĵ�ľ����ȥ
        ���ǹ����ʮ�֦ºͶ�ӦL�˻� L�� �Ĳ�ԣ������Ǹ� L�� = �ѡ�Ȼ��forѭ�����6����i��j���������״̬������Ŀ��Կ���������Ĺ�ʽ(15)
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

//���Ǹ�˹ţ�ٷ�����˹ţ�ٷ���Ӧ���� JTJ x = JT e��
void PnPsolver::gauss_newton(const CvMat * L_6x10, const CvMat * Rho,
    double betas[4])
{
    const int iterations_number = 5;

    double a[6 * 4], b[6], x[4];
    CvMat A = cvMat(6, 4, CV_64F, a);
    CvMat B = cvMat(6, 1, CV_64F, b);
    CvMat X = cvMat(4, 1, CV_64F, x);

    for (int k = 0; k < iterations_number; k++) {
        //����ط�������Ľ����ݵ�ָ�밴��double����ȡ��data.db����һ��doubleָ�룬��data��һ�������塣
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
