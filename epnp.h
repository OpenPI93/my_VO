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
        //���ò��������cpp�ļ�����Ľ���
        void SetRansacParameters(double probability = 0.99, int minInliers = 8, int maxIterations = 300, int minSet = 4, float epsilon = 0.4,
            float th2 = 5.991);
        
        //tracking����ʹ�ã�����ֵΪ���λ��
        Sophus::SE3d iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers);

    private:
        //�����ж��ٵ����ڵ㣬���е�ƥ�䶼Ҫ���뵽�ڵ����С���ֵ�õ�����mvMaxError
        void CheckInliers();
        /*����ģ�ͣ��������Ϊ��
        �ȸ����ڵ�ĸ�������һ��pws��alphas��Щ�����ĳߴ磬ʹ�óߴ粻С���ڵ����
        ������ɸѡ���ڵ�һ����һ��ģ�ͣ���iterate��������ֻ��4���㽨��ģ�ͣ�
        ����ж��ٵ����ڵ㣬����ڵ�����㹻�࣬��ô����true�����򷵻�false
        ��˵һ�£��������ֵΪtrue����ôiterate������ֱ�ӷ��ؾ������ģ�������������T
        */
        bool Refine();

        // Functions from the original EPnP code
        //�޸�maximum_number_of_correspondences��ֵ�����n > maximum_number_of_correspondences������Ӧ�ĸ���pws��us��alphas�Լ�pcs��size
        void set_maximum_number_of_correspondences(const int n);//��
        //number_of_correspondences��Ϊ0��iterate������RANSACʹ��
        void reset_correspondences(void);//��
        //��pws��us�����һ�Ե㣬ͬʱnumber_of_correspondences++�����������ǰ���������ģ�͵ĵ㱣��������ͨ������¼��ĸ���Ϳ��ԣ�Refine���������Ǽ�ȫ���ڵ�
        void add_correspondence(const double X, const double Y, const double Z,
            const double u, const double v);//��

        /*
         �ú�������compute_pose�����У��������㵱ǰģ�͵�λ�˲����ص�ǰģ�͵�����compute_pose���ܹ�ʹ�������Σ�ÿһ��find_betas_approx��Ҫ��¼һ��ģ���������ģ�ͼ�¼��R �� t���书���Ǽ�����Ƶ����ꡢ����alpha�����������M���󣩡�����£��͡����λ��
          ����ֵΪһ���յ���ת�����ƽ�Ʊ���
        */
        double compute_pose(Sophus::SE3d& T);//��
       
         //�ں���compute_R_and_t�����һ���е��ã�Ŀ��Ϊ�������е�ͼ��ʹ�õ�ǰģ�͵���ͶӰ������ֵΪƽ�����
        double reprojection_error(Sophus::SE3d T);//��

        //�����ĸ�������Ƶ������
        void choose_control_points(void);//��
        //����ÿһ����ͼ���Ӧ��alpha
        void compute_barycentric_coordinates(void);//��
        //����M���󣬾����Ǹ�n(��ͼ������) * 12 �ľ���
        void fill_M(CvMat * M, const int row, const double * alphas, const double u, const double v);//��
        //������Ƶ����������µ����꣬���㹫ʽΪc_c = ����UT��������������Ĺ�ʽ(8)
        void compute_ccs(const double * betas, const double * ut);//��
         //���ݿ��Ƶ������������ͶӰ�����ά���꣬�õ��Ĺ�ʽΪ��������Ĺ�ʽ(2)��������c_c��alpha���
        void compute_pcs(void);//��
        //������compute_pcs�����ͼ����������������ͼ�����������ϵ�µ����Ϊ�����򽫵�ͼ�����������ϵ�µ�����ȡ��
        void solve_for_sign(void);//��

        //����N������������Ӧ�Ħ£����������ֱ��ӦN = 4�� 2�� 3
        void find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho, double * betas);//��
        void find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho, double * betas);//��
        void find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho, double * betas);//��


        //���������ĵ��
        double dot(const double * v1, const double * v2);//��
        //����������ľ����ƽ�����м���ƽ��
        double dist2(const double * p1, const double * p2);//��

        //�����й�ʽ(13)�õ���һ������ L��=��
        //���㹫ʽ�еĦ�
        void compute_rho(double * rho);//��
        //���㹫ʽ�е�L
        void compute_L_6x10(const double * ut, double * l_6x10);//��

        //�����ֻ�ṩһ�������Ľӿڣ�����Ĺ�����compute_A_and_b_gauss_newton�����һ�׵������ļ����Լ�qr_solve���ⷽ��
        void gauss_newton(const CvMat * L_6x10, const CvMat * Rho, double current_betas[4]);//��
        //��������gauss_newton����ʹ��
        void compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
            double cb[4], CvMat * A, CvMat * b);//��

        //��������ṩ��һ���ӿڣ�������Ϊ������compute_ccs����Ƶ��������꣬����compute_pcs���ͼ���������꣬����solve_for_sign��鲢�������
        //������estimate_R_and_t���λ�ˣ�Ȼ�󷵻�reprojection_error��Ҳ�������ģ�͵���ͶӰ���
        double compute_R_and_t(const double * ut, const double * betas, Sophus::SE3d& T);//��
        //����ICP����������˶�����ϸ�㷨���Բο��߲���slamʮ�Ľ��������е�ICP��SVD����
        void estimate_R_and_t(Sophus::SE3d& T);//��

         //һ���򵥵ĸ�ֵ����������ĺ�������compute_pose�����õ�
        void copy_R_and_t(const Sophus::SE3d& T_src,
            Sophus::SE3d& T_des);

        void qr_solve(CvMat * A, CvMat * b, CvMat * X);

        //��ת����ת��Ԫ��
        void mat_to_quat(const double R[3][3], double q[4]);

        double fx, fy, cx, cy;

        //alphas�ľ���������ÿһ��ransac���������ȷ��
        //pwsΪ��ͼ����������꼯�ϡ�usΪ��ά���꼯�ϡ�alphasΪÿ����ͼ���Ӧ���ĸ�Ȩ�صļ��ϡ�pcsΪ��ͼ���������꼯��
        //���ĸ�������������ʼ��4���������ӣ������������ھ����׶ν��У���Ϊ��Ҫ����ǰģ�͵�ȫ���ڵ㶼����ģ�͵Ľ���
        double * pws, *us, *alphas, *pcs;
        //��¼�������ɵ�������Լ���Ӧ��Ϣ������
        int maximum_number_of_correspondences;
        //�����������Ҫ������ʾÿ���ö��ٵ����Ϣ���EPnP���⣬���磬��ͨ��ɸѡģ�ͽ׶�ֻ��Ҫ4����ԣ������ھ����׶�����Ҫ��ȫ���ĳ���ɸѡ���ڵ㽨��ģ��
        int number_of_correspondences;

        //��������ϵ���������ϵ�µ��ĸ����Ƶ�w��ʾworld��c��ʾcamera
        double cws[4][3], ccs[4][3];

        //һ��û�б��õ��ı���������ȥӦ�����ĸ����Ƶ�����������ϵ�¹��ɾ��������ֵ��
        double cws_determinant;

        //��¼�������ĵ�ͼ����Ϣ����Ҫ���������Ǹ�size����Ϊ3D-2Dƥ����Ϣ�ڹ��캯���о��Ѿ��������������Ϣ�Ͷ�Ӧ��֡��Ϣ���������ˣ������������������ȡsize
        vector<point3d> mvpMapPointMatches;

        // 2D Points
        //��ͼ���Ӧ��ÿһ��ͶӰ���2D����
        vector<point2d> mvP2D;
        
        // 3D Points
        //��ͼ�����������
        vector<point3d> mvP3Dw;

        // Index in Frame
        //��¼��ͼ���Ӧ��Frame�е���������ţ�������Щ��ͼ��isBad������������洢����Ų���������
        vector<size_t> mvKeyPointIndices;

        // Current Estimation
        Sophus::SE3d mTi;
        Sophus::SE3d mTcwi;
        //��¼��ǰģ���е��ڵ㣬����˵�ǽ��ڵ��Ӧ����ŵ�Ԫ����Ϊtrue����ߴ�ΪN����ƥ���Եĸ������ڹ��캯����resize
        vector<bool> mvbInliersi;
        //��¼��ǰģ�����ڵ�ĸ�������ֵ��CheckInliers()����Ϊ0����ʼ�ۼӣ�Ȼ����iterate���жϸ�ģ���Ƿ�Ϊ���ģ��
        int mnInliersi;

        // ��¼��ǰRANSAC�ĵ�������������iterate������
        int mnIterations;
        //������¼��ǰ���ģ�͵��ڵ㣬���ڵ����Ӧ��λ��Ϊtrue
        vector<bool> mvbBestInliers;
        //��¼���ģ�������м����ڵ�
        int mnBestInliers;
        Sophus::SE3d mBestTcw;

        // Refined
        //�������λ�ˣ�ͨ��������λ��ֱ�ӱ����������
        Sophus::SE3d mRefinedTcw;
        //��������ڵ�
        vector<bool> mvbRefinedInliers;
        //��������ڵ����
        int mnRefinedInliers;

        // Number of Correspondences
        //���ж�ά���������, ����˵��ƥ��ĸ���
        int N;

        // Indices for random selection [0 .. N-1]
        //��¼���кϸ�ĵ�ͼ�����������mvP3Dw���ʹ�ã�������Ǳ�Ȼ������
        vector<size_t> mvAllIndices;

        // RANSAC probability
        //ֻ��SetRansacParameters�������õ�������������Ҫ�������ٴε�ʱ��ʹ��
        double mRansacProb;

        // RANSAC����������Ҫ�ö��ٸ��ڵ����ģ��
        int mRansacMinInliers;

        // RANSAC���������ٴ�
        int mRansacMaxIts;

        // RANSAC expected inliers/total ratio
        // ��SetRansacParameters�����м��㣬������¼���ڵ����/�ܵ��������������ڵ������һ�����û��ʼ������Ǹ��ڵ������������캯���еĽ���
        float mRansacEpsilon;

        // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
        
        float mRansacTh;

        // RANSAC Minimun Set used at each iteration
        // ����ģ����Ҫ�����ٵ�������ڹ��캯�������ã���EPnP���������һ�����4
        int mRansacMinSet;

       

    };

} //namespace ORB_SLAM

