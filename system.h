#include "basetype.h"
#include <fstream>
#include "frame.h"
#include "match.h"
#include "map.h"
#include <memory>
#include <atomic>
#include "optimizer.h"

namespace clvo {
    /*
    class system
    ���캯���������ʼ��ϵϵͳ������������ͣ�����ڲΡ����в�����д��һ���ļ����棬�ļ�����Ӧ����
        sensor : 1(��Ŀ)��2��˫Ŀ����3��RGBD��
        fx
        fy
        cx
        cy
        baseline: ˫ĿΪ���ߣ�RGBDΪ��ȳ߶ȣ���ĿΪ0
        �ļ���ַ���õ�ַ��Ӧ������һ��associate.txt���ļ������ļ��а�����ȫ��ͼ����ļ���
    run����������ϵͳ������
    */
    class system {
    public:
        //
        system(const string& filename, int argc, char** argv);
        ~system() { img_dir.close(); }
        //������������ϵͳ
        void run(bool trackOnly = false);
        enum sensor
        {
            Mono = 1,
            Stereo = 2,
            RGBD = 3
        };
        //�ú���Ϊ����˫Ŀ���ģ��ʱ�Ĳ��Ժ�������ʵ�������в����κ�����
        void stereoTest();
        sensor mSensor;
        double fx;
        double fy;
        double cx;
        double cy;
        double baseline;
        vector< std::shared_ptr<frame> > vpframes;
        list<groundTruth> truths;
        list<groundTruth> myTruths;
        point3d startPosition;
    protected:
        //��¼��ȡ������֡
        std::atomic<int> frame_count;
        //��¼ͼ���Ƿ��ȡ���
        std::atomic<bool> read_over;
        std::ifstream img_dir;
        std::shared_ptr<cornerMatcher> mpMatcher;
        std::shared_ptr<mapDrawer> mpMapDrawer;
        std::shared_ptr<clvo::map> mpMap;
        std::shared_ptr<directPoseOnlyBackEnd> mpOptimizer;


        string file_dir;
        //����ͼƬ�̣߳������ļ���ַ��Ϣ����ͼƬ
        void loadImage();
        //��ÿһ֡���д�����Ҫ����������ȡ����������ȡ��˫Ŀ�������ȹ���
        void processFrame();
        //��ͼ�̣߳�ʹ��OpenGL���������̬
        void drawMap();
    };

}