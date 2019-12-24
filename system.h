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
    构造函数：负责初始化系系统，包括相机类型，相机内参。所有参数均写在一个文件里面，文件内容应包括
        sensor : 1(单目)；2（双目）；3（RGBD）
        fx
        fy
        cx
        cy
        baseline: 双目为基线，RGBD为深度尺度，单目为0
        文件地址：该地址中应当包含一个associate.txt的文件，该文件中包含了全部图像的文件名
    run：负责整个系统的运行
    */
    class system {
    public:
        //
        system(const string& filename, int argc, char** argv);
        ~system() { img_dir.close(); }
        //负责运行整个系统
        void run(bool trackOnly = false);
        enum sensor
        {
            Mono = 1,
            Stereo = 2,
            RGBD = 3
        };
        //该函数为增加双目相机模块时的测试函数，在实际运行中不起任何作用
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
        //记录读取到多少帧
        std::atomic<int> frame_count;
        //记录图像是否读取完毕
        std::atomic<bool> read_over;
        std::ifstream img_dir;
        std::shared_ptr<cornerMatcher> mpMatcher;
        std::shared_ptr<mapDrawer> mpMapDrawer;
        std::shared_ptr<clvo::map> mpMap;
        std::shared_ptr<directPoseOnlyBackEnd> mpOptimizer;


        string file_dir;
        //载入图片线程，根据文件地址信息载入图片
        void loadImage();
        //对每一帧进行处理，主要包括线条提取，特征点提取和双目相机的深度估计
        void processFrame();
        //绘图线程，使用OpenGL绘制相机姿态
        void drawMap();
    };

}