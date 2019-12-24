#include "system.h"

#include <chrono>
using namespace std::chrono;

#define COMPUTE_RMSE
//#define STOP_PRE_EIGHTY
//#define USE_BACKEND

clvo::system::system(const string& filename, int argc, char** argv) :frame_count(0), read_over(false){

    mpMapDrawer = std::shared_ptr<mapDrawer>(new mapDrawer(argc, argv));
    

    std::ifstream config_file;
    config_file.open(filename.c_str());

    if (!config_file.is_open()) {
        cout << filename.c_str() << " is not exit\n";
        exit(1);
    }
    
    string s;
    config_file >> s;
    int _sensor;
    try {
        _sensor = std::stoi(s);
        if (1 == _sensor) {
            mSensor = Mono;
        }
        else if (2 == _sensor) {
            mSensor = Stereo;
        }
        else if (3 == _sensor) {
            mSensor = RGBD;
        }
        else {
            cout << "sensor is not exit\n";
            exit(1);
        }
        config_file >> s;
        fx = std::stod(s);
        config_file >> s;
        fy = std::stod(s);
        config_file >> s;
        cx = std::stod(s);
        config_file >> s;
        cy = std::stod(s);
        config_file >> s;
        baseline = std::stod(s);
    }
    catch (std::exception& e) {
        cout << e.what() << endl;
        
        exit(1);
    }
    
    mpOptimizer = std::shared_ptr<directPoseOnlyBackEnd>(new directPoseOnlyBackEnd(std::vector<double>{fx, cx, fy, cy, baseline}));

    config_file >> s;
    img_dir.open(s + "associate.txt");
    if (!img_dir.is_open()) {
        cout << s << "associate.txt is not exit\n";
        exit(1);
    }
    file_dir = s;
    config_file.close();

#ifdef COMPUTE_RMSE
    std::ifstream groundtruth;
    groundtruth.open(s + "groundtruth.txt");
    if (!groundtruth.is_open()) {
        cout << "can not find the groundtruth file\n";
    }
    else {
        cout << "loading groundtruth ...\n";
        groundtruth.seekg(std::ios_base::beg);

        string s;
        //ǰ����û����Ϣ
        getline(groundtruth, s);
        getline(groundtruth, s);
        getline(groundtruth, s);

        point3d sum(0, 0, 0);

        while (!groundtruth.eof()) {
            groundtruth >> s;
            
            auto time = std::stold(s);
            groundtruth >> s;
            auto tx = std::stod(s);
            groundtruth >> s;
            auto ty = std::stod(s); 
            groundtruth >> s;
            auto tz = std::stod(s);

            groundtruth >> s;
            auto qx = std::stod(s);
            groundtruth >> s;
            auto qy = std::stod(s);
            groundtruth >> s;
            auto qz = std::stod(s);
            groundtruth >> s;
            auto qw = std::stod(s);
            Eigen::Quaterniond q1(qw, qx, qy, qz);
            Eigen::Vector3d t1(tx, ty, tz);
            Sophus::SE3d se3_qt1(q1, t1);

            point3d transl = se3_qt1.translation();
            
            truths.push_back(groundTruth(time, _STD move(transl)));
        }
        
    }
#endif
}

void clvo::system::run(bool trackOnly) {
    //��һ��������һ��ͼƬ��ȡ��Ȼ�󴴽�frame
    img_dir.seekg(std::ios_base::beg);//��λ���ļ���ͷ

    string s, ts;
    img_dir >> ts;//timestamp
    img_dir >> s;//rgb�ļ���ַ

    auto rgb = cv::imread(file_dir + s, CV_LOAD_IMAGE_GRAYSCALE);
    img_dir >> s;
    img_dir >> s;//depth�ļ���ַ
    auto depth = cv::imread(file_dir + s, CV_LOAD_IMAGE_UNCHANGED);

    drawMap();

    std::shared_ptr<frame> pframe(new frame(rgb, depth, this));
    pframe->index = 0;
    
#ifdef COMPUTE_RMSE
    pframe->timestamp = std::stold(ts);
    point3d realPosition;
    for (auto gtruth : truths) {
        if (gtruth.timestamp > pframe->timestamp) {
            startPosition = gtruth.position;
            break;
        }
    }
    myTruths.push_back(groundTruth(pframe->timestamp, point3d(0, 0, 0)));
    
#endif

    vpframes.push_back(pframe);//�ѵ�һ��ͼƬ����vpframes
    
    //�ڶ�����������ÿһ֡ͼƬ�����뵽vpframes
    std::thread t_ldimg{ &system::loadImage, this };//���õ������߳�����ȡͼ�񣬼ӿ��ٶ�
    std::thread t_psf{ &system::processFrame, this };
    std::thread t_map{ &mapDrawer::run, mpMapDrawer.get() };

    t_ldimg.join();
    t_psf.join();
    t_map.join();

    return;
}

void clvo::system::loadImage() {

    string s, ts;
    int index = 0;
    while (!img_dir.eof()) {
        img_dir >> ts;//rgb�ļ���, Ҳ��ʱ���
        img_dir >> s;//rgb�ļ���ַ

        auto rgb = cv::imread(file_dir + s, CV_LOAD_IMAGE_GRAYSCALE);
        img_dir >> s;
        img_dir >> s;//depth�ļ���ַ
        cv::Mat depth;
        //RGBD����ʮ��λ���ͼ��˫Ŀ�������8λ�Ҷ�ͼ����Ŀ����
        if (mSensor == RGBD) {
            depth = cv::imread(file_dir + s, CV_LOAD_IMAGE_UNCHANGED);
        }
        else if (mSensor == Stereo) {
            depth = cv::imread(file_dir + s, CV_LOAD_IMAGE_GRAYSCALE);
        }
        else if (mSensor == Mono) {
            depth = cv::Mat();
        }

        std::shared_ptr<frame> pframe(new frame(rgb, depth, this));
        pframe->index = ++index;

#ifdef COMPUTE_RMSE
        pframe->timestamp = std::stold(ts);
#endif

        /*cv::imshow("img", rgb);
        cv::waitKey(1);*/
        vpframes.push_back(pframe);
        frame_count++;
    }
    read_over = true;
    int imgcount = vpframes.size();
#ifdef _WIN64
    HANDLE hout;
    COORD coord;
    coord.X = 0;
    coord.Y = 4;
    hout = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO cursor_info = { 1, 0 };
    SetConsoleCursorInfo(hout, &cursor_info);
    SetConsoleCursorPosition(hout, coord);

#elif __linux__
    printf("\033[%d;%dH", 4, 0);
    printf("\033[?25l");
#endif
    
    printf("image read over, there %s %d image%s\r", imgcount > 1 ? "are" : "is", imgcount, imgcount > 1 ? "s" : "");
    
}

void clvo::system::processFrame() {
    
    mpMatcher = std::shared_ptr<cornerMatcher>(new cornerMatcher(this));
    //��1������ȡ��һ֡������
    vpframes[0]->getLines(40, 180);
    int k = frame_count;
    
    std::vector<double> usingtimes;

    vpframes[1]->getLines(40, 180);

    vpframes[1]->T = mpMatcher->computePose(vpframes[0].get(), vpframes[1].get(), nullptr);
    mpMapDrawer->mMap->updateCamera(vpframes[1]->T);

#ifdef COMPUTE_RMSE
    point3d realPosition;
    realPosition = vpframes[1]->T * myTruths.rbegin()->position;
    myTruths.push_back(groundTruth(vpframes[1]->timestamp, std::move(realPosition)));
#endif

   // cout << vpframes[1]->T.matrix3x4() << endl << endl;
    //�����ȡ���˶�ȡ����ҵ�ǰ�����֡Ϊvector�����һ֡��ѭ������
    for (int i = 2; (i < frame_count || (!read_over)); ++i) {

        auto start = system_clock::now();
        //���׷���ٶȴ��ڶ�ȡ�ٶȣ���ͨ���������ȴ�ͼƬ�Ķ�ȡ
        if (i >= vpframes.size()) {
            while (i >= frame_count) {
                frame_count;
                Sleep(20);
            }
        }
        
        vpframes[i]->getLines(40, 180);
        
        vpframes[i]->T = mpMatcher->computePose(vpframes[i - 1].get(), vpframes[i].get(), vpframes[i - 2].get());
#ifdef USE_DRIECTBACKEND
        {
            std::vector<std::shared_ptr<frame>> tmpvec;
            tmpvec.push_back(vpframes[i - 2]);
            tmpvec.push_back(vpframes[i - 1]);
            tmpvec.push_back(vpframes[i]);

            mpOptimizer->run(tmpvec);
        }
#endif

#ifdef COMPUTE_RMSE
        point3d realPosition;
        realPosition = vpframes[i]->T * myTruths.rbegin()->position;
        myTruths.push_back(groundTruth(vpframes[i]->timestamp, std::move(realPosition)));
#endif
#ifdef STOP_PRE_EIGHTY
         if (!(i % 80)) {
            cv::waitKey();
        }
#endif 

          mpMapDrawer->mMap->updateCamera(vpframes[i]->T);

          //��������Ϣ
#ifdef _WIN64
          HANDLE hout;
          COORD coord;
          coord.X = 0;
          coord.Y = 2;
          hout = GetStdHandle(STD_OUTPUT_HANDLE);
          CONSOLE_CURSOR_INFO cursor_info = { 1, 0 };
          SetConsoleCursorInfo(hout, &cursor_info);
          SetConsoleCursorPosition(hout, coord);
          
#elif __linux__
          printf("\033[%d;%dH", 4, 0);
          printf("\033[?25l");
#endif

          auto end = system_clock::now();
          auto duration = duration_cast<microseconds>(end - start);
          double usetime = (double(duration.count()) * microseconds::period::num / microseconds::period::den);
          usingtimes.push_back(usetime);
          printf("FPS��%3.2lf    ", 1.0 / usetime);
        //cout << vpframes[i]->T.matrix3x4() << endl << endl;
    }
#ifdef COMPUTE_RMSE
    std::ofstream write;
    point3d sum(0, 0, 0);
   /* for (auto truth : myTruths) {
        sum += truth.position;
    }
    sum /= myTruths.size();
    for (auto truth : myTruths) {
       truth.position -= sum;
    }*/
    write.open("truth.txt", std::ios::out);
    auto real = truths.begin();
    for (auto my = myTruths.begin(); my != myTruths.end();) {
        if (real->timestamp >= my->timestamp) {
            write << my->timestamp << "  " << my->position[0] << "  " << my->position[1] << "  " << my->position[2] << endl <<
                "real:  " << real->position[0] << "  " << real->position[1] << "  " << real->position[2] << endl;
            ++my;
        }
        ++real;
    }
    write.close();

    write.open("times.txt", std::ios::out);
    for (auto tim : usingtimes) {
        write << tim << endl;
    }
    write.close();

    cout << "ground is writed over\n";
#endif
    
}

void clvo::system::drawMap() {
    mpMap = std::shared_ptr<clvo::map>(new map());

    clvo::mapDrawer::mMap = mpMap.get();
    //mpMapDrawer->run();
}


void clvo::system::stereoTest() {

    mpMatcher = std::shared_ptr<cornerMatcher>(new cornerMatcher(this));

    loadImage();
    //ͼƬ����ɹ�
    vpframes[0]->getLines(40, 180);
    for (int i = 1; i < vpframes.size(); ++i) {
        vpframes[i]->getLines(40, 180);
        vpframes[i]->T = mpMatcher->ICPSolver(mpMatcher->match(vpframes[i - 1].get(), vpframes[i].get()));
        cout << vpframes[i]->T.matrix3x4() << endl << endl;
    }
}