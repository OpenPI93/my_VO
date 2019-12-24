//#include <opencv.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <exception>
#include <atomic>
#include <windows.h>

#include "system.h"
#include "basetype.h"
#include "match.h"
#include "map.h"
using namespace std;
using namespace cv;
#include <random>

using namespace chrono;

clvo::map* clvo::mapDrawer::mMap = nullptr;

int main(int argc, char** argv) {

    clvo::system SLAM("1.txt", argc, argv);
    SLAM.run();

    for (auto truth : SLAM.myTruths) {
        cout << truth.timestamp << "  " << truth.position[0] << "  " << truth.position[1] << "  " << truth.position[2] << endl;
    }
    /*clvo::system SLAM("2.txt", argc, argv);
    SLAM.stereoTest();*/
    system("pause");

    //clvo::map *pmp = new clvo::map();

    //clvo::mapDrawer::mMap = pmp;

    //clvo::mapDrawer md(argc, argv);
    //for (int i = 0; i < 4; ++i) {
    //    cout << md.mMap->camera[i] << endl;
    //}

    //std::thread t_draw{ &clvo::mapDrawer::run, &md };
    //
    //t_draw.join();

    //delete pmp;
    return 0;
    
}
