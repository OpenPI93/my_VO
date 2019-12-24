#pragma once
#include<GL/glut.h>
#include "basetype.h"

namespace clvo {
    class map {
    public:
        map();
        void updateCamera(Sophus::SE3d T);
        vector<point3d> camera;
        vector<point3d> points;
    };

    class mapDrawer {
    public:
        mapDrawer(int argc, char** argv);
        ~mapDrawer() {}
        void run();
        static map* mMap;
        static double eye[16];
    protected:
        static void draw();
        static void reshape(int, int);
        static void timer(int value);
        static void mouse(int button, int state, int x, int y);

        static double depth;
        
    };
}