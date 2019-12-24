#include "map.h"
#include <Eigen\Geometry>

double clvo::mapDrawer::depth = 1;

double clvo::mapDrawer::eye[] =
{ 1, 0, 0, 0
, 0, 1, 0, 0
, 0, 0, 1, 0
, 0, 0, 0, 1 };


clvo::mapDrawer::mapDrawer(int argc, char** argv) {
    
    glutInit(&argc, argv);

}

void clvo::mapDrawer::run() {

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
        glutInitWindowSize(800, 500);
        glutInitWindowPosition(600, 50);
    glutCreateWindow("game");

    glutDisplayFunc(this->draw);
    glutReshapeFunc(reshape);
    glutTimerFunc(10, timer, 1);

    glutMainLoop();
}

void clvo::mapDrawer::draw() {
    
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);  

    glLoadIdentity();
    glPushMatrix();
    glViewport(0, 0, 800, 500);
    gluPerspective(75, (double)8 / 5, 0.1, 10);
    gluLookAt(0, 0, depth, 0, 0, -100, 0, 1, 0);
    glMultMatrixd(eye);
    
    glColor3f(1, 1, 1);
    glBegin(5);
    {

        /*for (auto k : mMap->camera) {
            glVertex3d(-k[0], k[1], k[2]);
        }*/
        
    }glEnd();

    glPointSize(2.0);
    glBegin(GL_POINTS); {
        
        for (auto k : mMap->points) {
            glVertex3d(-k[0], k[1], k[2]);
        }

    }glEnd();

    glPopMatrix();
    glutSwapBuffers();
}

void clvo::mapDrawer::reshape(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 0, 0, 0, 100, 0, 1, 0);

    glutPostRedisplay();
}


void clvo::mapDrawer::timer(int value)
{
    glutPostRedisplay();
    glutTimerFunc(1, timer, value);
}

void clvo::mapDrawer::mouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON && state == 0)
    {

    }
    if (button == 3 && state == GLUT_UP)
    {
        eye[14] += 0.5;
        depth += 0.2;
    }
    if (button == 4 && state == GLUT_UP)
    {
        eye[14] -= 0.5;
        depth -= 0.2;
    }
    //cout << depth << endl;
    glutPostRedisplay();
}

clvo::map::map() {

    camera.push_back(point3d(-0.2, 0.2, 0.0));
    camera.push_back(point3d(0.2, 0.2, 0.0));
    camera.push_back(point3d(0.2, -0.2, 0.0));
    camera.push_back(point3d(-0.2, -0.2, 0.0));
    camera.push_back(point3d(-0.2, 0.2, 0.0));

    /*camera.push_back(point3d(0.2, 0.3, -0.1));
    camera.push_back(point3d(0.3, 0.3, -0.1));
    camera.push_back(point3d(0.3, 0.2, -0.1));
    camera.push_back(point3d(0.2, 0.2, -0.1));
    camera.push_back(point3d(0.2, 0.3, -0.1));*/

    double size = 0.015;

    std::ifstream ifs("point.txt");
    if (ifs.is_open()) {
        double x, y, z, r, g, b;

        while (!ifs.eof()) {
            ifs >> x >> y >> z /*>> r >> g >> b*/;
            points.push_back(point3d(x * size, y * size, z * size));
        }
        
    }

}

void clvo::map::updateCamera(Sophus::SE3d T) {


    //T = T.inverse();
    //point3d trans = T.translation();
    T = T.inverse();
    Eigen::Matrix<double, 6, 1> tmp = T.log();
    tmp[3] = -tmp[3];
    tmp[4] = -tmp[4];
    tmp[5] = -tmp[5];

    T = Sophus::SE3d::exp(tmp);
    for (int i = 0; i < camera.size(); ++i) {
        camera[i] = T * camera[i];
    }

    for (int i = 0; i < points.size(); ++i) {
        points[i] = T * points[i];
    }

    /*T = T.inverse();

    Eigen::Matrix4d trans = T.matrix();

    Eigen::Matrix4d eyes;

    for (int i = 0; i < 16; ++i) {
        eyes(i % 4, i / 4) = mapDrawer::eye[i];
    }

    trans = eyes * trans;

    for (int i = 0; i < 16; ++i) {
        mapDrawer::eye[i] = trans(i % 4, i / 4);
    }

    mapDrawer::eye[3] = -mapDrawer::eye[3];*/
    
}