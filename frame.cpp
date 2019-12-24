#include "frame.h"
#include "system.h"

//#define PRINT_CELLS_INFO

clvo::frame::frame(cv::Mat& _img, cv::Mat& _depth, system* _system) :img(_img), depth(_depth), mpSystem(_system){
    T = SE3d();
    int row = img.rows;
    int col = img.cols;

    int height = row / cell_size + 1;
    int weight = col / cell_size + 1;

    cells.resize(height);
    for (int i = 0; i < height; ++i) {
        cells[i].resize(weight);
    }
}

void clvo::frame::getLines(int shortest, int longest) {

    if (mpSystem->mSensor == system::RGBD) {
        getLinesRGBD(shortest, longest);
    }
    else if (mpSystem->mSensor == system::Stereo) {
        getLinesSTEREO(shortest, longest);
    }
}

void clvo::frame::cleanNextPtr(const point2d & pt){
    int row = pt[1] / cell_size;
    int col = pt[0] / cell_size;

    for (auto vpt : cells[row][col]) {
        if (vpt->pt == pt) {
            vpt->next = nullptr;
        }
    }
}

int clvo::frame::addToGoodKeypoints(){

    int count = 0;

    for (int i = 1; i < cells.size() - 1; ++i) {
        for (int j = 1; j < cells[i].size() - 1; ++j) {
            for (auto kp : cells[i][j]) {
                if (kp->next) {
                    goodKeypoints.push_back(kp);
                    ++count;
                }
            }
        }
    }

    return count;
}

void clvo::frame::getLinesSTEREO(int shortest, int longest) {

    int row = img.rows;
    int col = img.cols;

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    vector<cv::KeyPoint> keypoints1;
    cv::FAST(img, keypoints1, 40);
    //orb->detect(img, keypoints1);

    vector<cv::KeyPoint> keypoints2;
    cv::FAST(depth, keypoints2, 40);
    //orb->detect(depth, keypoints2);

    vector<vector<point2d> > left, right;
    left.resize(row);
    right.resize(row);

    //提取左右两帧中的角点，用来计算左帧中的特征点深度
    for (auto kp : keypoints1) {
        //cout << kp.pt << endl;
        left[kp.pt.y].push_back(point2d(kp.pt.x, kp.pt.y));
    }
    //cout << endl;
    for (auto kp : keypoints2) {
        right[kp.pt.y].push_back(point2d(kp.pt.x, kp.pt.y));
    }

    vector<point3d> keypoints_depth;

    for (int i = 0; i < row; ++i) {
        //对左边帧的每一个点进行检测
        for (auto pt1 : left[i]) {
            for (auto pt2 : right[i]) {
                if (abs(pt2[0] - pt1[0]) < 100 && abs(pt2[0] - pt1[0]) > 20) {
                    if (NCC(pt1, pt2, 5) > 0.99) {                     
                        keypoints_depth.push_back(point3d(pt1[0], pt1[1], abs(pt1[0] - pt2[0])));
                        
                    }
                }
            }
        }
    }

    //这些内容可以显示有多少点可以得到点的深度
    /*vector<cv::KeyPoint> cvkp;
    for (auto kp : keypoints_depth) {
        cvkp.push_back(cv::KeyPoint(kp[0], kp[1], kp[2] / 5));
    }

    cv::Mat result;
    cv::drawKeypoints(img, cvkp, result);
    cv::imshow("corner", result);
    cv::waitKey();*/

    //将可以成功计算出BRIEF描述子的角点加入到cells
    depth = cv::Mat(row, col, CV_16UC1, cv::Scalar(0));
    for (auto pt3d : keypoints_depth) {
        std::shared_ptr<keypoint> pkp(new keypoint(pt3d[0], pt3d[1]));
        if (BRIEF(pkp)) {
            cells[pt3d[1] / cell_size][pt3d[0] / cell_size].push_back((pkp));
            depth.at<ushort>(pt3d[1], pt3d[0]) = pt3d[2];
        }
    }
    
    cv::imshow("img", img);
    cv::waitKey(1);
}

void clvo::frame::getLinesRGBD(int shortest, int longest) {

    //middleFilter(img);
    getImgEdge();
    int row = img.rows;
    int col = img.cols;

    std::vector<clvo::keypoint> keypoints;

    clvo::ORBDrawer orb1(img, edge);
    //clvo::ORBDrawer orb1(img, cv::Mat());
    orb1.create(keypoints, 500);
    if (keypoints.size() < 250) {
        keypoints.clear();
        clvo::ORBDrawer orb2(img, cv::Mat());
        orb2.create(keypoints, 700);
    }
   
    for (auto kp : keypoints) {
        int x = kp.pt[0];
        int y = kp.pt[1];

        if (depth.at<ushort>(y, x)) {
            std::shared_ptr<keypoint> pkp1(new keypoint(x, y));
            if (BRIEF(pkp1)) {
                //所有新建的特征点的head都设置为自己，如果成功被匹配，其head将被修改为前一帧的匹配点的head
                pkp1->head = pkp1.get();
                pkp1->index = this->index;
                pkp1->next = nullptr;
                cells[y / cell_size][x / cell_size].push_back((pkp1));
                //++kp_size;
            }
        }
    }
    //cout << kp_size << endl;

    for (int i = 50; i < row - 50; i++) {
        for (int j = 50; j < col - 50; j++) {
            if (edge.data[i * col + j] && depth.at<ushort>(i, j)) {
                line2d aim_line;
                getLine(aim_line, point2d(j, i), longest); 
                
                //单目相机需要至少四对角点才可以计算homography，选择至少六个角点可以防止待匹配线条中未检测出当前线条中的角点
                if (mpSystem->mSensor == system::Mono && aim_line.kps.size() < 6)
                    continue;
                if (shortest < aim_line.size()) {
                    //只选择良好的线条中的角点计算描述子
                    //BRIEF(aim_line);
                    lines.push_back(aim_line);
                }
            }
        }
    }
    

#ifdef PRINT_CELLS_INFO//打印网格的信息，论文里面用了一下，可以直接忽略

    for (int i = 0; i < cells.size(); ++i) {
        for (int j = 0; j < cells[i].size(); ++j) {
            printf("%2d  ", cells[i][j].size());
        }
        cout << endl;
    }
    cout << "-----------------------------------" << endl;

#endif // 

    cv::imshow("img", img);
    cv::waitKey(1);
    //cv::imshow("edge", showLines());
    //cv::waitKey(0);
}

void clvo::frame::getImgEdge(double threshold) {
    int row = img.rows;
    int col = img.cols;

    cv::Mat des(row, col, CV_16UC1);
    int max_value_of_pixel = 0;

    for (int i = 1; i < row - 1; ++i) {
        for (int j = 1; j < col - 1; ++j) {
            int gx = img.data[i * col + j - 1] - img.data[i * col + j + 1];
            int gy = img.data[i * col + j - col] - img.data[i * col + j + col];
            des.at<ushort>(i, j) = sqrtForSimpleEdge(gy * gy + gx * gx);
            if (des.at<ushort>(i, j) > max_value_of_pixel)max_value_of_pixel = des.at<ushort>(i, j);
        }
    }
    int* account = new int[max_value_of_pixel + 1]();
    for (int i = 1; i < row - 1; ++i) {
        for (int j = 1; j < col - 1; ++j) {
            account[des.at<ushort>(i, j)]++;
        }
    }
    int all = (row - 2) * (col - 2);
    all -= account[0];
    int count = 0, cur = max_value_of_pixel;
    while (count < all * (1 - threshold)) {
        count += account[cur--];
    }

    edge = cv::Mat(row, col, CV_8UC1);

    for (int i = 1; i < row - 1; ++i) {
        for (int j = 1; j < col - 1; ++j) {
            edge.data[i * col + j] = (uchar)(des.at<ushort>(i, j) >= cur ? (int)des.at<ushort>(i, j) * 255 / max_value_of_pixel : 0);
        }
    }

}

ushort clvo::frame::sqrtForSimpleEdge(__in__ const float number) {
    long i;
    float x, y;
    const float f = 1.5F;
    x = number * 0.5F;
    y = number;
    i = *(long *)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float *)&i;
    return number * y;
}

void clvo::frame::getLine(line2d& aim_line, point2d _point, const int longest, const int edge_size) {

    if (aim_line.size() > longest)return;

    if (!(edge.at<uchar>(_point[1], _point[0]) && depth.at<ushort>(_point[1], _point[0])))
		return;
	if (_point[1] > img.rows - edge_size || _point[0] < edge_size || _point[1] < edge_size || _point[0] > img.cols - edge_size)
		return;

	aim_line.push_back(_point);
    //将深度不为0的角点加入到线条的角点中
    //if (depth.at<ushort>(_point[1], _point[0]) && is_corner(_point)) {
    //    keypoint kp(_point);
    //    aim_line.addKeypoint(kp);
    //    //getAngle(kp);
    //    int x = kp.pt[0], y = kp.pt[1];
    //
    //    bool is_max = false;
    //    for (int i = 0; i < cells[y / cell_size][x / cell_size].size(); ++i) {
    //        if (abs(cells[y / cell_size][x / cell_size][i].pt[0] - x) < 3 && abs(cells[y / cell_size][x / cell_size][i].pt[1] - y) < 3) {
    //            if (getHarrisScore(cells[y / cell_size][x / cell_size][i]) < getHarrisScore(kp)) {
    //                cells[y / cell_size][x / cell_size][i] = kp;
    //                is_max = true;
    //                break;
    //            }
    //        }
    //    }
    //
    //    /*if(!is_max)
    //        cells[y / cell_size][x / cell_size].push_back((kp));*/
    //    //addKeppoint函数的返回值为刚加入的特征点的地址
    //
    //}

    edge.at<uchar>(_point[1], _point[0]) = 0;

	getLine(aim_line, _point + point2d(1, 0));
	getLine(aim_line, _point + point2d(0, 1));
	getLine(aim_line, _point + point2d(0, -1));
	getLine(aim_line, _point + point2d(-1, 0));
	getLine(aim_line, _point + point2d(1, 1));
	getLine(aim_line, _point + point2d(1, -1));
	getLine(aim_line, _point + point2d(-1, -1));
	getLine(aim_line, _point + point2d(-1, 1));	

}

bool clvo::frame::is_corner(const point2d& pt) {
    int row = img.rows;
    int col = img.cols;

    //在图像边缘的点将自动判为非角点
    if (pt[0] < 3 || pt[1] < 3 || pt[0] > col - 4 || pt[1] > row - 4) {
        return false;
    }
    //采样的16个点的位置
    int round_x[] = { -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3 };
    int round_y[] = { 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 };
    
    int _tmp = img.at<uchar>(pt[1], pt[0]);
    int big = (_tmp - 128) * 12 / 128 + 25;
    uchar up_threshold = (_tmp + big) > 255 ? 255 : (_tmp + big);
    uchar down_threshold = (_tmp - big) < 0 ? 0 : (_tmp - big);
    //16表示选择周围16个采样点，9表示连续9个点超过阈值则为角点。其中result[i] == result[i + 16]，
    //通过这种方式可以避免因首尾相接连续9个元素为角点造成检测麻烦
    char result[16 + 9] = {0};

    //超过up_threshold的记为2， 低于down_threshold的记为1，否则记为0
    for (int i = 0; i < 16; ++i) {
        uchar tmp = img.at<uchar>(pt[1] + round_y[i], pt[0] + round_x[i]);
        if (tmp > up_threshold) {
            result[i] = 2;
            if (i < 10) {
                result[i + 16] = 2;
            }
        }
        else if (tmp < down_threshold) {
            result[i] = 1;
            if (i < 10) {
                result[i + 16] = 1;
            }
        }
    }
    //判断是否为角点
    //连续9个小于阈值
    int chack = 0;
    int max_chack = 0;
    for (int i = 0; i < 16 + 9; ++i) {
        if (1 == result[i]) {
            ++chack;
        }
        else {
            if (max_chack < chack)
                max_chack = chack;
            chack = 0;
        }
    }
    if (max_chack > 9)return true;
    //连续9个大于阈值
    chack = 0;
    max_chack = 0;
    for (int i = 0; i < 16 + 9; ++i) {
        if (2 == result[i]) {
            ++chack;
        }
        else {
            if (max_chack < chack)
                max_chack = chack;
            chack = 0;
        }
    }
    if (max_chack > 9)return true;
    return false;
}

bool clvo::frame::BRIEF(std::shared_ptr<keypoint>& kp) {

    int row = img.rows;
    int col = img.cols;

    if (kp->pt[0] < 20 || kp->pt[1] < 20 || kp->pt[0] > col - 20 || kp->pt[1] > row - 20) {
        return false;
    }

    int x = kp->pt[0], y = kp->pt[1];

    getAngle(kp);

    double cos_sita = std::cos(kp->angle);
    double sin_sita = std::sin(kp->angle);

    for (int i = 0; i < 256; i++) {
        int u1 = 0, v1 = 0, u2 = 0, v2 = 0;
        u1 = (int)(cos_sita * ORB_pattern[4 * i] - sin_sita * ORB_pattern[4 * i + 1] + 0.5);
        v1 = (int)(sin_sita * ORB_pattern[4 * i] + cos_sita * ORB_pattern[4 * i + 1] + 0.5);
        u2 = (int)(cos_sita * ORB_pattern[4 * i + 2] - sin_sita * ORB_pattern[4 * i + 3] + 0.5);
        v2 = (int)(sin_sita * ORB_pattern[4 * i + 2] + cos_sita * ORB_pattern[4 * i + 3] + 0.5);
        //计算描述子
        kp->descripter[i / 32] += (img.at<uchar>(v1 + y, u1 + x) > img.at<uchar>(v2 + y, u2 + x) ? 0 : 1);
        if ((i + 1) % 32)
            kp->descripter[i / 32] <<= 1;
    }

    return true;
}

void clvo::frame::BRIEF(line2d& aim_line) {

    int row = img.rows;
    int col = img.cols;

    for (auto kp = aim_line.kps.begin(); kp != aim_line.kps.end(); ) {
        if (kp->pt[0] < 15 || kp->pt[1] < 15 || kp->pt[0] > col - 15 || kp->pt[1] > row - 15) {
            kp = aim_line.kps.erase(kp);
            continue;
        }

        int x = kp->pt[0], y = kp->pt[1];

        getAngle(std::shared_ptr<keypoint>(&(*kp)));
        /*cells[y / cell_size][x / cell_size].push_back((*kp));*/
        
        double cos_sita = std::cos(kp->angle);
        double sin_sita = std::sin(kp->angle);

        for (int i = 0; i < 256; i++) {
            int u1 = 0, v1 = 0, u2 = 0, v2 = 0;
            u1 = (int)(cos_sita * ORB_pattern[4 * i] - sin_sita * ORB_pattern[4 * i + 1] + 0.5);
            v1 = (int)(sin_sita * ORB_pattern[4 * i] + cos_sita * ORB_pattern[4 * i + 1] + 0.5);
            u2 = (int)(cos_sita * ORB_pattern[4 * i + 2] - sin_sita * ORB_pattern[4 * i + 3] + 0.5);
            v2 = (int)(sin_sita * ORB_pattern[4 * i + 2] + cos_sita * ORB_pattern[4 * i + 3] + 0.5);
            //计算描述子
            kp->descripter[i / 32] += (img.at<uchar>(v1 + y, u1 + x) > img.at<uchar>(v2 + y, u2 + x) ? 0 : 1);
            if ((i + 1) % 32)
                kp->descripter[i / 32] <<= 1;
        }
        ++kp;
    }
}

cv::Mat clvo::frame::showLines() {
    
    int row = img.rows;
    int col = img.cols;

    cv::Mat result(row, col, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result.at<cv::Vec3b>(i, j) = (!((i % cell_size) && (j % cell_size))) ? cv::Vec3b(127, 127, 127) : cv::Vec3b(0, 0, 0);
        }
    }

    for (auto line : lines) {
        for (auto pt : line.points) {
            result.at<cv::Vec3b>(pt[1], pt[0]) = cv::Vec3b(0, 0, 255);
        }
        for (auto kp : line.kps) {
            result.at<cv::Vec3b>(kp.pt[1], kp.pt[0]) = cv::Vec3b(0, 255, 0);
        }
    }

    for (int i = 0; i < cells.size(); ++i) {
        for (int j = 0; j < cells[i].size(); ++j) {
            for (auto kp : cells[i][j]) {
                result.at<cv::Vec3b>(kp->pt[1], kp->pt[0]) = cv::Vec3b(0, 255, 0);
            }
        }
    }

    return result;
}

void clvo::frame::middleFilter(cv::Mat& src) {
    int row = src.rows;
    int col = src.cols;

    cv::Mat result(row, col, CV_8UC1, cv::Scalar(0));

    for (int i = 1; i < row - 2; ++i) {
        for (int j = 1; j < col - 2; ++j) {
            uchar data[9];
            uchar* pdata = &img.data[i * col + j];
            data[0] = *(pdata - 1 - col);
            data[1] = *(pdata - col);
            data[2] = *(pdata + 1 - col);
            data[3] = *(pdata - 1);
            data[4] = *(pdata);
            data[5] = *(pdata + 1);
            data[6] = *(pdata - 1 + col);
            data[7] = *(pdata + col);
            data[8] = *(pdata + 1 + col);

            for (int k = 0; k < 5; ++k) {
                for (int l = k; l < 9; ++l) {
                    if (data[l] > data[k]) {
                        auto tmp = data[l];
                        data[l] = data[k];
                        data[k] = tmp;
                    }
                }
            }
            result.at<uchar>(i, j) = data[4];
        }
    }

    for (int i = 1; i < row - 2; ++i) {
        for (int j = 1; j < col - 2; ++j) {
            src.data[i * col + j] = result.data[i * col + j];
        }
    }

}

double clvo::frame::getHarrisScore(keypoint& kp) {

    if (kp.harrisScore != 0)return kp.harrisScore;

    int x = kp.pt[0], y = kp.pt[1];

    int A, B, C;
    A = std::pow((int)img.at<uchar>(y, x + 1) - img.at<uchar>(y, x - 1), 2) +
        std::pow((int)img.at<uchar>(y, x + 2) - img.at<uchar>(y, x), 2);
    B = std::pow((int)img.at<uchar>(y + 1, x) - img.at<uchar>(y - 1, x), 2) +
        std::pow((int)img.at<uchar>(y + 2, x) - img.at<uchar>(y, x), 2);
    C = ((int)img.at<uchar>(y, x + 1) - img.at<uchar>(y, x - 1)) *
        ((int)img.at<uchar>(y + 1, x) - img.at<uchar>(y - 1, x)) +
        ((int)img.at<uchar>(y, x + 2) - img.at<uchar>(y, x)) *
        ((int)img.at<uchar>(y + 2, x) - img.at<uchar>(y, x));

    kp.harrisScore = A * B - C * C - 0.04 * (A + B) * (A + B);
    return kp.harrisScore;
}

double clvo::frame::getAngle(std::shared_ptr<keypoint>& kp) {

    double m_01, m_10;
    int half_patch_size = 8;
    int x = kp->pt[0];
    int y = kp->pt[1];

    for (int i = -half_patch_size; i < half_patch_size; ++i)
        for (int j = -half_patch_size; j < half_patch_size; ++j)
        {
            m_10 += j * 1.0 * img.at<uchar>(y + i, x + j);
            m_01 += i * 1.0 * img.at<uchar>(y + i, x + j);
        }
    kp->angle = atan2(m_01, m_10);
    return kp->angle;
}

double clvo::frame::NCC(point2d& pt1, point2d& pt2, const int half_window_size) {

    if (pt1[0] <= half_window_size || pt1[1] <= half_window_size || pt1[0] >= img.cols - half_window_size || pt1[1] >= img.cols - half_window_size)return 0;
    if (pt2[0] <= half_window_size || pt2[1] <= half_window_size || pt2[0] >= depth.cols - half_window_size || pt2[1] >= depth.cols - half_window_size)return 0;

    vector<double> window1/*((half_window_size * 2 + 1) * (half_window_size * 2 + 1))*/;
    vector<double> window2/*(window1.size())*/;

    double ave1 = 0, ave2 = 0;

    for (int i = -half_window_size; i <= half_window_size; ++i) {
        for (int j = -half_window_size; j < half_window_size; ++j) {
            uchar tmp = img.at<uchar>(pt1[1] + i, pt1[0] + j);
            window1.push_back(tmp);
            ave1 += tmp;
            tmp = depth.at<uchar>(pt1[1] + i, pt1[0] + j);
            window2.push_back(tmp);
            ave2 += tmp;
        }
    }
    ave1 /= window1.size();
    ave2 /= window2.size();
    double up = 0.0, down = 0.0;
    for (int i = 0; i < window1.size(); ++i) {
        window1[i] -= ave1;
        window2[i] -= ave2;
        up += window1[i] * window2[i];
        down = up * up;
    }
    return up / std::sqrt(down);
}


void clvo::ORBDrawer::create(std::vector<clvo::keypoint>& kps, int kp_number) {

    if (kp_number < 1)throw param_exception("bad kp_number in ORBDrawer" + std::to_string(__LINE__));
    N = kp_number;

    ORBKeypoint(kps);
}

void clvo::ORBDrawer::fastCorner(std::vector<clvo::keypoint>& kps, int boundry) {

    bool have_edge_img = false;

    if (edge.rows) {
        have_edge_img = true;
    }

    const int row = img.rows;
    const int col = img.cols;

    for (int i = boundry; i < row - boundry; ++i) {
        for (int j = boundry; j < col - boundry; ++j) {

            if (have_edge_img && 0 == edge.at<uchar>(i, j)) {
                continue;
            }

            bool is_corner = false;
            //采样的16个点的位置
            int round_x[] = { -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3 };
            int round_y[] = { 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 };

            int _tmp = img.at<uchar>(i, j);
            int big = (_tmp - 128) * 12 / 128 + 25;
            uchar up_threshold = (_tmp + big) > 255 ? 255 : (_tmp + big);
            uchar down_threshold = (_tmp - big) < 0 ? 0 : (_tmp - big);
            //16表示选择周围16个采样点，9表示连续9个点超过阈值则为角点。其中result[i] == result[i + 16]，
            //通过这种方式可以避免因首尾相接连续9个元素为角点造成检测麻烦
            char result[16 + 9] = { 0 };

            //超过up_threshold的记为2， 低于down_threshold的记为1，否则记为0
            for (int k = 0; k < 16; ++k) {
                uchar tmp = img.at<uchar>(i + round_y[k], j + round_x[k]);
                if (tmp > up_threshold) {
                    result[k] = 2;
                    if (k < 10) {
                        result[k + 16] = 2;
                    }
                }
                else if (tmp < down_threshold) {
                    result[k] = 1;
                    if (k < 10) {
                        result[k + 16] = 1;
                    }
                }
            }
            //判断是否为角点
            //连续9个小于阈值
            int chack = 0;
            int max_chack = 0;
            for (int i = 0; i < 16 + 9; ++i) {
                if (1 == result[i]) {
                    ++chack;
                }
                else {
                    if (max_chack < chack)
                        max_chack = chack;
                    chack = 0;
                }
            }
            if (max_chack > 9) {
                //kps.push_back(clvo::keypoint(j, i));
                is_corner = true;
            }
            //连续9个大于阈值
            chack = 0;
            max_chack = 0;
            for (int i = 0; i < 16 + 9; ++i) {
                if (2 == result[i]) {
                    ++chack;
                }
                else {
                    if (max_chack < chack)
                        max_chack = chack;
                    chack = 0;
                }
            }

            if (max_chack > 9) {
                //kps.push_back(clvo::keypoint(j, i));
                is_corner = true;
            }

            if (is_corner) {

                clvo::keypoint kp(j, i);
                //计算Harris响应值
                int A, B, C;
                A = std::pow((int)img.at<uchar>(i, j + 1) - img.at<uchar>(i, j - 1), 2) +
                    std::pow((int)img.at<uchar>(i, j + 2) - img.at<uchar>(i, j), 2);
                B = std::pow((int)img.at<uchar>(i + 1, j) - img.at<uchar>(i - 1, j), 2) +
                    std::pow((int)img.at<uchar>(i + 2, j) - img.at<uchar>(i, j), 2);
                C = ((int)img.at<uchar>(i, j + 1) - img.at<uchar>(i, j - 1)) *
                    ((int)img.at<uchar>(i + 1, j) - img.at<uchar>(i - 1, j)) +
                    ((int)img.at<uchar>(i, j + 2) - img.at<uchar>(i, j)) *
                    ((int)img.at<uchar>(i + 2, j) - img.at<uchar>(i, j));

                kp.harrisScore = A * B - C * C - 0.04 * (A + B) * (A + B);

                if (kp.harrisScore > 2)
                    kps.push_back(std::move(kp));
            }
        }
    }

}

void clvo::ORBDrawer::getNKeypoint(vector<clvo::keypoint>& kps) {
    int size = kps.size();

    if (N > size)return;

    int first = 0;
    int j = 1;

    while (1) {
        for (int i = first + 1; i < size; ++i) {
            if (kps[i].harrisScore > kps[first].harrisScore) {
                if (i == j)++j;
                else {
                    std::swap(kps[i].harrisScore, kps[j].harrisScore);
                    ++j;
                }
            }
        }
        --j;
        if (j == first) { std::swap(kps[j].harrisScore, kps[j + 1].harrisScore); }
        std::swap(kps[j].harrisScore, kps[first].harrisScore);
        if (j == N - 1)break;
        if (j < N - 1) {
            first = j + 1;
            j = first + 1;
        }
        else {
            j = first + 1;
        }
    }

    kps.resize(N);
}

void clvo::ORBDrawer::ORBKeypoint(std::vector<clvo::keypoint>& kps) {

    kps.clear();

    bool have_edge_img = false;

    if (edge.rows) {
        have_edge_img = true;
    }

    const int row = img.rows;
    const int col = img.cols;

    if (!(row || col))return;

    fastCorner(kps, 19);
    getNKeypoint(kps);

    for (auto kp : kps) {
        //计算角度
        double m_01, m_10;
        int half_patch_size = 8;
        int x = kp.pt[0];
        int y = kp.pt[1];

        for (int i = -half_patch_size; i < half_patch_size; ++i)
            for (int j = -half_patch_size; j < half_patch_size; ++j)
            {
                m_10 += j * 1.0 * img.at<uchar>(y + i, x + j);
                m_01 += i * 1.0 * img.at<uchar>(y + i, x + j);
            }
        kp.angle = atan2(m_01, m_10);

        //计算描述子
        double cos_sita = std::cos(kp.angle);
        double sin_sita = std::sin(kp.angle);

        for (int i = 0; i < 256; i++) {
            int u1 = 0, v1 = 0, u2 = 0, v2 = 0;
            u1 = (int)(cos_sita * ORB_pattern[4 * i] - sin_sita * ORB_pattern[4 * i + 1] + 0.5);
            v1 = (int)(sin_sita * ORB_pattern[4 * i] + cos_sita * ORB_pattern[4 * i + 1] + 0.5);
            u2 = (int)(cos_sita * ORB_pattern[4 * i + 2] - sin_sita * ORB_pattern[4 * i + 3] + 0.5);
            v2 = (int)(sin_sita * ORB_pattern[4 * i + 2] + cos_sita * ORB_pattern[4 * i + 3] + 0.5);
            //计算描述子
            kp.descripter[i / 32] += (img.at<uchar>(v1 + y, u1 + x) > img.at<uchar>(v2 + y, u2 + x) ? 0 : 1);
            if ((i + 1) % 32)
                kp.descripter[i / 32] <<= 1;
        }
    }
}

void showCorner(const cv::Mat& img, const std::vector<clvo::keypoint>& kps) {

    int row = img.rows;
    int col = img.cols;

    cv::Mat result(row, col, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            uchar tmp = img.at<uchar>(i, j);
            result.at<cv::Vec3b>(i, j) = cv::Vec3b(tmp, tmp, tmp);
        }
    }

    for (auto kp : kps) {
        result.at<cv::Vec3b>(kp.pt[1], kp.pt[0]) = cv::Vec3b(0, 255, 0);
    }

    cv::imshow("corner", result);
    cv::waitKey();
}