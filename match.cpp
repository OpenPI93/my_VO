#include "match.h"
#include "system.h"

void clvo::LucasKanade::getLines(int shortest, int longest) {
    getImgEdge();
    int row = src.rows;
    int col = des.cols;

    int corner_count = 0;

    for (int i = 50; i < row - 50; i++) {
        for (int j = 50; j < col - 50; j++) {
            if (edge.data[i * col + j]) {
                line2d aim_line;
                getLine(aim_line, point2d(j, i), longest);
                //单目相机需要至少四对角点才可以计算homography，选择至少六个角点可以防止待匹配线条中未检测出当前线条中的角点
                if (shortest < aim_line.size()) {
                    //只选择良好的线条中的角点计算描述子
                    lines.push_back(aim_line);
                }
            }
        }
    }

    cv::imshow("edge", showLines());
    cv::waitKey(0);
}


void clvo::LucasKanade::getImgEdge(double threshold) {
    int row = src.rows;
    int col = src.cols;

    cv::Mat tmp_img(row, col, CV_16UC1);
    int max_value_of_pixel = 0;

    for (int i = 1; i < row - 1; ++i) {
        for (int j = 1; j < col - 1; ++j) {
            int gx = src.data[i * col + j - 1] - src.data[i * col + j + 1];
            int gy = src.data[i * col + j - col] - src.data[i * col + j + col];
            tmp_img.at<ushort>(i, j) = sqrtForSimpleEdge(gy * gy + gx * gx);
            if (tmp_img.at<ushort>(i, j) > max_value_of_pixel)max_value_of_pixel = tmp_img.at<ushort>(i, j);
        }
    }
    int* account = new int[max_value_of_pixel + 1]();
    for (int i = 1; i < row - 1; ++i) {
        for (int j = 1; j < col - 1; ++j) {
            account[tmp_img.at<ushort>(i, j)]++;
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
            edge.data[i * col + j] = (uchar)(tmp_img.at<ushort>(i, j) >= cur ? (int)tmp_img.at<ushort>(i, j) * 255 / max_value_of_pixel : 0);
        }
    }

}

ushort clvo::LucasKanade::sqrtForSimpleEdge(__in__ const float number) {
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

void clvo::LucasKanade::getLine(line2d& aim_line, point2d _point, const int longest, const int edge_size) {

    if (aim_line.size() > longest)return;

    if (!edge.at<uchar>(_point[1], _point[0]))
        return;
    if (_point[1] > src.rows - edge_size || _point[0] < edge_size || _point[1] < edge_size || _point[0] > src.cols - edge_size)
        return;

    aim_line.push_back(_point);
   
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

cv::Mat clvo::LucasKanade::showLines() {

    int row = src.rows;
    int col = src.cols;

    cv::Mat result(row, col, CV_8UC3, cv::Scalar(0, 0, 0));

    for (auto line : lines) {
        for (auto pt : line.points) {
            result.at<cv::Vec3b>(pt[1], pt[0]) = cv::Vec3b(0, 0, 255);
        }
        
    }

    return result;
}

vector<clvo::line2d> clvo::LucasKanade::alignTrans() {
    line2d cur_line;
    line2d aim_line;

    vector<cv::Mat> srcs, dess;

    for (int level = 0; level < 4; level++) {
        srcs.push_back(downSample(src, pow(2.0, level)));
        dess.push_back(downSample(des, pow(2.0, level)));
    }

    for (auto pt : lines[1].points) {
        //分散取点，每个点都采用窗口追踪
        if (int(pt[0]) % 3 || int(pt[1]) % 3) {
            continue;
        }
        else {
            aim_line.push_back(pt);
        }
    }
    for (auto pt : aim_line.points) {
        for (int level = 3; level > -1; level--) {
            auto cur_src = srcs[level];
            auto cur_des = dess[level];

            
        }
    }
    return lines;
}

cv::Mat clvo::LucasKanade::downSample(cv::Mat& _src, double size) {

    if (size <= 1.0)return _src;

    int row = (_src.rows / size) - 1;
    int col = (_src.cols / size) - 1;

    cv::Mat result(row, col, CV_8UC1, cv::Scalar(0));
    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result.at<uchar>(i, j) = src.at<uchar>(i * size, j * size);
        }
    }
    return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

int clvo::cornerMatcher::destance(const keypoint& kp1, const keypoint& kp2, int max_distance) {
    int count = 0;
    for (int i = 0; i < 8; ++i) {
        unsigned long c = kp1.descripter[i] ^ kp2.descripter[i];
        
        while (c) {
            count += c & 0x01;
            c >>= 1;
        }
        if (count > max_distance)
            return count;
    }
    return count;
}

Sophus::SE3d clvo::cornerMatcher::computePose(frame* frame1, frame* frame2, frame* frame0) {

    //Sophus::SE3d T = ICPSolver(match(frame1, frame2), false);
    bool isOK;
    Sophus::SE3d T = EPnPSolver(match(frame1, frame2), frame1, isOK);
    //有可能是当前帧的点取得不好，舍弃当前帧再试试，如果依然凉凉，就只能直接法了
    if (!isOK && frame0) {
        Sophus::SE3d T02 = EPnPSolver(match(frame0, frame2), frame0, isOK);
        if (isOK) {
            T = frame0->T.inverse() * T02;
        }
        else {
            T = Sophus::SE3d();
        }
    }
    
    return directPostOptimize(frame1, frame2, T);
    
    //return unitiveDirectPostOptimize(frame1, frame2, Sophus::SE3d());
    
}

vector<clvo::matcher3d> clvo::cornerMatcher::match(frame* frame1, frame* frame2) {

    //cout << frame2->lines.size() << "  ";

    if (mpSystem->mSensor == system::RGBD) {
        return matchRGBD(frame1, frame2);
    }
    else if (mpSystem->mSensor == system::Stereo) {
        return matchSTEREO(frame1, frame2);
    }
}

vector<clvo::matcher3d> clvo::cornerMatcher::matchRGBD(frame* frame1, frame* frame2) {

#ifdef THREADING_POOL
    if(!done)
        return threadingPoolMatchRGBD(frame1, frame2);

#endif
    //清除上一次的匹配结果
    mvMatcher2d.clear();
    
    vector<keypoint*> match1_2;

    for (int i = 1; i < frame1->cells.size() - 1; ++i) {
        for (int j = 1; j < frame1->cells[i].size() - 1; ++j) {
            for (auto kp : frame1->cells[i][j]) {
                //取出一个特征点，之后需要与下一帧中相关网格中的每一个点进行匹配

                kp->next = nullptr;
                int min_distance = 30;
                
                keypoint* pBestMatch = nullptr;

                //对周围共9个格子进行匹配
                for (int round = 0; round < 9; ++round) {
                    for (auto kp2 : frame2->cells[i - 1 + round / 3][j - 1 + round % 3]) {
                        int dist = destance(*kp.get(), *kp2.get(), min_distance);
                        
                        if (dist < min_distance) {
                            min_distance = dist;

                            pBestMatch = kp2.get();
                        }
                    }
                }

                if(pBestMatch){
                    kp.get()->next = pBestMatch;
                    match1_2.push_back(kp.get());
                }
                else {
                    kp.get()->next = nullptr;
                }
            }
        }
    }

    int cell_size = frame1->getCellSize();
    int cell_col = frame1->img.cols / cell_size;
    int cell_row = frame1->img.rows / cell_size;

    //反着再匹配一遍

    for(auto matc: match1_2){

        keypoint* kp = matc->next;

        int i, j;
        frame2->getKeypointCell(kp->pt, j, i);
              
        kp->next = nullptr;
        int min_distance = 30;
                        
        keypoint* pBestMatch = nullptr;

        for (int round = 0; round < 9; ++round) {
            if (j - 1 + round % 3 < 0 || i - 1 + round / 3 < 0 || j - 1 + round % 3 > cell_col || i - 1 + round / 3 > cell_row)continue;
            for (auto kp2 : frame1->cells[i - 1 + round / 3][j - 1 + round % 3]) {
                int dist = destance(*kp, *kp2.get(), min_distance);

                if (dist < min_distance) {
                    min_distance = dist;
                    pBestMatch = kp2.get();                            
                }
            }
        }

        if (pBestMatch) {
            if (matc == pBestMatch) {
                mvMatcher2d.push_back(matcher2d(matc->pt, kp->pt));
                kp->head = matc->head;
            }
            
        }
        else {
             kp->next = nullptr;
             matc->next = nullptr;
        }
    
    }

    //mvMatcher2d = RANSAC(0.8, 0.99, 8, 0.01);    
    match2d_to_3d(frame1, frame2);

    /*match2d_to_3d(frame1, frame2);
    mvMatcher3d = RANSACICP(0.5, 0.99, 4, 0.5);
    match3d_to_2d();*/

    

    return mvMatcher3d;
}

#ifdef THREADING_POOL
vector<clvo::matcher3d> clvo::cornerMatcher::threadingPoolMatchRGBD(frame* frame1, frame* frame2) {
    mvMatcher2d.clear();

    mission = 0;
    threadinfo ti;
    ti.frame1 = frame1;
    ti.frame2 = frame2;
    ti.mission = 1;
    int h = frame1->cells.size() - 2;
    int each_h = h / (maxthreadsnumber + 1);

    int start = 1;
    for (int i = 0; i < maxthreadsnumber; ++i) {
        
        ti.mission = 1;
        ti.mission += (i << 8);
        ti.information.position = point2d(start, start + each_h);
        submit(ti);
        ++mission;
        start += each_h;
        
    }
    for (int i = start; i < frame1->cells.size() - 1; ++i) {
        for (int j = 1; j < frame1->cells[i].size() - 1; ++j) {
            if (frame1->cells[i][j].size() < 1)continue;
            for (auto kp : frame1->cells[i][j]) {
                //取出一个特征点，之后需要与下一帧中相关网格中的每一个点进行匹配

                kp->next = nullptr;
                int min_distance = 30;

                keypoint* pBestMatch = nullptr;

                //对周围共9个格子进行匹配
                for (int round = 0; round < 9; ++round) {
                    for (auto kp2 : frame2->cells[i - 1 + round / 3][j - 1 + round % 3]) {
                        int dist = destance(*kp.get(), *kp2.get(), min_distance);

                        if (dist < min_distance) {
                            min_distance = dist;

                            pBestMatch = kp2.get();
                        }
                    }
                }

                if (pBestMatch) {
                    kp.get()->next = pBestMatch;
                    
                    threadinfo ti;
                    ti.frame1 = frame1;
                    ti.frame2 = frame2;
                    ti.mission = 2;
                    ti.information.pkp = kp.get();
                    submit(ti);
                    ++mission;
                }
                else {
                    kp.get()->next = nullptr;
                }
            }
        }
    }
    //等待所有匹配执行完毕
    while (mission != 0);
    for (auto vkps : readymatchs) {
        for (auto kp : vkps) {
            mvMatcher2d.push_back(kp);
        }
    }

    match2d_to_3d(frame1, frame2);
    return mvMatcher3d;
}
void clvo::cornerMatcher::threadingPoolMatch(frame* frame1, frame* frame2, int start, int end, int threadnumber) {

    halfmatches[threadnumber].clear();
    readymatchs[threadnumber].clear();

    for (int i = start; i < end; ++i) {
        for (int j = 1; j < frame1->cells[i].size() - 1; ++j) {
            if (frame1->cells[i][j].size() < 1)continue;
            for (auto kp : frame1->cells[i][j]) {
                //取出一个特征点，之后需要与下一帧中相关网格中的每一个点进行匹配

                kp->next = nullptr;
                int min_distance = 30;

                keypoint* pBestMatch = nullptr;

                //对周围共9个格子进行匹配
                for (int round = 0; round < 9; ++round) {
                    for (auto kp2 : frame2->cells[i - 1 + round / 3][j - 1 + round % 3]) {
                        int dist = destance(*kp.get(), *kp2.get(), min_distance);

                        if (dist < min_distance) {
                            min_distance = dist;

                            pBestMatch = kp2.get();
                        }
                    }
                }

                if (pBestMatch) {
                    kp.get()->next = pBestMatch;
                    halfmatches[threadnumber].push_back(kp.get());
                }
                else {
                    kp.get()->next = nullptr;
                }
            }
        }
    }
    int cell_size = frame1->getCellSize();
    int cell_col = frame1->img.cols / cell_size;
    int cell_row = frame1->img.rows / cell_size;

    for (auto matc : halfmatches[threadnumber]) {

        keypoint* kp = matc->next;
        int i, j;
        frame2->getKeypointCell(kp->pt, j, i);

        kp->next = nullptr;
        int min_distance = 30;

        keypoint* pBestMatch = nullptr;

        for (int round = 0; round < 9; ++round) {
            if (j - 1 + round % 3 < 0 || i - 1 + round / 3 < 0 || j - 1 + round % 3 > cell_col || i - 1 + round / 3 > cell_row)continue;
            for (auto kp2 : frame1->cells[i - 1 + round / 3][j - 1 + round % 3]) {
                int dist = destance(*kp, *kp2.get(), min_distance);

                if (dist < min_distance) {
                    min_distance = dist;
                    pBestMatch = kp2.get();
                }
            }
        }

        if (pBestMatch) {
            if (matc == pBestMatch) {
                {
                    /*std::lock_guard<std::mutex> lg(mut);
                    mvMatcher2d.push_back(matcher2d(matc->pt, kp->pt));*/
                    kp->head = matc->head;
                    
                    readymatchs[threadnumber].push_back(matcher2d(matc->pt, kp->pt));
                }
            }

        }
        else {
            kp->next = nullptr;
            matc->next = nullptr;
        }
    }

    --mission;
}
void clvo::cornerMatcher::threadingPoolInverseMatch(frame* frame1, frame* frame2, keypoint* matc) {

    keypoint* kp = matc->next;
    int cell_size = frame1->getCellSize();
    int cell_col = frame1->img.cols / cell_size;
    int cell_row = frame1->img.rows / cell_size;

    int i, j;
    frame2->getKeypointCell(kp->pt, j, i);

    kp->next = nullptr;
    int min_distance = 30;

    keypoint* pBestMatch = nullptr;

    for (int round = 0; round < 9; ++round) {
        if (j - 1 + round % 3 < 0 || i - 1 + round / 3 < 0 || j - 1 + round % 3 > cell_col || i - 1 + round / 3 > cell_row)continue;
        for (auto kp2 : frame1->cells[i - 1 + round / 3][j - 1 + round % 3]) {
            int dist = destance(*kp, *kp2.get(), min_distance);

            if (dist < min_distance) {
                min_distance = dist;
                pBestMatch = kp2.get();
            }
        }
    }

    if (pBestMatch) {
        if (matc == pBestMatch) {
            {
                std::lock_guard<std::mutex> lg(mut);
                mvMatcher2d.push_back(matcher2d(matc->pt, kp->pt));
                kp->head = matc->head;
            }
        }

    }
    else {
        kp->next = nullptr;
        matc->next = nullptr;
    }
    --mission;
}

void clvo::cornerMatcher::threadingPoolDirectHcompute(const cv::Mat* cur_img1, const cv::Mat* cur_img2, int start, vector<point3d>* pvpts) {

    {

        Hs[start] = Eigen::Matrix<double, 6, 6>::Zero();
        bs[start] = Eigen::Matrix<double, 6, 1>::Zero();
        inBoundaryCounts[start] = 0;
        costs[start] = 0.0;
    }

    for (int i = start; i < pvpts->size(); i += maxthreadsnumber + 1) {
        point3d current_pt = (*pvpts)[i];
        point3d project_pt = tpT * current_pt;

        double x = project_pt[0], y = project_pt[1], z = project_pt[2], zz = z * z;
        point2d pro2d, src2d;
        project_pt /= project_pt[2];
        pro2d[0] = project_pt[0] * tpFx + tpCx;
        pro2d[1] = project_pt[1] * tpFy + tpCy;

        //越界则跳过，此时inBoundaryCount不进行累加
        if (pro2d[0] < 0 || pro2d[0] > cur_img2->cols || pro2d[1] < 0 || pro2d[1] > cur_img2->rows) {
            continue;
        }
        inBoundaryCounts[start]++;

        current_pt /= current_pt[2];
        src2d[0] = current_pt[0] * tpFx + tpCx;
        src2d[1] = current_pt[1] * tpFy + tpCy;

        double error = 0.0;
        try {
            error = usefulTool::getPixelValue(*cur_img1, src2d[0], src2d[1]) - usefulTool::getPixelValue(*cur_img2, pro2d[0], pro2d[1]);
        }
        catch (...) {
            inBoundaryCounts[start]--;
            continue;
        }

        Eigen::Matrix<double, 2, 6> J_pixel_xi;   // pixel to \xi in Lie algebra

        J_pixel_xi(0, 0) = tpFx / z;
        J_pixel_xi(0, 1) = 0;
        J_pixel_xi(0, 2) = -x * tpFx / zz;
        J_pixel_xi(0, 3) = -x * y * tpFx / zz;
        J_pixel_xi(0, 4) = tpFx + tpFx * x * x / zz;
        J_pixel_xi(0, 5) = -tpFx * y / z;

        J_pixel_xi(1, 0) = 0;
        J_pixel_xi(1, 1) = tpFy / z;
        J_pixel_xi(1, 2) = -tpFy * y / zz;
        J_pixel_xi(1, 3) = -tpFy - tpFy * y * y / zz;
        J_pixel_xi(1, 4) = tpFy * x * y / zz;
        J_pixel_xi(1, 5) = tpFy * x / z;

        Eigen::Vector2d J_img_pixel;    // image gradients

        try {
            J_img_pixel(0, 0) = (usefulTool::getPixelValue(*cur_img2, pro2d[0] + 1, pro2d[1]) - usefulTool::getPixelValue(*cur_img2, pro2d[0] - 1, pro2d[1])) * 0.5;
            J_img_pixel(1, 0) = (usefulTool::getPixelValue(*cur_img2, pro2d[0], pro2d[1] + 1) - usefulTool::getPixelValue(*cur_img2, pro2d[0], pro2d[1] - 1)) * 0.5;
        }
        catch (...) {
            inBoundaryCounts[start]--;
            continue;
        }

        Eigen::Matrix<double, 6, 1> J = -J_pixel_xi.transpose() * J_img_pixel;


        Hs[start] += J * J.transpose();
        bs[start] += -error * J;
        costs[start] += error * error;

    }

    --mission;
}

Sophus::SE3d clvo::cornerMatcher::threadingPoolDriect(frame* frame1, frame* frame2, Sophus::SE3d& T) {

    bool is_Iden = false;
    for (int i = 0; i < 6; ++i) {
        if (T.data()[i])is_Iden = true;
    }
    if (!is_Iden)T = frame1->T;

    //选取的点集，用来进行直接法的计算，
    vector<point3d> select_points;
    
    vector<bool> is_availiable;

    double fx = mpSystem->fx;
    double fy = mpSystem->fy;
    double cx = mpSystem->cx;
    double cy = mpSystem->cy;
    double baseline = mpSystem->baseline;

    const int half_pitch_size = 3;
    const int pitch_size = half_pitch_size * 2 + 1;

    //直接法求解位姿
    //第一步，取点，将所有线条中的点加入到点队列中
    for (auto line : frame1->lines) {
        for (auto pt : line.points) {
            if (is_Iden && int(pt[0]) % 2)continue;
            if (pt[0] < half_pitch_size || pt[1] < half_pitch_size || pt[0] > frame1->img.cols - half_pitch_size || pt[1] > frame1->img.rows - half_pitch_size)continue;
            point3d tmp;
            tmp[0] = (pt[0] - cx) / fx;
            tmp[1] = (pt[1] - cy) / fy;
            tmp[2] = 1.0;
            int ptvalue = frame1->img.at<uchar>(pt[1], pt[0]);
            vector<int> depths;
            for (int i = 0; i < pitch_size * pitch_size; ++i) {
                if (abs(ptvalue - frame1->img.at<uchar>(pt[1] + i / pitch_size - half_pitch_size, pt[0] + i % pitch_size - half_pitch_size) < 3)) {
                    auto cur_depth(frame1->depth.at<ushort>(pt[1] + i / pitch_size - half_pitch_size, pt[0] + i % pitch_size - half_pitch_size));
                    if (cur_depth > 1)
                        depths.push_back(cur_depth);
                }
            }
            //std::sort(depths.begin(), depths.end());
            //int middle_depth = depths[depths.size() / 2];

            int average_depth = 0;
            for (auto dep : depths) {
                average_depth += dep;
            }
            average_depth = average_depth / depths.size();

            tmp *= average_depth / baseline;
            select_points.push_back(tmp);
            is_availiable.push_back(true);
        }
    }

    //第二步，多层直接法迭代，这里分两种情况，当相机成功获得了粗位姿之后，采用单层直接法进行迭代，若没有获得粗位姿，则采用多层直接法
    int MaxLevel;
    if (is_Iden) {
        MaxLevel = 1;
    }
    else {
        MaxLevel = 8;
    }
    //每层降采样的图像缩小倍数
    double downSimpleSize = 1.2;

    tpT = T;

    //用来存储每一层降采样的实际缩小倍数
    vector<double> sizes(MaxLevel, 1.0);
    if (MaxLevel > 1.0) {
        for (int i = MaxLevel - 2; i > -1; --i)
            sizes[i] = sizes[i + 1] * downSimpleSize;
    }

    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;

    threadinfo ti;
    ti.information.pkp = &select_points;

    for (int level = MaxLevel - 1; level > -1; --level) {
        //当前层所用到的图像
        cv::Mat cur_img1, cur_img2;
        if (1.0 == sizes[level]) {
            cur_img1 = frame1->img;
            cur_img2 = frame2->img;
        }
        else {
            cur_img1 = usefulTool::DownSampling(frame1->img, sizes[level]);
            cur_img2 = usefulTool::DownSampling(frame2->img, sizes[level]);
        }
        ti.frame1 = &cur_img1;
        ti.frame2 = &cur_img2;
        //计算当前层的内参
        tpFx = mpSystem->fx / sizes[level];
        tpFy = mpSystem->fy / sizes[level];
        tpCx = mpSystem->cx / sizes[level];
        tpCy = mpSystem->cy / sizes[level];

        //每一层迭代前的数据准备
        double cost = 0.0, lastcost = 0.0;
        int maxIter = 30;

        for (int iter = 0; iter < maxIter; ++iter) {
            int inBoundaryCount = 0;
            cost = 0.0;

            H = Eigen::Matrix<double, 6, 6>::Zero();  // 6x6 Hessian
            b = Eigen::Matrix<double, 6, 1>::Zero();  // 6x1 bias

            //多线程
            for (int i = 0; i < maxthreadsnumber; ++i) {
                ti.mission = 3;
                ti.mission += (i << 8);

                submit(ti);
                ++mission;
            }
            
            for (int i = maxthreadsnumber; i < select_points.size(); i += maxthreadsnumber + 1) {
                point3d current_pt = select_points[i];
                point3d project_pt = tpT * current_pt;

                double x = project_pt[0], y = project_pt[1], z = project_pt[2], zz = z * z;
                point2d pro2d, src2d;
                project_pt /= project_pt[2];
                pro2d[0] = project_pt[0] * tpFx + tpCx;
                pro2d[1] = project_pt[1] * tpFy + tpCy;

                //越界则跳过，此时inBoundaryCount不进行累加
                if (pro2d[0] < 0 || pro2d[0] > cur_img2.cols || pro2d[1] < 0 || pro2d[1] > cur_img2.rows) {
                    continue;
                }
                inBoundaryCount++;

                current_pt /= current_pt[2];
                src2d[0] = current_pt[0] * tpFx + tpCx;
                src2d[1] = current_pt[1] * tpFy + tpCy;

                double error = 0.0;
                try {
                    error = usefulTool::getPixelValue(cur_img1, src2d[0], src2d[1]) - usefulTool::getPixelValue(cur_img2, pro2d[0], pro2d[1]);
                }
                catch (...) {
                    inBoundaryCount--;
                    continue;
                }

                Eigen::Matrix<double, 2, 6> J_pixel_xi;   // pixel to \xi in Lie algebra

                J_pixel_xi(0, 0) = tpFx / z;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -x * tpFx / zz;
                J_pixel_xi(0, 3) = -x * y * tpFx / zz;
                J_pixel_xi(0, 4) = tpFx + tpFx * x * x / zz;
                J_pixel_xi(0, 5) = -tpFx * y / z;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = tpFy / z;
                J_pixel_xi(1, 2) = -tpFy * y / zz;
                J_pixel_xi(1, 3) = -tpFy - tpFy * y * y / zz;
                J_pixel_xi(1, 4) = tpFy * x * y / zz;
                J_pixel_xi(1, 5) = tpFy * x / z;

                Eigen::Vector2d J_img_pixel;    // image gradients

                try {
                    J_img_pixel(0, 0) = (usefulTool::getPixelValue(cur_img2, pro2d[0] + 1, pro2d[1]) - usefulTool::getPixelValue(cur_img2, pro2d[0] - 1, pro2d[1])) * 0.5;
                    J_img_pixel(1, 0) = (usefulTool::getPixelValue(cur_img2, pro2d[0], pro2d[1] + 1) - usefulTool::getPixelValue(cur_img2, pro2d[0], pro2d[1] - 1)) * 0.5;
                }
                catch (...) {
                    inBoundaryCount--;
                    continue;
                }

                Eigen::Matrix<double, 6, 1> J = -J_pixel_xi.transpose() * J_img_pixel;


                H += J * J.transpose();
                b += -error * J;
                cost += error * error;

            }


            while (0 != mission);

            for (auto i = 0; i < maxthreadsnumber; ++i) {
                H += Hs[i];
                b += bs[i];
                cost += costs[i];
                inBoundaryCount += inBoundaryCounts[i];
            }

            Eigen::Matrix<double, 6, 1> update;
            update = H.ldlt().solve(b);
            tpT = Sophus::SE3d::exp(update) * tpT;
            // END YOUR CODE HERE

            cost /= inBoundaryCount;

            if (isnan(update[0])) {
                cout << "update is nan" << endl;
                break;
            }
            if (iter > 0 && cost > lastcost) {
                break;
            }
            lastcost = cost;
        }
    }

    T = tpT;
    return tpT;
}


#endif

vector<clvo::matcher3d> clvo::cornerMatcher::matchSTEREO(frame* frame1, frame* frame2) {

    //清除上一次的匹配结果
    mvMatcher2d.clear();

    for (int i = 1; i < frame1->cells.size() - 1; ++i) {
        for (int j = 1; j < frame1->cells[i].size() - 1; ++j) {
            for (auto kp : frame1->cells[i][j]) {
                //取出一个特征点，之后需要与下一帧中相关网格中的每一个点进行匹配

                int min_distance = 30;
                keypoint* pBestMatch = nullptr;

                //对周围共9个格子进行匹配
                for (int round = 0; round < 9; ++round) {
                    for (auto kp2 : frame2->cells[i - 1 + round / 3][j - 1 + round % 3]) {
                        int dist = destance(*kp.get(), *kp2.get(), min_distance);

                        if (dist < min_distance) {
                            //min_distance = dist;
                            mvMatcher2d.push_back(matcher2d(kp->pt, kp2->pt));
                            //pBestMatch = &(kp2);
                        }
                    }
                }

                if (nullptr != pBestMatch) {
                    // mvMatcher2d.push_back(matcher2d(kp.pt, pBestMatch->pt));
                }
            }
        }
    }

    match2d_to_3d_Stereo(frame1, frame2);

    auto vec = RANSACICP(0.5, 0.9999, 3, 0.1);

    mvMatcher3d.clear();
    for (auto m : vec) {
        mvMatcher3d.push_back(m);
    }
    match3d_to_2d();

    cv::imshow("match", showMatch(frame1, frame2));
    cv::waitKey();

    return mvMatcher3d;
}

void clvo::cornerMatcher::match2d_to_3d(frame* frame1, frame* frame2) {

    if (mvMatcher3d.size())mvMatcher3d.clear();
    for (auto matc : mvMatcher2d) {
        point2d pt1 = matc.pt1;
        point3d pt3d1;
        pt3d1[0] = (pt1[0] - mpSystem->cx) / mpSystem->fx;
        pt3d1[1] = (pt1[1] - mpSystem->cy) / mpSystem->fy;
        pt3d1[2] = 1.0;
        int ptvalue = frame1->img.at<uchar>(pt1[1], pt1[0]);
        int sum_depth = 0;
        int round_select = 0;
        for (int i = 0; i < 25; ++i) {
            if (abs(ptvalue - frame1->img.at<uchar>(pt1[1] + i / 5 - 2, pt1[0] + i % 5 - 2) < 3)) {
                auto cur_dep = frame1->depth.at<ushort>(pt1[1] + i / 5 - 2, pt1[0] + i % 5 - 2);
                if (cur_dep > 1)
                    sum_depth += cur_dep;
                ++round_select;
            }
        }
        sum_depth /= round_select;
        //pt3d1 *= 1.0 * frame1->depth.at<ushort>(pt1[1], pt1[0]) / mpSystem->baseline;
        pt3d1 *= 1.0 * sum_depth / mpSystem->baseline;

        point2d pt2 = matc.pt2;
        point3d pt3d2;
        pt3d2[0] = (pt2[0] - mpSystem->cx) / mpSystem->fx;
        pt3d2[1] = (pt2[1] - mpSystem->cy) / mpSystem->fy;
        pt3d2[2] = 1.0;
        ptvalue = frame2->img.at<uchar>(pt2[1], pt2[0]);
        sum_depth = 0;
        round_select = 0;
        for (int i = 0; i < 25; ++i) {
            if (abs(ptvalue - frame2->img.at<uchar>(pt2[1] + i / 5 - 2, pt2[0] + i % 5 - 2) < 3)) {
                sum_depth += frame2->depth.at<ushort>(pt2[1] + i / 5 - 2, pt2[0] + i % 5 - 2);
                ++round_select;
            }
        }
        sum_depth /= round_select;
        pt3d2 *= 1.0 * frame2->depth.at<ushort>(pt2[1], pt2[0]) / mpSystem->baseline;

        /*if (abs(pt3d1[2] - pt3d2[2]) < 0.08)*/
        mvMatcher3d.push_back(matcher3d(pt3d1, pt3d2));
    }
}

void clvo::cornerMatcher::match2d_to_3d_Stereo(frame* frame1, frame* frame2) {
    if (mvMatcher3d.size())mvMatcher3d.clear();
    for (auto matc : mvMatcher2d) {
        point2d pt1 = matc.pt1;
        point3d pt3d1;
        pt3d1[0] = (pt1[0] - mpSystem->cx) / mpSystem->fx;
        pt3d1[1] = (pt1[1] - mpSystem->cy) / mpSystem->fy;
        pt3d1[2] = 1.0;
        pt3d1 *= mpSystem->baseline / frame1->depth.at<ushort>(pt1[1], pt1[0]);

        point2d pt2 = matc.pt2;
        point3d pt3d2;
        pt3d2[0] = (pt2[0] - mpSystem->cx) / mpSystem->fx;
        pt3d2[1] = (pt2[1] - mpSystem->cy) / mpSystem->fy;
        pt3d2[2] = 1.0;
        pt3d2 *= mpSystem->baseline / frame2->depth.at<ushort>(pt2[1], pt2[0]);

       // printf("%lf  %lf  %lf  %d\n", mpSystem->baseline, pt3d1[2], pt3d2[2], frame1->depth.at<ushort>(pt1[1], pt1[0]));

        mvMatcher3d.push_back(matcher3d(pt3d1, pt3d2));
    }
}

cv::Mat clvo::cornerMatcher::showMatch(frame* frame1, frame* frame2) {

    int row = frame1->img.rows;
    int col = frame2->img.cols;

    cv::Mat result;

    if (col < 700) {

        result = cv::Mat(row, col * 2, CV_8UC3, cv::Scalar(0));

        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                uchar tmp1 = frame1->img.at<uchar>(i, j);
                uchar tmp2 = frame2->img.at<uchar>(i, j);
                result.at<cv::Vec3b>(i, j) = cv::Vec3b(tmp1, tmp1, tmp1);
                result.at<cv::Vec3b>(i, j + col) = cv::Vec3b(tmp2, tmp2, tmp2);
            }
        }

        for (int i = 0; i < mvMatcher2d.size(); i += 1) {

            double x1 = mvMatcher2d[i].pt1[0];
            double x2 = mvMatcher2d[i].pt2[0] + col;
            double y1 = mvMatcher2d[i].pt1[1];
            double y2 = mvMatcher2d[i].pt2[1];
            cv::Point2d pt1(x1, y1);
            cv::Point2d pt2(x2, y2);
            int r = (i & 0x07) << 5;
            int g = (i & 0x38) << 2;
            int b = (i & 0x1c0) >> 1;
            cv::line(result, pt1, pt2, cv::Scalar(b, g, r));
        }
    }
    else {
        result = cv::Mat(row * 2, col, CV_8UC3, cv::Scalar(0));

        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                uchar tmp1 = frame1->img.at<uchar>(i, j);
                uchar tmp2 = frame2->img.at<uchar>(i, j);
                result.at<cv::Vec3b>(i, j) = cv::Vec3b(tmp1, tmp1, tmp1);
                result.at<cv::Vec3b>(i + row, j) = cv::Vec3b(tmp2, tmp2, tmp2);
            }
        }

        for (int i = 0; i < mvMatcher2d.size(); i += 1) {

            double x1 = mvMatcher2d[i].pt1[0];
            double x2 = mvMatcher2d[i].pt2[0];
            double y1 = mvMatcher2d[i].pt1[1];
            double y2 = mvMatcher2d[i].pt2[1] + row;
            cv::Point2d pt1(x1, y1);
            cv::Point2d pt2(x2, y2);
            int r = (i & 0x07) << 5;
            int g = (i & 0x38) << 2;
            int b = (i & 0x1c0) >> 1;
            cv::line(result, pt1, pt2, cv::Scalar(b, g, r));
        }
    }

    return result;

}

void clvo::cornerMatcher::match3d_to_2d() {
    if (mvMatcher2d.size())mvMatcher2d.clear();
    for (auto matc : mvMatcher3d) {
        point3d pt1 = matc.pt1;
        point2d pt2d1;
        pt1 /= pt1[2];
        pt2d1[0] = int(pt1[0] * mpSystem->fx + mpSystem->cx + 0.5);
        pt2d1[1] = int(pt1[1] * mpSystem->fy + mpSystem->cy + 0.5);

        point3d pt2 = matc.pt2;
        point2d pt2d2;
        pt2 /= pt2[2];
        pt2d2[0] = int(pt2[0] * mpSystem->fx + mpSystem->cx + 0.5);
        pt2d2[1] = int(pt2[1] * mpSystem->fy + mpSystem->cy + 0.5);

        mvMatcher2d.push_back(matcher2d(pt2d1, pt2d2));
    }
}

Eigen::Matrix3d clvo::cornerMatcher::estimateEssentialEightPoints(vector<clvo::matcher2d>& data) {

    if (data.size() < 8)throw param_exception("八点法参数应至少为八对点");
    Eigen::Matrix<double, 8, 9> A;

    //八点法
    for (int i = 0; i < 8; ++i) {
        A(i, 0) = data[i].pt1[0] * data[i].pt2[0];
        A(i, 1) = data[i].pt1[0] * data[i].pt2[1];
        A(i, 2) = data[i].pt1[0];
        A(i, 3) = data[i].pt1[1] * data[i].pt2[0];
        A(i, 4) = data[i].pt1[1] * data[i].pt2[1];
        A(i, 5) = data[i].pt1[1];
        A(i, 6) = data[i].pt2[0];
        A(i, 7) = data[i].pt2[1];
        A(i, 8) = 1;
    }

    //SVD分解得到的V矩阵中，最后一行（S值最小）的行向量为零空间
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 9> V = svd.matrixV();

    Eigen::Matrix3d E;
    E(0, 0) = V(8, 0);
    E(0, 1) = V(8, 1);
    E(0, 2) = V(8, 2);
    E(1, 0) = V(8, 3);
    E(1, 1) = V(8, 4);
    E(1, 2) = V(8, 5);
    E(2, 0) = V(8, 6);
    E(2, 1) = V(8, 7); 
    E(2, 2) = V(8, 8);

    return E;
}

vector<clvo::matcher2d> clvo::cornerMatcher::RANSAC(double p, double z, int k, double threshold) {
    //初始化随机数
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(1, mvMatcher2d.size());

    //计算迭代次数
    int N = log(1 - z) / log(1 - pow(p, k));
    int max_inner = 0;
    Eigen::Matrix3d best_E;

    //迭代N次，找到最好的模型
    for (int i = 0; i < N; ++i) {
        vector<matcher2d> indexes;
        for (int j = 0; j < k; ++j)
            indexes.push_back(mvMatcher2d[dist(mt)]);
        
        Eigen::Matrix3d tmp_E = estimateEssentialEightPoints(indexes);

        //计算有多少点与当前模型误差小于阈值
        int cur_inner = 0;
        for (auto matc : mvMatcher2d) {
            //double u1 = matc.pt1[0], u2 = matc.pt2[0], v1 = matc.pt1[1], v2 = matc.pt2[1];
            point3d x1(matc.pt1[0], matc.pt1[1], 1), x2(matc.pt2[0], matc.pt2[1], 1);
            double err = x1.transpose() * tmp_E * x2;
            if (err < threshold) {
                cur_inner++;
            }
        }
        //如果当前模型比最好的模型有更多的内点，则将当前模型置为最佳模型
        if (cur_inner > max_inner) {
            max_inner = cur_inner;
            best_E = tmp_E;
        }
    }

    vector<matcher2d> result;
    //筛选内点
    for (auto matc : mvMatcher2d) {
        point3d x1(matc.pt1[0], matc.pt1[1], 1), x2(matc.pt2[0], matc.pt2[1], 1);
        double err = x1.transpose() * best_E * x2;
        if (err < threshold) {
            result.push_back(matc);
        }
    }

    return result;
}

vector<clvo::matcher3d> clvo::cornerMatcher::RANSACICP(double p, double z, int k, double threshold) {

    if (mvMatcher3d.size() < k)return vector<clvo::matcher3d>();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0, mvMatcher3d.size());

    int N = log(1 - z) / log(1 - pow(p, k));
    int minInner = int(p * mvMatcher3d.size() + 0.5);
    int best_inner = 0;
    Sophus::SE3d best_se3;
    vector<bool> vbbest_inner;
    vbbest_inner.resize(mvMatcher3d.size(), false);

    double fx = mpSystem->fx;
    double cx = mpSystem->cx;
    double fy = mpSystem->fy;
    double cy = mpSystem->cy;

    for (int i = 0; i < N; ++i) {
        vector<matcher3d> indexes;
        for (int j = 0; j < k; ++j)
            indexes.push_back(mvMatcher3d[int(dist(mt))]);
       
        Sophus::SE3d tmp_se3 = ICPSolver(indexes, true);

       // cout << tmp_se3.matrix3x4() << endl;
        int cur_inner = 0;
        vector<bool> vbcur_inner(vbbest_inner.size(), false);

        //检查内点
        for (int i = 0; i < mvMatcher3d.size(); ++i) {
            
            point3d ptc = tmp_se3 * mvMatcher3d[i].pt1;
            ptc /= ptc[2];

            point2d pt_proje = point2d(ptc[0] * fx + cx, ptc[1] * fy + cy);

            point2d pt_err = pt_proje - mvMatcher2d[i].pt2;

            double err = sqrt(pt_err[0] * pt_err[0] + pt_err[1] * pt_err[1]);

            if (err < threshold) {
                vbcur_inner[i] = true;
                cur_inner++;
            }

        }

        if (cur_inner > best_inner) {
            best_se3 = tmp_se3;
            best_inner = cur_inner;
            vbbest_inner = vbcur_inner;
        }
        
    }

    vector<matcher3d> result;

    for (int i = 0; i < mvMatcher3d.size(); ++i) {
        if (vbbest_inner[i]) {
            result.push_back(mvMatcher3d[i]);
        }
    }

    //cout << best_se3.matrix3x4() <<endl;

    return result;
    
}

Sophus::SE3d clvo::cornerMatcher::directPostOptimize(frame* frame1, frame* frame2, Sophus::SE3d& T) {

#ifdef THREADING_POOL
    if (!done)
        return threadingPoolDriect(frame1, frame2, T);

#endif

    //如果传入的初试位姿为单位李代数，有可能有两种情况：第一种为特征点法定位失败；第二种是估算相机没有运动。两种情况均采用前一帧的运动为当前帧的初始值
    bool is_Iden = false;
    for (int i = 0; i < 6; ++i) {
        if (T.data()[i])is_Iden = true;
    }
    if (!is_Iden)T = frame1->T;
    
    //选取的点集，用来进行直接法的计算，
    vector<point3d> select_points;
    /*
    点集中对应的点是否可用
    判断可用性的标准：
        1：深度值准确且邻域深度值也准确
        2：连续两次迭代带来的误差大于阈值
        3：点经过降采样后越界
    */
    vector<bool> is_availiable;
    
    double fx = mpSystem->fx;
    double fy = mpSystem->fy;
    double cx = mpSystem->cx;
    double cy = mpSystem->cy;
    double baseline = mpSystem->baseline;

    const int half_pitch_size = 3;
    const int pitch_size = half_pitch_size * 2 + 1;

    //直接法求解位姿
    //第一步，取点，将所有线条中的点加入到点队列中
    for (auto line : frame1->lines) {
        for (auto pt : line.points) {
            if (is_Iden && int(pt[0]) % 2)continue;
            if(pt[0] < half_pitch_size || pt[1] < half_pitch_size || pt[0] > frame1->img.cols - half_pitch_size || pt[1] > frame1->img.rows - half_pitch_size)continue;
            point3d tmp;
            tmp[0] = (pt[0] - cx) / fx;
            tmp[1] = (pt[1] - cy) / fy;
            tmp[2] = 1.0;
            int ptvalue = frame1->img.at<uchar>(pt[1], pt[0]);
            vector<int> depths;
            for (int i = 0; i < pitch_size * pitch_size; ++i) {
                if (abs(ptvalue - frame1->img.at<uchar>(pt[1] + i / pitch_size - half_pitch_size, pt[0] + i % pitch_size - half_pitch_size) < 3)) {
                    auto cur_depth(frame1->depth.at<ushort>(pt[1] + i / pitch_size - half_pitch_size, pt[0] + i % pitch_size - half_pitch_size));
                    if (cur_depth > 1)
                        depths.push_back(cur_depth);
                }
            }
            //std::sort(depths.begin(), depths.end());
            //int middle_depth = depths[depths.size() / 2];

            int average_depth = 0;
            for (auto dep : depths) {
                average_depth += dep;
            }
            average_depth = average_depth / depths.size();
            
            tmp *= average_depth / baseline;
            select_points.push_back(tmp);
            is_availiable.push_back(true);
        }
    }
   
    //第二步，多层直接法迭代，这里分两种情况，当相机成功获得了粗位姿之后，采用单层直接法进行迭代，若没有获得粗位姿，则采用多层直接法
    int MaxLevel;
    if (is_Iden) {
        MaxLevel = 1;
    }
    else {
        MaxLevel = 8;
    }
    //每层降采样的图像缩小倍数
    double downSimpleSize = 1.2;

    //用来存储每一个观测中的误差，在直接法中，误差值为像素差值
    vector<double> errors(select_points.size());
    //用来记录Ω值
    vector<double> omigas(select_points.size(), 1.0);

    //用来存储每一层降采样的实际缩小倍数
    vector<double> sizes(MaxLevel, 1.0);
    if (MaxLevel > 1.0) {
        for (int i = MaxLevel - 2; i > -1; --i)
            sizes[i] = sizes[i + 1] * downSimpleSize;
    }

    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;

    //开始迭代，首先最外层为金字塔的层数，具体可参照论文第二章多层直接法的流程图
    for (int level = MaxLevel - 1; level > -1; --level) {
        //当前层所用到的图像
        cv::Mat cur_img1, cur_img2;
        if (1.0 == sizes[level]) {
            cur_img1 = frame1->img;
            cur_img2 = frame2->img;
        }
        else {
            cur_img1 = usefulTool::DownSampling(frame1->img, sizes[level]);
            cur_img2 = usefulTool::DownSampling(frame2->img, sizes[level]);
        }
        //计算当前层的内参
        fx = mpSystem->fx / sizes[level];
        fy = mpSystem->fy / sizes[level];
        cx = mpSystem->cx / sizes[level];
        cy = mpSystem->cy / sizes[level];

        //每一层迭代前的数据准备
        double cost = 0.0, lastcost = 0.0;
        int maxIter = 30;
        
        for (int iter = 0; iter < maxIter; ++iter) {
            int inBoundaryCount = 0;
            cost = 0.0;


            H = Eigen::Matrix<double, 6, 6>::Zero();  // 6x6 Hessian
            b = Eigen::Matrix<double, 6, 1>::Zero();  // 6x1 bias

            for (int i = 0; i < select_points.size(); ++i) {
                if (!is_availiable[i])continue;
                point3d current_pt = select_points[i];
                point3d project_pt = T * current_pt;

                double x = project_pt[0], y = project_pt[1], z = project_pt[2], zz = z * z;
                point2d pro2d, src2d;
                project_pt /= project_pt[2];
                pro2d[0] = project_pt[0] * fx + cx;
                pro2d[1] = project_pt[1] * fy + cy;

                //越界则跳过，此时inBoundaryCount不进行累加
                if (pro2d[0] < 0 || pro2d[0] > cur_img2.cols || pro2d[1] < 0 || pro2d[1] > cur_img2.rows) {
                    continue;
                }
                inBoundaryCount++;

                current_pt /= current_pt[2];
                src2d[0] = current_pt[0] * fx + cx;
                src2d[1] = current_pt[1] * fy + cy;

                double error = 0.0;
                try {
                    error = usefulTool::getPixelValue(cur_img1, src2d[0], src2d[1]) - usefulTool::getPixelValue(cur_img2, pro2d[0], pro2d[1]);
                }
                catch (...) {
                    inBoundaryCount--;
                    continue;
                }
                errors[i] = error;
                /*if (iter > 3 && error > lastcost * 3 / select_points.size()) {
                    is_availiable[i] = false;
                }*/

                Eigen::Matrix<double, 2, 6> J_pixel_xi;   // pixel to \xi in Lie algebra

                J_pixel_xi(0, 0) = fx / z;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -x * fx / zz;
                J_pixel_xi(0, 3) = -x * y * fx / zz;
                J_pixel_xi(0, 4) = fx + fx * x * x / zz;
                J_pixel_xi(0, 5) = -fx * y / z;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy / z;
                J_pixel_xi(1, 2) = -fy * y / zz;
                J_pixel_xi(1, 3) = -fy - fy * y * y / zz;
                J_pixel_xi(1, 4) = fy * x * y / zz;
                J_pixel_xi(1, 5) = fy * x / z;

                Eigen::Vector2d J_img_pixel;    // image gradients

                try {
                    J_img_pixel(0, 0) = (usefulTool::getPixelValue(cur_img2, pro2d[0] + 1, pro2d[1]) - usefulTool::getPixelValue(cur_img2, pro2d[0] - 1, pro2d[1])) * 0.5;
                    J_img_pixel(1, 0) = (usefulTool::getPixelValue(cur_img2, pro2d[0], pro2d[1] + 1) - usefulTool::getPixelValue(cur_img2, pro2d[0], pro2d[1] - 1)) * 0.5;
                }
                catch (...) {
                    inBoundaryCount--;
                    continue;
                }
                
                Eigen::Matrix<double, 6, 1> J = -J_pixel_xi.transpose() * J_img_pixel;

                H += J * J.transpose();
                b +=  -error * J;
                cost += error * error;

            }// end of circulate for each point

            Eigen::Matrix<double, 6, 1> update;
            update = H.ldlt().solve(b);
            T = Sophus::SE3d::exp(update) * T;
            // END YOUR CODE HERE

            cost /= inBoundaryCount;

            if (isnan(update[0])) {               
                cout << "update is nan" << endl;
                break;
            }
            if (iter > 0 && cost > lastcost) {               
                break;
            }
            lastcost = cost;
            
        }//end of circulate for each iteration

    }//end of circulate for each level of pyramid

    return T;
}


Sophus::SE3d clvo::cornerMatcher::ICPSolver(vector<matcher3d>& data, bool is_RANSAC) {

    Sophus::SE3d T;

    if (!(is_RANSAC) && data.size() < 15)return Sophus::SE3d();

    double cost = 0;
    for (int i = 0; i < 30; ++i) {  
        cost = 0;

        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        for (auto matc : data) {
            point3d cur_pt1 = T * matc.pt1;
            point3d err = matc.pt2 - cur_pt1;
            cost += err.transpose() * err;

            double x = cur_pt1[0], y = cur_pt1[1], z = cur_pt1[2];

            Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();

            /*
            1  0  0   0    z   -y
            0  1  0  -z    0    x
            0  0  1   y   -x    0
            */

            J(0, 0) = 1;
            J(0, 4) = z;
            J(0, 5) = -y;

            J(1, 1) = 1;
            J(1, 3) = -z;
            J(1, 5) = x;

            J(2, 2) = 1;
            J(2, 3) = y;
            J(2, 4) = -x;

            H += J.transpose() * J;
            b += J.transpose() * err;
        }

        Eigen::Matrix<double, 6, 1> update;
        update = H.ldlt().solve(b);

        T = Sophus::SE3d::exp(update) * T;

        if (cost < 1e-8) {          
            break;
        }
    }
    
    return T;
}

Sophus::SE3d clvo::cornerMatcher::EPnPSolver(vector<matcher3d>& data, frame* pframe1, bool& isOK) {

    vector<point2d> mvP2ds;
    vector<point3d> mvP3ds;

    double fx = mpSystem->fx, cx = mpSystem->cx, fy = mpSystem->fy, cy = mpSystem->cy;

    for (auto match : data) {
        point3d p1 = match.pt1;
        point3d p2 = match.pt2;

        p2 /= p2[2];
        point2d pt2d(p2[0] * fx + cx, p2[1] * fy + cy);
        mvP3ds.push_back(p1);
        mvP2ds.push_back(pt2d);
    }

    PnPsolver epnp(mvP2ds, mvP3ds, vector<double>{fx,cx, fy, cy });

    epnp.SetRansacParameters(0.99, 10, 400, 4, 0.5, 5.991);

    vector<bool> vbInliers;
    int nInliers = 0;
    bool bNoMore;

    //参数分别为： 内点最少数量，无意义，是否为内点，内点个数
    Sophus::SE3d T = epnp.iterate(15, bNoMore, vbInliers, nInliers);
    if (nInliers < 15) {
        isOK = false;
    }
    
    if (!vbInliers.size()) {
        vbInliers = vector<bool>(mvMatcher2d.size(), false);
    }
    for (int i = 0; i < vbInliers.size(); ++i) {
        if (!vbInliers[i]) {
            pframe1->cleanNextPtr(mvMatcher2d[i].pt1);
        }
    }
    //输出相关信息
#ifdef _WIN64
    HANDLE hout;
    COORD coord;
    coord.X = 0;
    coord.Y = 3;
    hout = GetStdHandle(STD_OUTPUT_HANDLE); 
    CONSOLE_CURSOR_INFO cursor_info = { 1, 0 };
    SetConsoleCursorInfo(hout, &cursor_info);
    SetConsoleCursorPosition(hout, coord);

#elif __linux__
printf("\033[%d;%dH", 3, 0);
printf("\033[?25l");
#endif
    //将成功匹配的特征点单独加到一个队列中，若干帧之后将删除cells
    printf("当前帧内点个数：%3d    ", pframe1->addToGoodKeypoints());
    isOK = true;

    return T;
}

Sophus::SE3d clvo::cornerMatcher::pointDirectPostOptimize(frame* frame1, frame* frame2, Sophus::SE3d&T) {

    int row = frame1->img.rows;
    int col = frame1->img.cols;

    int half_window_size = 3;
    int window_size = half_window_size * 2 + 1;

    

    return T;
}