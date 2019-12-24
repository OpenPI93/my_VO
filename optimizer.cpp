#include "optimizer.h"

void clvo::directPoseOnlyBackEnd::run(vector<std::shared_ptr<frame>>& frames) {
    //第一帧的留空
    vector<Eigen::Matrix<double, 6, 6>> Hs(frames.size());
    vector<Eigen::Matrix<double, 6, 1>> bs(frames.size());

    //预备数据，将第一帧的所有线条中的点计算深度并加入到队列中
    double inv_fx = 1.0 / fx;
    double inv_fy = 1.0 / fy;

    vector<bool> is_availiable;

    const int half_pitch_size = 3;
    const int pitch_size = half_pitch_size * 2 + 1;

    vector<point3d> select_points;
    for (auto line : frames[0]->lines) {
        for (auto pt : line.points) {
            
            if (pt[0] < half_pitch_size || pt[1] < half_pitch_size || pt[0] > frames[0]->img.cols - half_pitch_size || pt[1] > frames[0]->img.rows - half_pitch_size)continue;
            point3d tmp;
            tmp[0] = (pt[0] - cx) / fx;
            tmp[1] = (pt[1] - cy) / fy;
            tmp[2] = 1.0;
            int ptvalue = frames[0]->img.at<uchar>(pt[1], pt[0]);
            vector<int> depths;
            for (int i = 0; i < pitch_size * pitch_size; ++i) {
                if (abs(ptvalue - frames[0]->img.at<uchar>(pt[1] + i / pitch_size - half_pitch_size, pt[0] + i % pitch_size - half_pitch_size) < 3)) {
                    auto cur_depth(frames[0]->depth.at<ushort>(pt[1] + i / pitch_size - half_pitch_size, pt[0] + i % pitch_size - half_pitch_size));
                    if (cur_depth > 1)
                        depths.push_back(cur_depth);
                }
            }

            int arrive_depth = 0;
            for (auto dep : depths) {
                arrive_depth += dep;
            }
            
            arrive_depth = arrive_depth / depths.size();

            tmp *= arrive_depth / baseline;
            select_points.push_back(tmp);
            is_availiable.push_back(true);
        }
    }

    double downSimpleSize = 1.8;

    int MaxLevel = 3;

    //用来存储每一层降采样的实际缩小倍数
    vector<double> sizes(MaxLevel, 1.0);
    if (MaxLevel > 1.0) {
        for (int i = MaxLevel - 2; i > -1; --i)
            sizes[i] = sizes[i + 1] * downSimpleSize;
    }

    Sophus::SE3d T = frames[1]->T * frames[2]->T;

    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;

    for (int level = MaxLevel - 1; level > -1; --level) {
        //当前层所用到的图像
        cv::Mat cur_img1, cur_img2;
        if (1.0 == sizes[level]) {
            cur_img1 = frames[0]->img;
            cur_img2 = frames[2]->img;
        }
        else {
            cur_img1 = usefulTool::DownSampling(frames[0]->img, sizes[level]);
            cur_img2 = usefulTool::DownSampling(frames[2]->img, sizes[level]);
        }
        //计算当前层的内参
        double cfx = fx / sizes[level];
        double cfy = fy / sizes[level];
        double ccx = cx / sizes[level];
        double ccy = cy / sizes[level];

        //每一层迭代前的数据准备
        double cost = 0.0, lastcost = 0.0;
        int maxIter = 30;
        for (int iter = 0; iter < maxIter; ++iter) {
            int inBoundaryCount = 0;
            cost = 0.0;

            H = Eigen::Matrix<double, 6, 6>::Zero();  // 6x6 Hessian
            b = Eigen::Matrix<double, 6, 1>::Zero();  // 6x1 bias
            
            for (int i = 0; i < select_points.size(); ++i) {
                
                point3d current_pt = select_points[i];
                point3d project_pt = T * current_pt;
                
                double x = project_pt[0], y = project_pt[1], z = project_pt[2], zz = z * z;
                point2d pro2d, src2d;
                project_pt /= project_pt[2];
                pro2d[0] = project_pt[0] * cfx + ccx;
                pro2d[1] = project_pt[1] * cfy + ccy;

                //越界则跳过，此时inBoundaryCount不进行累加
                if (pro2d[0] < 0 || pro2d[0] > cur_img2.cols || pro2d[1] < 0 || pro2d[1] > cur_img2.rows) {
                    continue;
                }
                inBoundaryCount++;
                
                current_pt /= current_pt[2];
                src2d[0] = current_pt[0] * cfx + ccx;
                src2d[1] = current_pt[1] * cfy + ccy;

                double error = 0.0;
                try {
                    error = usefulTool::getPixelValue(cur_img1, src2d[0], src2d[1]) - usefulTool::getPixelValue(cur_img2, pro2d[0], pro2d[1]);
                    /*error += usefulTool::getPixelValue(cur_img1, src2d[0] + 1, src2d[1]) - usefulTool::getPixelValue(cur_img2, pro2d[0] + 1, pro2d[1]);
                    error += usefulTool::getPixelValue(cur_img1, src2d[0], src2d[1] + 1) - usefulTool::getPixelValue(cur_img2, pro2d[0], pro2d[1] + 1);
                    error += usefulTool::getPixelValue(cur_img1, src2d[0] - 1, src2d[1]) - usefulTool::getPixelValue(cur_img2, pro2d[0] - 1, pro2d[1]);
                    error += usefulTool::getPixelValue(cur_img1, src2d[0], src2d[1] - 1) - usefulTool::getPixelValue(cur_img2, pro2d[0], pro2d[1] - 1);
                    error /= 5;*/
                }
                catch (...) {
                    inBoundaryCount--;
                    continue;
                }
               
                Eigen::Matrix<double, 2, 6> J_pixel_xi;   // pixel to \xi in Lie algebra
                
                J_pixel_xi(0, 0) = cfx / z;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -x * cfx / zz;
                J_pixel_xi(0, 3) = -x * y * cfx / zz;
                J_pixel_xi(0, 4) = cfx + cfx * x * x / zz;
                J_pixel_xi(0, 5) = -cfx * y / z;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = cfy / z;
                J_pixel_xi(1, 2) = -cfy * y / zz;
                J_pixel_xi(1, 3) = -cfy - cfy * y * y / zz;
                J_pixel_xi(1, 4) = cfy * x * y / zz;
                J_pixel_xi(1, 5) = cfy * x / z;

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
    frames[2]->T = frames[1]->T.inverse() * T;
}


void clvo::optimizer::runLocalMap(vector<Sophus::SE3d>& poses, vector<point3d>& points, vector<clvo::observation>& observations) {
    B.clear();
    C.clear();
    E.clear();

    int poseSize = poses.size();
    int ptSize = points.size();
    B.resize(poseSize);
    C.resize(ptSize);
    E.resize(poseSize * ptSize);

    computeJacobian(poses, points, observations);
}

void clvo::optimizer::computeJacobian(vector<Sophus::SE3d>& poses, vector<point3d>& points, vector<clvo::observation>& observations) {

    //第一步，将当前Hessian矩阵清空
    for (auto b : B) {
        b = H_cam::Zero();
    }
    for (auto c : C) {
        c = H_pt::Zero();
    }
    for (auto e : E) {
        e = H_Edge::Zero();
    }
    //第二步，遍历所有观测，计算对应的雅可比以及Hessian
    for (auto ob : observations) {

        // 2.1 计算Jacobian
        Eigen::Matrix<double, 3, 6> Jaco_pose = Eigen::Matrix<double, 3, 6>::Zero();
        Eigen::Matrix<double, 3, 3> Jaco_pt;

        /*
        Jaco_pose
        | 1  0  0  0    z  -y |
        | 0  1  0  -z   0   x |
        | 0  0  1  y   -x   0 |
        */

        point3d& tmp = ob.obs;

        Jaco_pose(0, 0) = 1;
        Jaco_pose(1, 1) = 1;
        Jaco_pose(2, 2) = 1;

        Jaco_pose(0, 4) = tmp[2];
        Jaco_pose(0, 5) = -tmp[1];
        Jaco_pose(1, 3) = -tmp[2];
        Jaco_pose(1, 5) = tmp[0];
        Jaco_pose(2, 3) = tmp[1];
        Jaco_pose(2, 4) = -tmp[0];

        /*
        Jaco_pt
        | R |
        */
        Jaco_pt = poses[ob.camera].rotationMatrix();

        //2.2 计算H矩阵对应的 B、E、C
        B[ob.camera] += Jaco_pose.transpose() * Jaco_pose;
        C[ob.point] += Jaco_pt.transpose() * Jaco_pt;
    }
}

void clvo::optimizer::runPostOnly(vector< std::shared_ptr<frame> >& vpframes, int start, int end) {

    /*具体的实现方式可以参考：
        METHODS FOR NON - LINEARLEAST SQUARES PROBLEMS
        2nd Edition, April 2004
        K.Madsen, H.B.Nielsen, O.Tingleff
        page 24 in the book
    */
    //配置基本信息
    //这里我们仅需要B矩阵
    B.clear();
    
}