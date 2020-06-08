#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "FeatureTrackerOptions.h"
#include "track/TrackKLT.h"

using namespace std;
using namespace Eigen;
using namespace cv;

using namespace feature_tracker;

/// Our sparse feature tracker
TrackBase* trackFEATS = nullptr;


string data_folder = "‎⁨/Users/zhangjingwen/Downloads/liudong/pro/dataset/PV_1";
string param_folder = "../config/test.yaml";

void load_params(FeatureTrackerOptions &params)
{
    // Read rectification parameters
    cv::FileStorage fsParams(param_folder, cv::FileStorage::READ);
    if (!fsParams.isOpened())
    {
        cout << "ERROR: Wrong path to settings" << endl;
        return;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsParams["LEFT.K"] >> K_l;
    fsParams["RIGHT.K"] >> K_r;

    fsParams["LEFT.P"] >> P_l;
    fsParams["RIGHT.P"] >> P_r;

    fsParams["LEFT.R"] >> R_l;
    fsParams["RIGHT.R"] >> R_r;

    fsParams["LEFT.D"] >> D_l;
    fsParams["RIGHT.D"] >> D_r;

    int rows_l = fsParams["LEFT.height"];
    int cols_l = fsParams["LEFT.width"];
    int rows_r = fsParams["RIGHT.height"];
    int cols_r = fsParams["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
        rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cout << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return ;
    }

    // and load each of the cameras
    for(int i=0; i<2; i++) {
        // If the desired fov we should simulate
        std::pair<int,int> wh(rows_l,cols_l);

        // Camera intrinsic properties
        Eigen::Matrix<double,8,1> cam_calib;
        if (i == 0) {
            std::vector<double> matrix_k = {K_l.at<float>(0,0),K_l.at<float>(1,1),K_l.at<float>(0,2),K_l.at<float>(1,2)};
            std::vector<double> matrix_d = {D_l.at<float>(0),D_l.at<float>(1),D_l.at<float>(2),D_l.at<float>(3)};
            cam_calib << matrix_k.at(0),matrix_k.at(1),matrix_k.at(2),matrix_k.at(3),matrix_d.at(0),matrix_d.at(1),matrix_d.at(2),matrix_d.at(3);
        }
        else if (i == 1) {
            std::vector<double> matrix_k = {K_r.at<float>(0,0),K_r.at<float>(1,1),K_r.at<float>(0,2),K_r.at<float>(1,2)};
            std::vector<double> matrix_d = {D_r.at<float>(0),D_r.at<float>(1),D_r.at<float>(2),D_r.at<float>(3)};
            cam_calib << matrix_k.at(0),matrix_k.at(1),matrix_k.at(2),matrix_k.at(3),matrix_d.at(0),matrix_d.at(1),matrix_d.at(2),matrix_d.at(3);
        }

        // Our camera extrinsics transform
        Eigen::Matrix4d T_CtoI;
        if (i == 0) {
            std::vector<double> matrix_TCtoI = {R_l.at<float>(0,0),R_l.at<float>(0,1),R_l.at<float>(0,2),0,
                                                R_l.at<float>(1,0),R_l.at<float>(1,1),R_l.at<float>(1,2),0,
                                                R_l.at<float>(2,0),R_l.at<float>(2,1),R_l.at<float>(2,2),0,
                                                0,0,0,1};
            T_CtoI << matrix_TCtoI.at(0),matrix_TCtoI.at(1),matrix_TCtoI.at(2),matrix_TCtoI.at(3),
                    matrix_TCtoI.at(4),matrix_TCtoI.at(5),matrix_TCtoI.at(6),matrix_TCtoI.at(7),
                    matrix_TCtoI.at(8),matrix_TCtoI.at(9),matrix_TCtoI.at(10),matrix_TCtoI.at(11),
                    matrix_TCtoI.at(12),matrix_TCtoI.at(13),matrix_TCtoI.at(14),matrix_TCtoI.at(15);
        }

        else if (i == 1) {
            std::vector<double> matrix_TCtoI = {R_r.at<float>(0,0),R_r.at<float>(0,1),R_r.at<float>(0,2),0,
                                                R_r.at<float>(1,0),R_r.at<float>(1,1),R_r.at<float>(1,2),0,
                                                R_r.at<float>(2,0),R_r.at<float>(2,1),R_r.at<float>(2,2),0,
                                                0,0,0,1};
            T_CtoI << matrix_TCtoI.at(0),matrix_TCtoI.at(1),matrix_TCtoI.at(2),matrix_TCtoI.at(3),
                    matrix_TCtoI.at(4),matrix_TCtoI.at(5),matrix_TCtoI.at(6),matrix_TCtoI.at(7),
                    matrix_TCtoI.at(8),matrix_TCtoI.at(9),matrix_TCtoI.at(10),matrix_TCtoI.at(11),
                    matrix_TCtoI.at(12),matrix_TCtoI.at(13),matrix_TCtoI.at(14),matrix_TCtoI.at(15);
        }



        // Load these into our state
        Eigen::Matrix<double,7,1> cam_eigen;
        cam_eigen.block(0,0,4,1) = rot_2_quat(T_CtoI.block(0,0,3,3).transpose());
        cam_eigen.block(4,0,3,1) = -T_CtoI.block(0,0,3,3).transpose()*T_CtoI.block(0,3,3,1);

        // Insert
        params.camera_intrinsics.insert({i, cam_calib});
        params.camera_extrinsics.insert({i, cam_eigen});
        params.camera_wh.insert({i, wh});
    }

}


void feed_measurement_imu()
{
    string imuFile = "/Users/zhangjingwen/Downloads/liudong/pro/dataset/PV_1/imu.txt";

    std::ifstream f;
    f.open(imuFile.c_str());
    if (!f.is_open())
    {
        cout << "imu data path is not open! file: " << imuFile << endl;
        return;
    }

    Vector3d vAcc;
    Vector3d vGyr;

    while (!f.eof())
    {
        std::string s;
        std::getline(f, s);

        if (!s.empty())
        {
            stringstream ss;
            ss << s;

            double dStampUSec;
            ss >> dStampUSec >> vAcc.x() >> vAcc.y() >> vAcc.z() >>
                vGyr.x() >> vGyr.y() >> vGyr.z();

            trackFEATS->feed_imu(dStampUSec * 1e3, vAcc, vGyr);
        }
    }
}


void feed_measurement_stereo()
{
    string strFile = "/Users/zhangjingwen/Downloads/liudong/pro/dataset/PV_1/forw/timestamp.txt";
    string filepath = "/Users/zhangjingwen/Downloads/liudong/pro/dataset/PV_1/forw/";

    // Retrieve paths to images
    vector<string> vstr_left_image;
    vector<string> vstr_right_image;
    vector<double> vTimeStamp;

    ifstream f; f.open(strFile.c_str());
    if (!f.is_open())
    {
        cout << "Image path is not open! file: " << strFile << endl;
        return;
    }
    while (!f.eof())
    {
        string s;
        getline(f, s);
        if (!s.empty())
        {
            stringstream ss; ss << s; 
            int t; ss >> t;
            string left_image_file = "left/" + to_string(t) + ".jpg";
            string right_image_file = "right/" + to_string(t) + ".jpg";

            vTimeStamp.push_back(t * 1e3);
            vstr_left_image.push_back(left_image_file);
            vstr_right_image.push_back(right_image_file);
        }
    }
    cout << "Image size is: " << vstr_left_image.size() << endl;

    const int nImages = vstr_left_image.size();

    cv::Mat im_right, im_left;
    for (int ni = 0; ni < nImages; ni += 1)
    {
        if (vTimeStamp[ni] < 60e9 || vTimeStamp[ni] > 300e9)
        {
            // continue;
        }

        im_left = cv::imread(filepath + vstr_left_image[ni]);
        im_right = cv::imread(filepath + vstr_right_image[ni]);

        // imshow("im_left", im_left);
        // imshow("im_right", im_right);
        // cv::waitKey(0);

        if(im_left.empty() || im_right.empty())
        {
            cout << "Image is empty! " << filepath + "/" + vstr_left_image[ni] << endl;
            continue;
        }

        trackFEATS->feed_stereo((uint64_t)vTimeStamp[ni], im_left, im_right, 0, 1);

        usleep(20000);
    }
}


int main()
{
    // 加载params
    FeatureTrackerOptions params;
    load_params(params);
    // 打印参数信息
    params.print_trackers();
    params.print_state();

    trackFEATS = new TrackKLT(params.num_pts,0,params.fast_threshold,params.grid_x,params.grid_y,params.min_px_dist);
    trackFEATS->set_calibration(params.camera_intrinsics);


    thread thd_pub_imu(feed_measurement_imu);
    thd_pub_imu.join();

    feed_measurement_stereo();

    // while (true)
    {
        cout << "------------" << endl;
        sleep(10);
    }

    return 0;
}
