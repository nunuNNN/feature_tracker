/**
 * 文件说明
*/

#ifndef FEATURE_TRACKER_OPTIONS_H
#define FEATURE_TRACKER_OPTIONS_H

#include <string>
#include <vector>
#include <iostream>

#include <Eigen/Eigen>

#include "utils/quat_ops.h"

using namespace std;
using namespace Eigen;

namespace feature_tracker {

    /**
     * @brief Struct which stores all options needed for feature tracker
     * 
    */
    struct FeatureTrackerOptions {

        int num_cameras = 2;

        /************ tracker *************/ 

        /// If we should use KLT tracking, or descriptor matcher
        bool use_klt = true;

        /// The number of points we should extract and track in *each* image frame. This highly effects the computation required for tracking.
        int num_pts = 150;

        /// Fast extraction threshold
        int fast_threshold = 20;

        /// Number of grids we should split column-wise to do feature extraction in
        int grid_x = 5;

        /// Number of grids we should split row-wise to do feature extraction in
        int grid_y = 5;

        /// Will check after doing KLT track and remove any features closer than this
        int min_px_dist = 10;

        /// KNN ration between top two descriptor matcher which is required to be a good match
        double knn_ratio = 0.85;

                /**
         * @brief This function will print out all parameters releated to our visual trackers.
         */
        void print_trackers() {
            printf("FEATURE TRACKING PARAMETERS:\n");
            printf("\t- num_pts: %d\n", num_pts);
        }

        /************** camera model **************/

        /// Map between camid and intrinsics. Values depends on the model but each should be a 4x1 vector normally.
        std::map<size_t,Eigen::VectorXd> camera_intrinsics;

        /// Map between camid and camera extrinsics (q_ItoC, p_IinC).
        std::map<size_t,Eigen::VectorXd> camera_extrinsics;

        /// Map between camid and the dimensions of incoming images (width/cols, height/rows). This is normally only used during simulation.
        std::map<size_t,std::pair<int,int>> camera_wh;


        /**
         * @brief This function will print out all simulated parameters loaded.
         * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
         */
        void print_state() {
            printf("STATE PARAMETERS:\n");
            assert(num_cameras==(int)camera_intrinsics.size());
            for(int n=0; n<num_cameras; n++) {
                std::cout << "cam_" << n << "_wh:" << endl << camera_wh.at(n).first << " x " << camera_wh.at(n).second << std::endl;
                std::cout << "cam_" << n << "_intrinsic(0:3):" << endl << camera_intrinsics.at(n).block(0,0,4,1).transpose() << std::endl;
                std::cout << "cam_" << n << "_intrinsic(4:7):" << endl << camera_intrinsics.at(n).block(4,0,4,1).transpose() << std::endl;
                std::cout << "cam_" << n << "_extrinsic(0:3):" << endl << camera_extrinsics.at(n).block(0,0,4,1).transpose() << std::endl;
                std::cout << "cam_" << n << "_extrinsic(4:6):" << endl << camera_extrinsics.at(n).block(4,0,3,1).transpose() << std::endl;
                Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
                T_CtoI.block(0,0,3,3) = quat_2_Rot(camera_extrinsics.at(n).block(0,0,4,1)).transpose();
                T_CtoI.block(0,3,3,1) = -T_CtoI.block(0,0,3,3)*camera_extrinsics.at(n).block(4,0,3,1);
                std::cout << "T_C" << n << "toI:" << endl << T_CtoI << std::endl << std::endl;
            }
        }
    };

} // feature_tracker

#endif //FEATURE_TRACKER_OPTIONS_H