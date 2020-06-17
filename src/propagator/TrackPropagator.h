/**
 * 文件说明
*/

#ifndef TRACK_PROPAGATOR_H
#define TRACK_PROPAGATOR_H

#include <vector>
#include <mutex>
#include <Eigen/Eigen>

namespace feature_tracker {

    /**
     * @brief Performs the state covariance and mean propagation using imu measurements
     *
     * We will first select what measurements we need to propagate with.
     * We then compute the state transition matrix at each step and update the state and covariance.
     * For derivations look at @ref propagation page which has detailed equations.
     */
    class TrackPropagator {
    
    public:

        TrackPropagator() {}

        /**
         * @brief Struct for a single imu measurement (time, wm, am)
         */
        struct IMUDATA {

            /// Timestamp of the reading
            double timestamp;

            /// Gyroscope reading, angular velocity (rad/s)
            Eigen::Matrix<double, 3, 1> wm;

            /// Accelerometer reading, linear acceleration (m/s^2)
            Eigen::Matrix<double, 3, 1> am;

        };

    public:

        /**
         * @brief Process new stereo pair of images
         * @param timestamp timestamp this pair occured at (stereo is synchronised)
         * @param am Acceleration measurement
         * @param wm Gyroscope measurement
         */
        void push_imu(double timestamp, Eigen::Vector3d &am, Eigen::Vector3d &wm) {

            std::unique_lock<std::mutex> lck(mtx_imu);

            // Create our imu data object
            IMUDATA data;
            data.timestamp = timestamp;
            data.wm = wm;
            data.am = am;

            // Append it to our vector
            imu_data.emplace_back(data);

            // // Loop through and delete imu messages that are older then 20 seconds
            // // TODO: we should probably have more elegant logic then this
            // // TODO: but this prevents unbounded memory growth and slow prop with high freq imu
            // auto it0 = imu_data.begin();
            // while(it0 != imu_data.end()) {
            //     if(timestamp-(*it0).timestamp > 20) {
            //         it0 = imu_data.erase(it0);
            //     } else {
            //         it0++;
            //     }
            // }
        }

        /***
         * @brief set_calibration
        */
        void set_imu_cam_calib(std::map<size_t,Eigen::Isometry3d> imu_cam_calib)
        {
            // Assert stereo
            assert(cam.size()==2);

            for (auto const &cam : imu_cam_calib) {
                if (cam.first == 0)
                {
                    R_cam0_imu = cam.second.linear();
                    t_cam0_imu = cam.second.translation();
                }
                else if (cam.first == 1)
                {
                    R_cam1_imu = cam.second.linear();
                    t_cam1_imu = cam.second.translation();
                }
            }
        }

        /**
         * @brief propagator 
        */
        void integrate_by_acc(double lastTime, double currTime, Eigen::Matrix3d &cam0_R_p_c, 
                                    Eigen::Matrix3d &cam1_R_p_c, bool remove=false) {
            // Find the start and the end limit within the imu msg buffer.
            std::unique_lock<std::mutex> lck(mtx_imu);

            auto begin_iter = imu_data.begin();
            while (begin_iter != imu_data.end()) {
                if ((begin_iter->timestamp - lastTime) < -0.01)
                    ++begin_iter;
                else
                    break;
            }

            auto end_iter = begin_iter;
            while (end_iter != imu_data.end()) {
                if ((begin_iter->timestamp - currTime) < 0.005)
                    ++end_iter;
                else
                    break;
            }

            Eigen::Vector3d mean_ang_vel(0.0, 0.0, 0.0);
            for (auto iter = begin_iter; iter < end_iter; ++iter)
                mean_ang_vel += Eigen::Vector3d(iter->am.x(), 
                                iter->am.y(), iter->am.z());

            if (end_iter-begin_iter > 0)
                mean_ang_vel *= 1.0f / (end_iter-begin_iter);
            
            // Transform the mean angular velocity from the IMU
            // frame to the cam0 and cam1 frames.
            Eigen::Vector3d cam0_mean_ang_vel = R_cam0_imu.transpose() * mean_ang_vel;
            Eigen::Vector3d cam1_mean_ang_vel = R_cam1_imu.transpose() * mean_ang_vel;

            // Compute the relative rotation.
            double dtime = currTime - lastTime;
            // cv::Rodrigues(cam0_mean_ang_vel*dtime, cam0_R_p_c);
            // cv::Rodrigues(cam1_mean_ang_vel*dtime, cam1_R_p_c);
            cam0_R_p_c = cam0_R_p_c.transpose();
            cam1_R_p_c = cam1_R_p_c.transpose();

            // Delete the useless and used imu messages.
            if (remove)
                imu_data.erase(imu_data.begin(), end_iter);
            else
                imu_data.erase(imu_data.begin(), begin_iter);

        }

        /**
         * @brief propagator 
        */
        static void integrate_by_imu(double lastTime, double currTime, Eigen::Matrix3d &cam0_R_p_c, 
                                    Eigen::Matrix3d &cam1_R_p_c, bool remove=false) {
            // // Find the start and the end limit within the imu msg buffer.
            // auto begin_iter = imu_data.begin();
            // while (begin_iter != imu_data.end()) {
            //     if ((begin_iter.timestamp - lastTime) < -0.01)
            //         ++begin_iter;
            //     else
            //         break;
            // }

            // auto end_iter = begin_iter;
            // while (end_iter != imu_data.end()) {
            //     if ((begin_iter.timestamp - currTime) < 0.005)
            //         ++end_iter;
            //     else
            //         break;
            // }

            // // cal integrate by imu
            // double last_imu_timestamp = iter.timestamp;
            // Eigen::Vector3d last_gyr = iter.wm;
            // Eigen::Quaterniond R_w_from_t = Eigen::Zeros();
            // for (auto iter = begin_iter; iter < end_iter; ++iter)
            // {
            //     // delta time between two frame imu
            //     double dt = iter.timestamp - last_imu_timestamp;
            //     if (dt<0)
            //         continue;

            //     //mean gyr between last and curr
            //     Eigen::Vector3d un_gyr = 0.5 * (last_gyr + iter.wm);
            //     R_w_from_t *= deltaQ(un_gyr * dt);
            //     last_gyr = imu_msg->gyr;
            // }
            ;

        }


    protected:

        /// Mutex lock for our map
        std::mutex mtx_imu;

        /// IMU buffer for rotation between two frames
        std::vector<IMUDATA> imu_data;

        /// Take a vector from cam0 frame to the IMU frame.
        Eigen::Matrix3d R_cam0_imu;
        Eigen::Vector3d t_cam0_imu;
        /// Take a vector from cam1 frame to the IMU frame.
        Eigen::Matrix3d R_cam1_imu;
        Eigen::Vector3d t_cam1_imu;

    }; /* class TrackPropagator */

}

#endif /* TRACK_PROPAGATOR_H */
