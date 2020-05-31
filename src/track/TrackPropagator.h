/**
 * 文件说明
*/

#ifndef TRACK_PROPAGATOR_H
#define TRACK_PROPAGATOR_H

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

        static void (double lastTime, double currTime, std::vector<IMUDATA> &imu_data, 
                     Eigen::Matrix3d &cam0_R_p_c, Eigen::Matrix3d &cam1_R_p_c) {
            // Find the start and the end limit within the imu msg buffer.
            auto begin_iter = imu_data.begin();
            while (begin_iter != imu_data.end()) {
                if ((begin_iter.timestamp - lastTime) < -0.01)
                    ++begin_iter;
                else
                    break;
            }

            auto end_iter = begin_iter;
            while (end_iter != imu_data.end()) {
                if ((begin_iter.timestamp - currTime) < 0.005)
                    ++end_iter;
                else
                    break;
            }

            Eigen::Vector3d mean_ang_vel(0.0, 0.0, 0.0);
            for (auto iter = begin_iter; iter < end_iter; ++iter)
                mean_ang_vel += Eigen::Vector3d(iter.aw.x(), 
                                iter.aw.y(), iter.aw.z());

            if (end_iter-begin_iter > 0)
                mean_ang_vel *= 1.0f / (end_iter-begin_iter);
            
            // Transform the mean angular velocity from the IMU
            // frame to the cam0 and cam1 frames.
            Eigen::Vector3d cam0_mean_ang_vel = R_cam0_imu.t() * mean_ang_vel;
            Eigen::Vector3d cam1_mean_ang_vel = R_cam1_imu.t() * mean_ang_vel;

            // Compute the relative rotation.
            double dtime = currTime - lastTime;
            cv::Rodrigues(cam0_mean_ang_vel*dtime, cam0_R_p_c);
            cv::Rodrigues(cam1_mean_ang_vel*dtime, cam1_R_p_c);
            cam0_R_p_c = cam0_R_p_c.t();
            cam1_R_p_c = cam1_R_p_c.t();

            // Delete the useless and used imu messages.
            imu_data.erase(imu_data.begin(), end_iter);
            return;
        }
    

    }

}

#endif /* TRACK_PROPAGATOR_H */
