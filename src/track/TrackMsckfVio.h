#ifndef TRACK_MSCKF_VIO_H
#define TRACK_MSCKF_VIO_H


#include "TrackBase.h"


namespace feature_tracker {


    /**
     * @brief KLT tracking of features.
     *
     * This is the implementation of a KLT visual frontend for tracking sparse features.
     * We can track either monocular cameras across time (temporally) along with
     * stereo cameras which we also track across time (temporally) but track from left to right
     * to find the stereo correspondence information also.
     * This uses the [calcOpticalFlowPyrLK](https://github.com/opencv/opencv/blob/master/modules/video/src/lkpyramid.cpp) OpenCV function to do the KLT tracking.
     */
    class TrackMsckfVio : public TrackBase {

    public:

        /**
         * @brief Public default constructor
         */
        TrackMsckfVio() : 
            TrackBase(), 
            threshold(10), 
            grid_row(8), 
            grid_col(5)，
            prev_features_ptr(new GridFeatures()),
            curr_features_ptr(new GridFeatures()) 
        {
            // Create feature detector.
            detector_ptr = FastFeatureDetector::create(threshold);
        }

        /**
         * @brief Public constructor with configuration variables
         * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
         * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
         * @param fast_threshold FAST detection threshold
         * @param gridx size of grid in the x-direction / u-direction
         * @param gridy size of grid in the y-direction / v-direction
         * @param minpxdist features need to be at least this number pixels away from each other
         */
        explicit TrackMsckfVio(int numfeats, int numaruco, int fast_threshold, int gridx, int gridy) :
                 TrackBase(numfeats, numaruco),
                 threshold(fast_threshold), 
                 grid_row(gridx), 
                 grid_col(gridy)
                 prev_features_ptr(new GridFeatures()),
                 curr_features_ptr(new GridFeatures()) 
        {
            detector_ptr = FastFeatureDetector::create(threshold);
        }


        /**
         * @brief Process new stereo pair of images
         * @param timestamp timestamp this pair occured at (stereo is synchronised)
         * @param img_left first grayscaled image
         * @param img_right second grayscaled image
         * @param cam_id_left first image camera id
         * @param cam_id_right second image camera id
         */
        void feed_stereo(double timestamp, cv::Mat &img_left, cv::Mat &img_right, size_t cam_id_left, size_t cam_id_right) override;

    private:

        /*
        * @brief FeatureIDType An alias for unsigned long long int.
        */
        typedef unsigned long long int FeatureIDType;

        /*
        * @brief FeatureMetaData Contains necessary information
        *    of a feature for easy access.
        */
        struct FeatureMetaData {
            FeatureIDType id;
            float response;
            int lifetime;
            cv::Point2f cam0_point;
            cv::Point2f cam1_point;
        };

        /*
        * @brief GridFeatures Organize features based on the grid
        *    they belong to. Note that the key is encoded by the
        *    grid index.
        */
        typedef std::map<int, std::vector<FeatureMetaData> > GridFeatures;

        /*
        * @brief keyPointCompareByResponse
        *    Compare two keypoints based on the response.
        */
        static bool keyPointCompareByResponse(const cv::KeyPoint& pt1,
                                                const cv::KeyPoint& pt2) 
        {
            // Keypoint with higher response will be at the
            // beginning of the vector.
            return pt1.response > pt2.response;
        }

        /*
        * @brief featureCompareByResponse
        *    Compare two features based on the response.
        */
        static bool featureCompareByResponse(const FeatureMetaData& f1,
                                                const FeatureMetaData& f2) 
        {
            // Features with higher response will be at the
            // beginning of the vector.
            return f1.response > f2.response;
        }

        /*
        * @brief removeUnmarkedElements Remove the unmarked elements
        *    within a vector.
        * @param raw_vec: vector with outliers.
        * @param markers: 0 will represent a outlier, 1 will be an inlier.
        * @return refined_vec: a vector without outliers.
        *
        * Note that the order of the inliers in the raw_vec is perserved
        * in the refined_vec.
        */
        template <typename T>
        void removeUnmarkedElements(const std::vector<T>& raw_vec,
                                    const std::vector<unsigned char>& markers,
                                    std::vector<T>& refined_vec) 
        {
            if (raw_vec.size() != markers.size()) 
            {
                printf("The input size of raw_vec(%lu) and markers(%lu) does not match...",
                    raw_vec.size(), markers.size());
            }

            for (int i = 0; i < markers.size(); ++i) 
            {
                if (markers[i] == 0) continue;
                refined_vec.push_back(raw_vec[i]);
            }

            return;
        }

        void undistortPoints(const vector<cv::Point2f>& pts_in,
                            cv::Matx33d K,
                            const cv::Vec4d& distortion_coeffs,
                            vector<cv::Point2f>& pts_out,
                            const cv::Matx33d &rectification_matrix,
                            const cv::Vec4d &new_intrinsics) 
        {
            if (pts_in.size() == 0) return;

            const cv::Matx33d K_new(
                new_intrinsics[0], 0.0, new_intrinsics[2],
                0.0, new_intrinsics[1], new_intrinsics[3],
                0.0, 0.0, 1.0);

            cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                                rectification_matrix, K_new);



            return;
        }

        vector<cv::Point2f> ImageProcessor::distortPoints(
            const vector<cv::Point2f>& pts_in,
            const cv::Vec4d& intrinsics,
            const string& distortion_model,
            const cv::Vec4d& distortion_coeffs) {

        const cv::Matx33d K(intrinsics[0], 0.0, intrinsics[2],
                            0.0, intrinsics[1], intrinsics[3],
                            0.0, 0.0, 1.0);

        vector<cv::Point2f> pts_out;
        if (distortion_model == "radtan") {
            vector<cv::Point3f> homogenous_pts;
            cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
            cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                            distortion_coeffs, pts_out);
        } else if (distortion_model == "equidistant") {
            cv::fisheye::distortPoints(pts_in, pts_out, K, distortion_coeffs);
        } else {
            ROS_WARN_ONCE("The model %s is unrecognized, using radtan instead...",
                        distortion_model.c_str());
            vector<cv::Point3f> homogenous_pts;
            cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
            cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                            distortion_coeffs, pts_out);
        }

        return pts_out;
        }


        void perform_detection_msckf_vio();

        void stereoMatch(const std::vector<cv::Mat>& img0pyr, 
                        const std::vector<cv::Mat>& img1pyr，
                        const vector<cv::Point2f>& cam0_points,
                        vector<cv::Point2f>& cam1_points,
                        vector<unsigned char>& inlier_markers);

        void trackFeatures();

        void twoPointRansac(
            const std::vector<cv::Point2f>& pts1,
            const std::vector<cv::Point2f>& pts2,
            const cv::Matx33f& R_p_c,
            const cv::Vec4d& intrinsics,
            const std::string& distortion_model,
            const cv::Vec4d& distortion_coeffs,
            const double& inlier_error,
            const double& success_probability,
            std::vector<int>& inlier_markers);

    protected:

        // Timing variables
        boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

        // Number of features after each outlier removal step.
        int before_tracking;
        int after_tracking;
        int after_matching;
        int after_ransac;

        // Parameters for our FAST grid detector
        int threshold;
        int grid_row;
        int grid_col;

        int grid_min_feature_num = 2;
        int grid_max_feature_num = 4;
        int max_iteration = 30;
        double track_precision = 0.01;
        double ransac_threshold = 3;
        double stereo_threshold = 3;
        // Feature detector
        cv::Ptr<cv::Feature2D> detector_ptr;

        // How many pyramid levels to track on and the window size to reduce by
        int pyr_levels = 3;
        cv::Size win_size = cv::Size(15, 15);

        // ID for the next new feature.
        FeatureIDType next_feature_id;

        // Features in the previous and current image.
        boost::shared_ptr<GridFeatures> prev_features_ptr;
        boost::shared_ptr<GridFeatures> curr_features_ptr;

        // Pyramids for previous and current image
        std::vector<cv::Mat> prev_cam0_pyramid_;
        std::vector<cv::Mat> curr_cam0_pyramid_;
        std::vector<cv::Mat> curr_cam1_pyramid_;

    };


}


#endif /* TRACK_MSCKF_VIO_H */