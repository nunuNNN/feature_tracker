#include "TrackMsckfVio.h"


void TrackMsckfVio::feed_stereo(double timestamp, cv::Mat &img_leftin, cv::Mat &img_rightin, size_t cam_id_left, size_t cam_id_right) {

    // Start timing
    rT1 =  boost::posix_time::microsec_clock::local_time();

    // Lock this data feed for this camera
    std::unique_lock<std::mutex> lck1(mtx_feeds.at(cam_id_left));
    std::unique_lock<std::mutex> lck2(mtx_feeds.at(cam_id_right));


    // clone imgin to img
    cv::Mat img_left, img_right;
    img_left = img_leftin.clone();
    img_right = img_rightin.clone();

    // createImagePyramids()
    cv::buildOpticalFlowPyramid(
        img_left, curr_cam0_pyramid_,
        win_size, pyr_levels, true, cv::BORDER_REFLECT_101,
        cv::BORDER_CONSTANT, false);
    cv::buildOpticalFlowPyramid(
        img_right, curr_cam1_pyramid_,
        win_size, pyr_levels, true, cv::BORDER_REFLECT_101,
        cv::BORDER_CONSTANT, false);
    rT2 =  boost::posix_time::microsec_clock::local_time();

    if (is_first_img)
    {
        perform_detection_msckf_vio(curr_cam0_pyramid_, curr_cam1_pyramid_);

        // 交换状态
        prev_cam0_pyramid_ = curr_cam0_pyramid_;
        prev_cam1_pyramid_ = curr_cam1_pyramid_;
        prev_features_ptr = curr_features_ptr;
          
        // Initialize the current features to empty vectors.
        curr_features_ptr.reset(new GridFeatures());
        for (int code = 0; code <grid_row*grid_col; ++code) 
        {
            (*curr_features_ptr)[code] = vector<FeatureMetaData>(0);
        }
        is_first_img = false;
        return;
    }

    perform_detection_msckf_vio(prev_cam0_pyramid_, prev_cam1_pyramid_);



    // Timing information
    // printf("[TIME-KLT]: %.4f seconds for pyramid\n",(rT2-rT1).total_microseconds() * 1e-6);
    // printf("[TIME-KLT]: %.4f seconds for detection\n",(rT3-rT2).total_microseconds() * 1e-6);
    // printf("[TIME-KLT]: %.4f seconds for temporal klt\n",(rT4-rT3).total_microseconds() * 1e-6);
    // printf("[TIME-KLT]: %.4f seconds for stereo klt\n",(rT5-rT4).total_microseconds() * 1e-6);
    // printf("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n",(rT6-rT5).total_microseconds() * 1e-6, (int)good_left.size());
    // printf("[TIME-KLT]: %.4f seconds for total\n",(rT6-rT1).total_microseconds() * 1e-6);

}


void TrackMsckfVio::perform_detection_msckf_vio(const std::vector<cv::Mat>& img0pyr, const std::vector<cv::Mat>& img1pyr)
{                                   
    cv::Mat detection_img = img0pyr.at(0);
    // Size of each grid
    static int grid_height = detection_img.rows / grid_row;
    static int grid_width = detection_img.cols / grid_col;

    // Create a mask to avoid redetecting existing features.
    cv::Mat mask(detection_img.rows, detection_img.cols, CV_8U, Scalar(1));

    for (const auto &features : *prev_features_ptr)
    {
        for (const auto& feature : features.second)
        {
            const int y = static_cast<int>(feature.cam0_point.y);
            const int x = static_cast<int>(feature.cam0_point.x);

            int up_lim = y-2, bottom_lim = y+3, left_lim = x-2, right_lim = x+3;
            if (up_lim < 0) up_lim = 0;
            if (bottom_lim > detection_img.rows) bottom_lim = detection_img.rows;
            if (left_lim < 0) left_lim = 0;
            if (right_lim > detection_img.cols) right_lim = detection_img.cols;

            Range row_range(up_lim, bottom_lim);
            Range col_range(left_lim, right_lim);
            mask(row_range, col_range) = 0;
        }
    }

    // Detect new features.
    vector<KeyPoint> new_features(0);
    detector_ptr->detect(detection_img, new_features, mask);

    // Collect the new detected features based on the grid.
    // Select the ones with top response within each grid afterwards.
    vector<vector<KeyPoint> > new_feature_sieve(grid_row*grid_col);
    for (const auto& feature : new_features) 
    {
        int row = static_cast<int>(feature.pt.y / grid_height);
        int col = static_cast<int>(feature.pt.x / grid_width);
        new_feature_sieve[row*grid_row+col].push_back(feature);
    }

    new_features.clear();
    for (auto& item : new_feature_sieve) 
    {
        if (item.size() > grid_max_feature_num) 
        {
            std::sort(item.begin(), item.end(), &TrackMsckfVio::keyPointCompareByResponse);
            item.erase(item.begin()+grid_max_feature_num, item.end());
        }
        new_features.insert(new_features.end(), item.begin(), item.end());
    }

    int detected_new_features = new_features.size();

    // Find the stereo matched points for the newly
    // detected features.
    vector<cv::Point2f> cam0_points(new_features.size());
    for (int i = 0; i < new_features.size(); ++i)
    {
        cam0_points[i] = new_features[i].pt;
    }

    vector<cv::Point2f> cam1_points(0);
    vector<unsigned char> inlier_markers(0);
    stereoMatch(img0pyr, img1pyr, cam0_points, cam1_points, inlier_markers);

    vector<cv::Point2f> cam0_inliers(0);
    vector<cv::Point2f> cam1_inliers(0);
    vector<float> response_inliers(0);
    for (int i = 0; i < inlier_markers.size(); ++i) 
    {
        if (inlier_markers[i] == 0) continue;
        cam0_inliers.push_back(cam0_points[i]);
        cam1_inliers.push_back(cam1_points[i]);
        response_inliers.push_back(new_features[i].response);
    }

    int matched_new_features = cam0_inliers.size();

    if (matched_new_features < 5 &&
        static_cast<double>(matched_new_features)/
        static_cast<double>(detected_new_features) < 0.1)
    {
        cout << "Images at [%f] seems unsynced..."<< endl;
    }

    // Group the features into grids
    GridFeatures grid_new_features;
    for (int code = 0; code < grid_row*grid_col; ++code)
    {
        grid_new_features[code] = vector<FeatureMetaData>(0);
    }

    for (int i = 0; i < cam0_inliers.size(); ++i) 
    {
        const cv::Point2f& cam0_point = cam0_inliers[i];
        const cv::Point2f& cam1_point = cam1_inliers[i];
        const float& response = response_inliers[i];

        int row = static_cast<int>(cam0_point.y / grid_height);
        int col = static_cast<int>(cam0_point.x / grid_width);
        int code = row*grid_row + col;

        FeatureMetaData new_feature;
        new_feature.response = response;
        new_feature.cam0_point = cam0_point;
        new_feature.cam1_point = cam1_point;
        grid_new_features[code].push_back(new_feature);
    }

    // Sort the new features in each grid based on its response.
    for (auto& item : grid_new_features)
    {
        std::sort(item.second.begin(), item.second.end(), &TrackMsckfVio::featureCompareByResponse);
    }

    int new_added_feature_num = 0;
    // Collect new features within each grid with high response.
    for (int code = 0; code < grid_row*grid_col; ++code) 
    {
        vector<FeatureMetaData>& features_this_grid = (*prev_features_ptr)[code];
        vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

        if (features_this_grid.size() >= grid_min_feature_num) continue;

        int vacancy_num = grid_min_feature_num - features_this_grid.size();
        for (int k = 0; k < vacancy_num && k < new_features_this_grid.size(); ++k) {
        features_this_grid.push_back(new_features_this_grid[k]);
        features_this_grid.back().id = next_feature_id++;
        features_this_grid.back().lifetime = 1;

        ++new_added_feature_num;
        }
    }

    //printf("\033[0;33m detected: %d; matched: %d; new added feature: %d\033[0m\n",
    //    detected_new_features, matched_new_features, new_added_feature_num);

    // grid_max_feature_num
    for (auto& item : *prev_features_ptr) 
    {
        auto& grid_features = item.second;
        // Continue if the number of features in this grid does
        // not exceed the upper bound.
        if (grid_features.size() <= grid_max_feature_num) continue;
        std::sort(grid_features.begin(), grid_features.end(), &TrackMsckfVio::featureCompareByLifetime);
        grid_features.erase(grid_features.begin()+grid_max_feature_num, grid_features.end());
    }

    return;
}

void TrackMsckfVio::stereoMatch(const std::vector<cv::Mat>& img0pyr, 
                                const std::vector<cv::Mat>& img1pyr, 
                                const vector<cv::Point2f>& cam0_points,
                                vector<cv::Point2f>& cam1_points,
                                vector<unsigned char>& inlier_markers) 
{

    if (cam0_points.size() == 0) return;

    // rotation from stereo extrinsics
    const cv::Matx33d R_cam0_cam1 = R_cam_imu.at(0).t() * R_cam_imu.at(1);
    // Compute the relative rotation between the cam0
    // frame and cam1 frame.
    const cv::Vec3d t_cam0_cam1 = R_cam_imu.at(1).t() * (t_cam_imu.at(0)-t_cam_imu.at(1));
    // Initialize cam1_points by projecting cam0_points to cam1 using the
    const cv::Matx33d camK0 = camera_k_OPENCV.at(0);
    const cv::Vec4d camD0 = camera_d_OPENCV.at(0);
    const cv::Matx33d camK1 = camera_k_OPENCV.at(1);
    const cv::Vec4d camD1 = camera_d_OPENCV.at(1);

    if (cam1_points.size() == 0) 
    {
        // undistortPoints
        vector<cv::Point2f> cam0_points_undistorted;
        cv::undistortPoints(cam0_points, cam0_points_undistorted, 
                            camK0, camD0, R_cam0_cam1);
        // distortPoints
        vector<cv::Point3f> homogenous_pts;
        cv::convertPointsToHomogeneous(cam0_points_undistorted, homogenous_pts);
        cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), 
                        camK1, camD1, cam1_points);
    }

    // Track features using LK optical flow method.
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.01);
    calcOpticalFlowPyrLK(img0pyr, img1pyr, cam0_points, cam1_points,
                    inlier_markers, noArray(), win_size, pyr_levels, term_crit,
                    cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (int i = 0; i < cam1_points.size(); ++i) 
    {
        if (inlier_markers[i] == 0) continue;
        
        if (cam1_points[i].y < 0 ||
            cam1_points[i].y > img0pyr.at(0).rows-1 ||
            cam1_points[i].x < 0 ||
            cam1_points[i].x > img0pyr.at(0).cols-1)
        inlier_markers[i] = 0;
    }

    // Compute the essential matrix.
    const cv::Matx33d t_cam0_cam1_hat(
        0.0, -t_cam0_cam1[2], t_cam0_cam1[1],
        t_cam0_cam1[2], 0.0, -t_cam0_cam1[0],
        -t_cam0_cam1[1], t_cam0_cam1[0], 0.0);
    const cv::Matx33d E = t_cam0_cam1_hat * R_cam0_cam1;

    // Further remove outliers based on the known
    // essential matrix.
    vector<cv::Point2f> cam0_points_undistorted(0);
    vector<cv::Point2f> cam1_points_undistorted(0);
    cv::undistortPoints(cam0_points, cam0_points_undistorted, camK0, camD0);
    cv::undistortPoints(cam1_points, cam1_points_undistorted, camK1, camD1);

    double norm_pixel_unit = 4.0 / (camK0(0,0)+camK0(1,1)+camK1(0,0)+camK1(1,1));

    for (int i = 0; i < cam0_points_undistorted.size(); ++i) 
    {
        if (inlier_markers[i] == 0) continue;

        cv::Vec3d pt0(cam0_points_undistorted[i].x, cam0_points_undistorted[i].y, 1.0);
        cv::Vec3d pt1(cam1_points_undistorted[i].x, cam1_points_undistorted[i].y, 1.0);
        cv::Vec3d epipolar_line = E * pt0;
        double error = fabs((pt1.t() * epipolar_line)[0]) / sqrt(
            epipolar_line[0]*epipolar_line[0]+epipolar_line[1]*epipolar_line[1]);

        if (error > stereo_threshold*norm_pixel_unit)
            inlier_markers[i] = 0;
    }

    return;
}


// void TrackMsckfVio::trackFeatures(const std::vector<cv::Mat>& img0pyr, 
//                                 const std::vector<cv::Mat>& img1pyr,
//                                 const std::vector<cv::Mat>& img0lastpyr, 
//                                 const std::vector<cv::Mat>& img1lastpyr) 
// {
//     // Size of each grid.
//     static int grid_height = img0pyr.at(0).rows / grid_row;
//     static int grid_width = img0pyr.at(0).cols / grid_col;

//     // Compute a rough relative rotation which takes a vector
//     // from the previous frame to the current frame.
//     Matx33f cam0_R_p_c;
//     Matx33f cam1_R_p_c;
//     integrateImuData(cam0_R_p_c, cam1_R_p_c);

//     // Organize the features in the previous image.
//     vector<FeatureIDType> prev_ids(0);
//     vector<int> prev_lifetime(0);
//     vector<Point2f> prev_cam0_points(0);
//     vector<Point2f> prev_cam1_points(0);

//     for (const auto& item : *prev_features_ptr) 
//     {
//         for (const auto& prev_feature : item.second) 
//         {
//         prev_ids.push_back(prev_feature.id);
//         prev_lifetime.push_back(prev_feature.lifetime);
//         prev_cam0_points.push_back(prev_feature.cam0_point);
//         prev_cam1_points.push_back(prev_feature.cam1_point);
//         }
//     }

//     // Number of the features before tracking.
//     before_tracking = prev_cam0_points.size();

//     // Abort tracking if there is no features in
//     // the previous frame.
//     if (prev_ids.size() == 0) return;

//     // Track features using LK optical flow method.
//     vector<Point2f> curr_cam0_points(0);
//     vector<unsigned char> track_inliers(0);

//     curr_cam0_points.resize(prev_cam0_points.size());
//     cv::Matx33f camK0 = camera_k_OPENCV.at(0);
//     cv::Matx33f H = camK0 * cam0_R_p_c * camK0.inv();
//     for (int i = 0; i < prev_cam0_points.size(); ++i) {
//         cv::Vec3f p1(prev_cam0_points[i].x, prev_cam0_points[i].y, 1.0f);
//         cv::Vec3f p2 = H * p1;
//         curr_cam0_points[i].x = p2[0] / p2[2];
//         curr_cam0_points[i].y = p2[1] / p2[2];
//     }

//     // Track features using LK optical flow method.
//     cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 15, 0.01);
//     calcOpticalFlowPyrLK(img0pyr, img0lastpyr, prev_cam0_points, curr_cam0_points,
//                     track_inliers, noArray(), win_size, pyr_levels, term_crit,
//                     cv::OPTFLOW_USE_INITIAL_FLOW);

//     // Mark those tracked points out of the image region
//     // as untracked.
//     for (int i = 0; i < curr_cam0_points.size(); ++i) 
//     {
//         if (track_inliers[i] == 0) continue;
//         if (curr_cam0_points[i].y < 0 ||
//             curr_cam0_points[i].y > img0pyr.at(0).rows-1 ||
//             curr_cam0_points[i].x < 0 ||
//             curr_cam0_points[i].x > img0pyr.at(0).cols-1)ååå
//             track_inliers[i] = 0;
//     }

//     // Collect the tracked points.
//     vector<FeatureIDType> prev_tracked_ids(0);
//     vector<int> prev_tracked_lifetime(0);
//     vector<Point2f> prev_tracked_cam0_points(0);
//     vector<Point2f> prev_tracked_cam1_points(0);
//     vector<Point2f> curr_tracked_cam0_points(0);

//     removeUnmarkedElements(
//         prev_ids, track_inliers, prev_tracked_ids);
//     removeUnmarkedElements(
//         prev_lifetime, track_inliers, prev_tracked_lifetime);
//     removeUnmarkedElements(
//         prev_cam0_points, track_inliers, prev_tracked_cam0_points);
//     removeUnmarkedElements(
//         prev_cam1_points, track_inliers, prev_tracked_cam1_points);
//     removeUnmarkedElements(
//         curr_cam0_points, track_inliers, curr_tracked_cam0_points);

//     // Number of features left after tracking.
//     after_tracking = curr_tracked_cam0_points.size();


//     // Outlier removal involves three steps, which forms a close
//     // loop between the previous and current frames of cam0 (left)
//     // and cam1 (right). Assuming the stereo matching between the
//     // previous cam0 and cam1 images are correct, the three steps are:
//     //
//     // prev frames cam0 ----------> cam1
//     //              |                |
//     //              |ransac          |ransac
//     //              |   stereo match |
//     // curr frames cam0 ----------> cam1
//     //
//     // 1) Stereo matching between current images of cam0 and cam1.
//     // 2) RANSAC between previous and current images of cam0.
//     // 3) RANSAC between previous and current images of cam1.
//     //
//     // For Step 3, tracking between the images is no longer needed.
//     // The stereo matching results are directly used in the RANSAC.

//     // Step 1: stereo matching.
//     vector<Point2f> curr_cam1_points(0);
//     vector<unsigned char> match_inliers(0);
//     stereoMatch(img0pyr, img1pyr, curr_tracked_cam0_points, curr_cam1_points, match_inliers);

//     vector<FeatureIDType> prev_matched_ids(0);
//     vector<int> prev_matched_lifetime(0);
//     vector<Point2f> prev_matched_cam0_points(0);
//     vector<Point2f> prev_matched_cam1_points(0);
//     vector<Point2f> curr_matched_cam0_points(0);
//     vector<Point2f> curr_matched_cam1_points(0);

//     removeUnmarkedElements(
//         prev_tracked_ids, match_inliers, prev_matched_ids);
//     removeUnmarkedElements(
//         prev_tracked_lifetime, match_inliers, prev_matched_lifetime);
//     removeUnmarkedElements(
//         prev_tracked_cam0_points, match_inliers, prev_matched_cam0_points);
//     removeUnmarkedElements(
//         prev_tracked_cam1_points, match_inliers, prev_matched_cam1_points);
//     removeUnmarkedElements(
//         curr_tracked_cam0_points, match_inliers, curr_matched_cam0_points);
//     removeUnmarkedElements(
//         curr_cam1_points, match_inliers, curr_matched_cam1_points);

//     // Number of features left after stereo matching.
//     after_matching = curr_matched_cam0_points.size();

//     // Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.
//     vector<int> cam0_ransac_inliers(0);
//     twoPointRansac(prev_matched_cam0_points, curr_matched_cam0_points,
//         cam0_R_p_c, cam0_intrinsics, cam0_distortion_model,
//         cam0_distortion_coeffs, processor_config.ransac_threshold,
//         0.99, cam0_ransac_inliers);

//     vector<int> cam1_ransac_inliers(0);
//     twoPointRansac(prev_matched_cam1_points, curr_matched_cam1_points,
//         cam1_R_p_c, cam1_intrinsics, cam1_distortion_model,
//         cam1_distortion_coeffs, processor_config.ransac_threshold,
//         0.99, cam1_ransac_inliers);

//     // Number of features after ransac.
//     after_ransac = 0;

//     for (int i = 0; i < cam0_ransac_inliers.size(); ++i) 
//     {
//         if (cam0_ransac_inliers[i] == 0 || cam1_ransac_inliers[i] == 0) 
//             continue;
//         int row = static_cast<int>(curr_matched_cam0_points[i].y / grid_height);
//         int col = static_cast<int>(curr_matched_cam0_points[i].x / grid_width);
//         int code = row*processor_config.grid_col + col;
//         (*curr_features_ptr)[code].push_back(FeatureMetaData());

//         FeatureMetaData& grid_new_feature = (*curr_features_ptr)[code].back();
//         grid_new_feature.id = prev_matched_ids[i];
//         grid_new_feature.lifetime = ++prev_matched_lifetime[i];
//         grid_new_feature.cam0_point = curr_matched_cam0_points[i];
//         grid_new_feature.cam1_point = curr_matched_cam1_points[i];

//         ++after_ransac;
//     }

//     // Compute the tracking rate.
//     int prev_feature_num = 0;
//     for (const auto& item : *prev_features_ptr)
//         prev_feature_num += item.second.size();

//     int curr_feature_num = 0;
//     for (const auto& item : *curr_features_ptr)
//         curr_feature_num += item.second.size();

//     //printf(
//     //    "\033[0;32m candidates: %d; raw track: %d; stereo match: %d; ransac: %d/%d=%f\033[0m\n",
//     //    before_tracking, after_tracking, after_matching,
//     //    curr_feature_num, prev_feature_num,
//     //    static_cast<double>(curr_feature_num)/
//     //    (static_cast<double>(prev_feature_num)+1e-5));

//     return;
// }

// void TrackMsckfVio::twoPointRansac(const vector<Point2f>& pts1, const vector<Point2f>& pts2,
//                                 const cv::Matx33f& R_p_c, const cv::Vec4d& intrinsics,
//                                 const std::string& distortion_model,
//                                 const cv::Vec4d& distortion_coeffs,
//                                 const double& inlier_error,
//                                 const double& success_probability,
//                                 vector<int>& inlier_markers) 
// {

//     // Check the size of input point size.
//     if (pts1.size() != pts2.size())
//         printf("Sets of different size (%lu and %lu) are used...", pts1.size(), pts2.size());

//     double norm_pixel_unit = 2.0 / (intrinsics[0]+intrinsics[1]);
//     int iter_num = static_cast<int>(ceil(log(1-success_probability) / log(1-0.7*0.7)));

//     // Initially, mark all points as inliers.
//     inlier_markers.clear();
//     inlier_markers.resize(pts1.size(), 1);

//     const cv::Matx33d camK0 = camera_k_OPENCV.at(0);
//     const cv::Vec4d camD0 = camera_d_OPENCV.at(0);
//     const cv::Matx33d camK1 = camera_k_OPENCV.at(1);
//     const cv::Vec4d camD1 = camera_d_OPENCV.at(1);

//     // Undistort all the points.
//     vector<Point2f> pts1_undistorted(pts1.size());
//     vector<Point2f> pts2_undistorted(pts2.size());
//     cv::undistortPoints(pts1, pts1_undistorted, camK0, camD0);
//     cv::undistortPoints(pts2, pts2_undistorted, camK0, camD0);

//     // Compenstate the points in the previous image with
//     // the relative rotation.
//     for (auto& pt : pts1_undistorted) 
//     {
//         Vec3f pt_h(pt.x, pt.y, 1.0f);
//         //Vec3f pt_hc = dR * pt_h;
//         Vec3f pt_hc = R_p_c * pt_h;
//         pt.x = pt_hc[0];
//         pt.y = pt_hc[1];
//     }

//     // Normalize the points to gain numerical stability.
//     float scaling_factor = 0.0f;
//     for (int i = 0; i < pts1_undistorted.size(); ++i) 
//     {
//         scaling_factor += sqrt(pts1_undistorted[i].dot(pts1_undistorted[i]));
//         scaling_factor += sqrt(pts2_undistorted[i].dot(pts2_undistorted[i]));
//     }

//     scaling_factor = (pts1_undistorted.size()+pts2_undistorted.size()) / scaling_factor * sqrt(2.0f);

//     for (int i = 0; i < pts1_undistorted.size(); ++i) {
//         pts1_undistorted[i] *= scaling_factor;
//         pts2_undistorted[i] *= scaling_factor;
//     }
//     norm_pixel_unit *= scaling_factor;

//     // Compute the difference between previous and current points,
//     // which will be used frequently later.
//     vector<Point2d> pts_diff(pts1_undistorted.size());
//     for (int i = 0; i < pts1_undistorted.size(); ++i)
//         pts_diff[i] = pts1_undistorted[i] - pts2_undistorted[i];

//     // Mark the point pairs with large difference directly.
//     // BTW, the mean distance of the rest of the point pairs
//     // are computed.
//     double mean_pt_distance = 0.0;
//     int raw_inlier_cntr = 0;
//     for (int i = 0; i < pts_diff.size(); ++i) 
//     {
//         double distance = sqrt(pts_diff[i].dot(pts_diff[i]));
//         // 25 pixel distance is a pretty large tolerance for normal motion.
//         // However, to be used with aggressive motion, this tolerance should
//         // be increased significantly to match the usage.
//         if (distance > 50.0*norm_pixel_unit) 
//         {
//             inlier_markers[i] = 0;
//         } 
//         else 
//         {
//             mean_pt_distance += distance;
//             ++raw_inlier_cntr;
//         }
//     }
//     mean_pt_distance /= raw_inlier_cntr;

//     // If the current number of inliers is less than 3, just mark
//     // all input as outliers. This case can happen with fast
//     // rotation where very few features are tracked.
//     if (raw_inlier_cntr < 3) 
//     {
//         for (auto& marker : inlier_markers) marker = 0;
//         return;
//     }

//     // Before doing 2-point RANSAC, we have to check if the motion
//     // is degenerated, meaning that there is no translation between
//     // the frames, in which case, the model of the RANSAC does not
//     // work. If so, the distance between the matched points will
//     // be almost 0.
//     //if (mean_pt_distance < inlier_error*norm_pixel_unit) {
//     if (mean_pt_distance < norm_pixel_unit) 
//     {
//         //ROS_WARN_THROTTLE(1.0, "Degenerated motion...");
//         for (int i = 0; i < pts_diff.size(); ++i) 
//         {
//             if (inlier_markers[i] == 0) 
//                 continue;
//             if (sqrt(pts_diff[i].dot(pts_diff[i])) > inlier_error*norm_pixel_unit)
//                 inlier_markers[i] = 0;
//         }
//         return;
//     }

//     // In the case of general motion, the RANSAC model can be applied.
//     // The three column corresponds to tx, ty, and tz respectively.
//     MatrixXd coeff_t(pts_diff.size(), 3);
//     for (int i = 0; i < pts_diff.size(); ++i) 
//     {
//         coeff_t(i, 0) = pts_diff[i].y;
//         coeff_t(i, 1) = -pts_diff[i].x;
//         coeff_t(i, 2) = pts1_undistorted[i].x*pts2_undistorted[i].y - pts1_undistorted[i].y*pts2_undistorted[i].x;
//     }

//     vector<int> raw_inlier_idx;
//     for (int i = 0; i < inlier_markers.size(); ++i) 
//     {
//         if (inlier_markers[i] != 0)
//             raw_inlier_idx.push_back(i);
//     }

//     vector<int> best_inlier_set;
//     double best_error = 1e10;
//     random_numbers::RandomNumberGenerator random_gen;

//     for (int iter_idx = 0; iter_idx < iter_num; ++iter_idx) 
//     {
//         // Randomly select two point pairs.
//         // Although this is a weird way of selecting two pairs, but it
//         // is able to efficiently avoid selecting repetitive pairs.
//         int select_idx1 = random_gen.uniformInteger(0, raw_inlier_idx.size()-1);
//         int select_idx_diff = random_gen.uniformInteger(1, raw_inlier_idx.size()-1);
//         int select_idx2 = select_idx1+select_idx_diff<raw_inlier_idx.size() ? 
//                         select_idx1+select_idx_diff : select_idx1+select_idx_diff-raw_inlier_idx.size();

//         int pair_idx1 = raw_inlier_idx[select_idx1];
//         int pair_idx2 = raw_inlier_idx[select_idx2];

//         // Construct the model;
//         Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
//         Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
//         Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
//         vector<double> coeff_l1_norm(3);
//         coeff_l1_norm[0] = coeff_tx.lpNorm<1>();
//         coeff_l1_norm[1] = coeff_ty.lpNorm<1>();
//         coeff_l1_norm[2] = coeff_tz.lpNorm<1>();
//         int base_indicator = min_element(coeff_l1_norm.begin(), coeff_l1_norm.end())-coeff_l1_norm.begin();

//         Vector3d model(0.0, 0.0, 0.0);
//         if (base_indicator == 0) {
//             Matrix2d A;
//             A << coeff_ty, coeff_tz;
//             Vector2d solution = A.inverse() * (-coeff_tx);
//             model(0) = 1.0;
//             model(1) = solution(0);
//             model(2) = solution(1);
//         } 
//         else if (base_indicator ==1) 
//         {
//             Matrix2d A;
//             A << coeff_tx, coeff_tz;
//             Vector2d solution = A.inverse() * (-coeff_ty);
//             model(0) = solution(0);
//             model(1) = 1.0;
//             model(2) = solution(1);
//         } 
//         else 
//         {
//             Matrix2d A;
//             A << coeff_tx, coeff_ty;
//             Vector2d solution = A.inverse() * (-coeff_tz);
//             model(0) = solution(0);
//             model(1) = solution(1);
//             model(2) = 1.0;
//         }

//         // Find all the inliers among point pairs.
//         VectorXd error = coeff_t * model;

//         vector<int> inlier_set;
//         for (int i = 0; i < error.rows(); ++i) 
//         {
//             if (inlier_markers[i] == 0) continue;
//             if (std::abs(error(i)) < inlier_error*norm_pixel_unit)
//                 inlier_set.push_back(i);
//         }

//         // If the number of inliers is small, the current
//         // model is probably wrong.
//         if (inlier_set.size() < 0.2*pts1_undistorted.size())
//             continue;

//         // Refit the model using all of the possible inliers.
//         VectorXd coeff_tx_better(inlier_set.size());
//         VectorXd coeff_ty_better(inlier_set.size());
//         VectorXd coeff_tz_better(inlier_set.size());
//         for (int i = 0; i < inlier_set.size(); ++i) 
//         {
//             coeff_tx_better(i) = coeff_t(inlier_set[i], 0);
//             coeff_ty_better(i) = coeff_t(inlier_set[i], 1);
//             coeff_tz_better(i) = coeff_t(inlier_set[i], 2);
//         }

//         Vector3d model_better(0.0, 0.0, 0.0);
//         if (base_indicator == 0) 
//         {
//             MatrixXd A(inlier_set.size(), 2);
//             A << coeff_ty_better, coeff_tz_better;
//             Vector2d solution = (A.transpose() * A).inverse() * A.transpose() * (-coeff_tx_better);
//             model_better(0) = 1.0;
//             model_better(1) = solution(0);
//             model_better(2) = solution(1);
//         } 
//         else if (base_indicator ==1) 
//         {
//             MatrixXd A(inlier_set.size(), 2);
//             A << coeff_tx_better, coeff_tz_better;
//             Vector2d solution = (A.transpose() * A).inverse() * A.transpose() * (-coeff_ty_better);
//             model_better(0) = solution(0);
//             model_better(1) = 1.0;
//             model_better(2) = solution(1);
//         } 
//         else 
//         {
//             MatrixXd A(inlier_set.size(), 2);
//             A << coeff_tx_better, coeff_ty_better;
//             Vector2d solution = (A.transpose() * A).inverse() * A.transpose() * (-coeff_tz_better);
//             model_better(0) = solution(0);
//             model_better(1) = solution(1);
//             model_better(2) = 1.0;
//         }

//         // Compute the error and upate the best model if possible.
//         VectorXd new_error = coeff_t * model_better;

//         double this_error = 0.0;
//         for (const auto& inlier_idx : inlier_set)
//             this_error += std::abs(new_error(inlier_idx));
//         this_error /= inlier_set.size();

//         if (inlier_set.size() > best_inlier_set.size()) {
//             best_error = this_error;
//             best_inlier_set = inlier_set;
//         }
//     }

//     // Fill in the markers.
//     inlier_markers.clear();
//     inlier_markers.resize(pts1.size(), 0);
//     for (const auto& inlier_idx : best_inlier_set)
//         inlier_markers[inlier_idx] = 1;

//     //printf("inlier ratio: %lu/%lu\n",
//     //    best_inlier_set.size(), inlier_markers.size());

//     return;
// }
