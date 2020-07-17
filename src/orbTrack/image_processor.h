#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <vector>
#include <map>
#include <memory>
#include<thread>
#include <mutex>          // std::mutex
#include <iostream>
#include <algorithm>
#include <set>
#include <utility>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "orbTrack/ORBextractor.h"

namespace feature_tracker
{

// 前后端接口数据,现在前后端都有该结构体,后面需要单独出来
typedef struct
{
    uint32_t id;
    float u0;
    float v0;
    float u1;
    float v1;
} Stereo_feature_t;

typedef struct
{
    uint64_t stamp;
    std::vector<Stereo_feature_t> features;
} Feature_measure_t;

/*
 * @brief ImageProcessor Detects and tracks features
 *    in image sequences.
 */
class ImageProcessor
{
public:
    // Constructor
    ImageProcessor();
    // Disable copy and assign constructors.
    ImageProcessor(const ImageProcessor &) = delete;
    ImageProcessor operator=(const ImageProcessor &) = delete;

    // Destructor
    ~ImageProcessor();

    void feed_measurement_imu(double timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

    // Initialize the object.
    bool initialize();

    /*
     * @brief featureUpdateCallback
     *    Publish the features on the current image including
     *    both the tracked and newly detected ones.
    */
    void featureUpdateCallback(std::deque<Feature_measure_t> &feature_buffer);

    typedef std::shared_ptr<ImageProcessor> Ptr;
    typedef std::shared_ptr<const ImageProcessor> ConstPtr;

    /*
   * @brief stereoCallback
   *    Callback function for the stereo images.
   * @param cam0_img left image.
   * @param cam1_img right image.
   */
    void stereoCallback(double timestamp,
                        const cv::Mat &cam0_img,
                        const cv::Mat &cam1_img);

    void DrawMatchPoints(const cv::Mat &src1, const cv::Mat &src2,
                         const std::vector<cv::Point2f> &kpt1,
                         const std::vector<cv::Point2f> &kpt2,
                         bool draw_line, cv::Mat &res_out);

private:
    cv::Mat curr_cam0_img;
    cv::Mat curr_cam1_img;

    double curr_cam_timestamp;
    double last_cam_timestamp;

    // IMU data buffer
    // This is buffer is used to handle the unsynchronization or
    // transfer delay between IMU and Image messages.
    struct StrImuData
    {
        double timestamp;
        Eigen::Vector3d acc;
        Eigen::Vector3d gyr;
    };
    typedef std::shared_ptr<StrImuData> PtrStrImuData;
    std::vector<PtrStrImuData> imu_msg_buffer;


    /*
   * @brief FeatureIDType An alias for unsigned long long int.
   */
    typedef unsigned long long int FeatureIDType;

    /*
   * @brief FeatureMetaData Contains necessary information
   *    of a feature for easy access.
   */
    struct FeatureMetaData
    {
        FeatureIDType id;
        float response;
        int lifetime;
        cv::Point2f cam0_point;
        cv::Point2f cam1_point;
    };

    int grid_row, grid_col;

    /*
   * @brief GridFeatures Organize features based on the grid
   *    they belong to. Note that the key is encoded by the
   *    grid index.
   */
    typedef std::map<int, std::vector<FeatureMetaData>> GridFeatures;

    //mutex imu
    std::mutex mutex_imu;

    // ID for the next new feature.
    FeatureIDType next_feature_id;

    // Indicate if this is the first image message.
    bool is_first_img;

    // Features in the previous and current image.
    std::shared_ptr<GridFeatures> prev_features_ptr;
    std::shared_ptr<GridFeatures> curr_features_ptr;

    // Camera calibration parameters
    cv::Vec4d cam0_intrinsics;
    cv::Vec4d cam0_distortion_coeffs;
    cv::Vec4d cam1_intrinsics;
    cv::Vec4d cam1_distortion_coeffs;

    // Take a vector from cam0 frame to the IMU frame.
    cv::Matx33d R_imu_from_cam0;
    cv::Vec3d t_imu_from_cam0;
    // Take a vector from cam1 frame to the IMU frame.
    cv::Matx33d R_imu_from_cam1;
    cv::Vec3d t_imu_from_cam1;

    // ORBSLAM2 部分参数
    ORB_SLAM2::ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;

    // 当前帧提取出来的作图的特征点数量
    int curr_left_size_fea;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<int> mvBestIdxR;
    std::vector<float> mvDepth;
    std::vector<std::pair<int, int> > vDistIdx;

    // 匹配的门限值
    const int TH_HIGH = 100;
    const int TH_LOW = 50;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // Stereo baseline multiplied by fx.
    float mbf;
    // Stereo baseline in meters.
    float baseline;

private:
    /*
    * @brief drawFeaturesStereo
    *    Draw tracked and newly detected features on the
    *    stereo images.
    */
    void drawFeaturesStereo();

    /*
    * @brief integrateImuData Integrates the IMU gyro readings
    *    between the two consecutive images, which is used for
    *    both tracking prediction and 2-point RANSAC.
    * @return cam0_R_p_c: a rotation matrix which takes a vector
    *    from previous cam0 frame to current cam0 frame.
    * @return cam1_R_p_c: a rotation matrix which takes a vector
    *    from previous cam1 frame to current cam1 frame.
    */
    void integrateImuData(cv::Matx33f& cam0_R_p_c, cv::Matx33f& cam1_R_p_c);

    // ORBSLAM2 部分函数
    void ExtractORB(int flag, const cv::Mat &im);

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    void initializeFirstFrame();
};

typedef ImageProcessor::Ptr ImageProcessorPtr;
typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;

} // end namespace msckf_vio

#endif
