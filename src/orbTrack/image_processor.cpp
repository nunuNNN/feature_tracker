#include <iostream>
#include <algorithm>
#include <set>
#include<thread>
#include <Eigen/Dense>

#include "orbTrack/image_processor.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace ORB_SLAM2;

namespace feature_tracker
{
ImageProcessor::ImageProcessor()
    : is_first_img(true),
      prev_features_ptr(new GridFeatures()),
      curr_features_ptr(new GridFeatures())
{
    initialize();

    int nFeatures = 200;
    float fScaleFactor = 1.2;
    int nLevels = 3;
    int fIniThFAST = 20;
    int fMinThFAST = 7;
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

}

ImageProcessor::~ImageProcessor()
{
    destroyAllWindows();
    return;
}

bool ImageProcessor::initialize()
{
    cam0_intrinsics[0] = 532.048;
    cam0_intrinsics[1] = 532.048;
    cam0_intrinsics[2] = 234.392;
    cam0_intrinsics[3] = 304.082;

    cam1_intrinsics[0] = 532.048;
    cam1_intrinsics[1] = 532.048;
    cam1_intrinsics[2] = 234.392;
    cam1_intrinsics[3] = 304.082;

    cam0_distortion_coeffs[0] = 0;
    cam0_distortion_coeffs[1] = 0;
    cam0_distortion_coeffs[2] = 0;
    cam0_distortion_coeffs[3] = 0;

    cam1_distortion_coeffs[0] = 0;
    cam1_distortion_coeffs[1] = 0;
    cam1_distortion_coeffs[2] = 0;
    cam1_distortion_coeffs[3] = 0;

    mbf = 31.390832;
    baseline = mbf / cam0_intrinsics[0]; // fx



    Eigen::Matrix3d R_b_c_forw;
    Eigen::Vector3d t_b_c_forw;
    R_b_c_forw << 0.0, 0.0, -1.0,
        -1.0, 0.0, 0.0,
        0.0, 1.0, 0.0;
    t_b_c_forw << 0.01809, 0.04454, 0.02205;
    Eigen::Isometry3d T_b_c1;
    T_b_c1.linear() = R_b_c_forw;
    T_b_c1.translation() = t_b_c_forw;

    Eigen::Matrix3d R_c0_c1_forw;
    Eigen::Vector3d t_c0_c1_forw;
    R_c0_c1_forw << 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0;
    t_c0_c1_forw << 0.059, 0, 0;
    Eigen::Isometry3d T_c0_c1;
    T_c0_c1.linear() = R_c0_c1_forw;
    T_c0_c1.translation() = t_c0_c1_forw;


    Eigen::Isometry3d T_b_c0 = T_b_c1 * T_c0_c1.inverse();
    Matrix3d R_b_c0 = T_b_c0.linear();
    Vector3d t_b_c0 = T_b_c0.translation();

    for (int n = 0; n < 3; ++n)
    {
        t_imu_from_cam1[n] = t_c0_c1_forw[n];
        t_imu_from_cam0[n] = t_b_c0[n];

        for (int m = 0; m < 3; ++m)
        {
            R_imu_from_cam1(n, m) = R_b_c_forw(n, m);
            R_imu_from_cam0(n, m) = R_b_c0(n, m);
        }
    }


    // Image Processor parameters
    grid_row = 4;
    grid_col = 5;


    printf("===========================================\n");
    printf("cam0_intrinscs: %f, %f, %f, %f\n",
           cam0_intrinsics[0], cam0_intrinsics[1],
           cam0_intrinsics[2], cam0_intrinsics[3]);
    printf("cam0_distortion_coefficients: %f, %f, %f, %f\n",
           cam0_distortion_coeffs[0], cam0_distortion_coeffs[1],
           cam0_distortion_coeffs[2], cam0_distortion_coeffs[3]);

    printf("cam1_intrinscs: %f, %f, %f, %f\n",
           cam1_intrinsics[0], cam1_intrinsics[1],
           cam1_intrinsics[2], cam1_intrinsics[3]);
    printf("cam1_distortion_coefficients: %f, %f, %f, %f\n",
           cam1_distortion_coeffs[0], cam1_distortion_coeffs[1],
           cam1_distortion_coeffs[2], cam1_distortion_coeffs[3]);

    cout << "R_imu_from_cam0: " << R_imu_from_cam0 << endl;
    cout << "t_imu_from_cam0: " << t_imu_from_cam0.t() << endl;

    printf("grid_row: %d\n", grid_row);
    printf("grid_col: %d\n", grid_col);
    printf("===========================================\n");

    return true;
}

void ImageProcessor::feed_measurement_imu(double timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
{
    PtrStrImuData ptr_str_imu_data(std::make_shared<StrImuData>());
    ptr_str_imu_data->timestamp = timestamp;
    ptr_str_imu_data->acc = acc;
    ptr_str_imu_data->gyr = gyr;

    mutex_imu.lock();
    imu_msg_buffer.push_back(ptr_str_imu_data);
    mutex_imu.unlock();

}

void ImageProcessor::stereoCallback(
    double timestamp,
    const Mat &cam0_img,
    const Mat &cam1_img)
{
    curr_cam_timestamp = timestamp;
    // cout << "curr_cam_timestamp: " << fixed << curr_cam_timestamp << endl;

    curr_cam0_img = cam0_img.clone();
    curr_cam1_img = cam1_img.clone();

    // ORB extraction
    thread threadLeft(&ImageProcessor::ExtractORB,this,0,curr_cam0_img);
    thread threadRight(&ImageProcessor::ExtractORB,this,1,curr_cam1_img);
    threadLeft.join();
    threadRight.join();

    curr_left_size_fea = mvKeys.size();
    
    if(mvKeys.empty())
        return;

    ComputeStereoMatches();

    
    
    if(curr_left_size_fea > 100)
    {
        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<curr_left_size_fea; i++)
        {
            float z = mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }

    drawFeaturesStereo();

    return;
}

void ImageProcessor::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void ImageProcessor::ComputeStereoMatches()
{
    mvuRight = vector<float>(curr_left_size_fea, -1.0f);
    mvDepth = vector<float>(curr_left_size_fea, -1.0f);

    const int thOrbDist = (TH_HIGH + TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = baseline;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(curr_left_size_fea);

    for(int iL=0; iL<curr_left_size_fea; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ImageProcessor::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void ImageProcessor::drawFeaturesStereo()
{
    // Colors for different features.
    Scalar tracked(0, 255, 0);
    Scalar new_feature(0, 255, 255);

    static int grid_height = curr_cam0_img.rows / grid_row;
    static int grid_width = curr_cam0_img.cols / grid_col;

    // Create an output image.
    int img_height = curr_cam0_img.rows;
    int img_width = curr_cam0_img.cols;
    Mat out_img(img_height, img_width * 2, CV_8UC3);
    cvtColor(curr_cam0_img, out_img.colRange(0, img_width), CV_GRAY2RGB);
    cvtColor(curr_cam1_img, out_img.colRange(img_width, img_width * 2), CV_GRAY2RGB);

    // Draw grids on the image.
    for (int i = 1; i < grid_row; ++i)
    {
        Point pt1(0, i * grid_height);
        Point pt2(img_width * 2, i * grid_height);
        line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }
    for (int i = 1; i < grid_col; ++i)
    {
        Point pt1(i * grid_width, 0);
        Point pt2(i * grid_width, img_height);
        line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }
    for (int i = 1; i < grid_col; ++i)
    {
        Point pt1(i * grid_width + img_width, 0);
        Point pt2(i * grid_width + img_width, img_height);
        line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }

    // Collect features ids in the previous frame.
    vector<FeatureIDType> prev_ids(0);
    for (const auto &grid_features : *prev_features_ptr)
    {
        for (const auto &feature : grid_features.second)
        {
            prev_ids.push_back(feature.id);
        }
    }
    // Collect feature points in the previous frame.
    map<FeatureIDType, Point2f> prev_cam0_points;
    map<FeatureIDType, Point2f> prev_cam1_points;
    for (const auto &grid_features : *prev_features_ptr)
    {
        for (const auto &feature : grid_features.second)
        {
            prev_cam0_points[feature.id] = feature.cam0_point;
            prev_cam1_points[feature.id] = feature.cam1_point;
        }
    }

    // Collect feature points in the current frame.
    map<FeatureIDType, Point2f> curr_cam0_points;
    map<FeatureIDType, Point2f> curr_cam1_points;
    for (const auto &grid_features : *curr_features_ptr)
    {
        for (const auto &feature : grid_features.second)
        {
            curr_cam0_points[feature.id] = feature.cam0_point;
            curr_cam1_points[feature.id] = feature.cam1_point;
        }
    }

    // Draw tracked features.
    for (const auto &id : prev_ids)
    {
        if (prev_cam0_points.find(id) != prev_cam0_points.end() &&
            curr_cam0_points.find(id) != curr_cam0_points.end())
        {
            cv::Point2f prev_pt0 = prev_cam0_points[id];
            cv::Point2f prev_pt1 = prev_cam1_points[id] + Point2f(img_width, 0.0);
            cv::Point2f curr_pt0 = curr_cam0_points[id];
            cv::Point2f curr_pt1 = curr_cam1_points[id] + Point2f(img_width, 0.0);

            circle(out_img, curr_pt0, 3, tracked, -1);
            circle(out_img, curr_pt1, 3, tracked, -1);
            line(out_img, prev_pt0, curr_pt0, tracked, 1);
            line(out_img, prev_pt1, curr_pt1, tracked, 1);

            prev_cam0_points.erase(id);
            prev_cam1_points.erase(id);
            curr_cam0_points.erase(id);
            curr_cam1_points.erase(id);
        }
    }

    // Draw new features.
    for (const auto &new_cam0_point : curr_cam0_points)
    {
        cv::Point2f pt0 = new_cam0_point.second;
        cv::Point2f pt1 = curr_cam1_points[new_cam0_point.first] + Point2f(img_width, 0.0);

        circle(out_img, pt0, 3, new_feature, -1);
        circle(out_img, pt1, 3, new_feature, -1);
    }
    imshow("Feature", out_img);
    waitKey(5);

    return;
}

void ImageProcessor::DrawMatchPoints(const cv::Mat &src1, const cv::Mat &src2,
                                     const std::vector<cv::Point2f> &kpt1,
                                     const std::vector<cv::Point2f> &kpt2,
                                     bool draw_line, cv::Mat &res_out)
{
    const int height = std::max(src1.rows, src2.rows);
    const int width = src1.cols + src2.cols;
    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Mat src1_rgb, src2_rgb;
    if (src1.channels() == 1)
    {
        cvtColor(src1, src1_rgb, CV_GRAY2RGB);
        cvtColor(src2, src2_rgb, CV_GRAY2RGB);
    }
    else if (src1.channels() == 3)
    {
        src1_rgb = src1;
        src2_rgb = src2;
    }
    else
    {
        return;
    }

    src1_rgb.copyTo(output(cv::Rect(0, 0, src1.cols, src1.rows)));
    src2_rgb.copyTo(output(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

    for (size_t i = 0; i < kpt1.size(); i++)
    {
        cv::Point2f left = kpt1[i];
        cv::Point2f right = (kpt2[i] + cv::Point2f((float)src1.cols, 0.f));
        unsigned char r = random();
        unsigned char g = random();
        unsigned char b = random();

        cv::circle(output, left, 2, cv::Scalar(r, g, b), 2);
        cv::circle(output, right, 2, cv::Scalar(r, g, b), 2);

        if (draw_line)
        {
            cv::line(output, left, right, cv::Scalar(r, g, b));
        }
    }

    res_out = output.clone();
}


} // end namespace msckf_vio
