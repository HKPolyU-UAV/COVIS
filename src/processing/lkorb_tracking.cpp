#include "include/lkorb_tracking.h"

LKORBTracking::LKORBTracking(int width_in,int height_in)
{  
    this->width=width_in;
    this->height=height_in;
}
/** @brief Tracking Lucas-Kanade Optical flow and calculate camera pose by Perspective-n-Point.
 * Either the number of LK tracking inliers, Fundamental matrix check inliers or Ransac inliers less than 10 will output false.
 *
@param from Previous frame. We get 2D and 3D landmarks from it.
@param to Current frame. We calculate LK flow and update landmarks in it.
@param T_c_w_guess Has IMU guess
@param use_guess Has IMU guess and use IMU guess
@param lm2d_from flow_last of current frame. Store successful LK tracking, F check last frame's 2d landmarks.
@param lm2d_to flow_to of current frame. Store successful LK tracking, F check current frame's 2d landmarks.
@param outlier tracking_outlier of current frame. Store either failed LK tracking or F check 2d landmarks.

*/
bool LKORBTracking::tracking(CameraFrame& from,
                             CameraFrame& to,
                             SE3 T_c_w_guess,
                             bool use_guess,
                             vector<cv::Point2f>& lm2d_from,
                             vector<cv::Point2f>& lm2d_to,
                             vector<cv::Point2f>& outlier)
{
    lm2d_from.clear();
    lm2d_to.clear();
    outlier.clear();
    //STEP1: Optical Flow
    //STEP2: F matrix check
    //STEP3: PNP_RANSAC
    bool ret = false;

    //STEP1
    vector<int>         in_frame_idx;
    vector<cv::Point2f> from_p2d_plane;       /// lm_2d_plane
    vector<cv::Point2f> tracked_p2d_plane;    /// output vector of 2D points (with single-precision floating-point coordinates)
    /// containing the calculated new positions of input features in the second image
    vector<cv::Point2f> from_p2d_undistort;   /// lm_2d_undistort
    vector<cv::Point2f> tracked_p2d_undistort;
    vector<cv::Point3f> from_p3d;             /// lm_3d_w
    from.getAll2dPlaneUndistort3d_cvPf(from_p2d_plane,from_p2d_undistort,from_p3d);
    tracked_p2d_plane = from_p2d_plane;


    vector<unsigned char> mask_tracked;
    vector<float>         err;

    if(use_guess){//project 3d lms to 2d using guess
        switch(d_camera.cam_type)
        {
        case DEPTH_D435:
            for(size_t i=0; i<from_p2d_plane.size(); i++){
                Vec3 lm3d_w(from_p3d.at(i).x,from_p3d.at(i).y,from_p3d.at(i).z);
                Vec3 lm3d_c = DepthCamera::world2cameraT_c_w(lm3d_w,T_c_w_guess);
                Vec2 reProj=from.d_camera.camera2pixel(lm3d_c,
                                                       from.d_camera.cam0_fx,
                                                       from.d_camera.cam0_fy,
                                                       from.d_camera.cam0_cx,
                                                       from.d_camera.cam0_cy);
                tracked_p2d_plane.at(i) = cv::Point2f(reProj[0],reProj[1]);
            }
            break;
        case STEREO_RECT:
        case STEREO_UNRECT:
            vector<cv::Point2f> project_to_to_img0_plane;
            cv::Mat r_,t_;
            SE3_to_rvec_tvec(T_c_w_guess,r_,t_);
            cv::projectPoints(from_p3d,r_,t_,d_camera.K0,d_camera.D0,project_to_to_img0_plane);
            for(size_t i=0; i<from_p2d_plane.size(); i++){
                tracked_p2d_plane.at(i) = project_to_to_img0_plane.at(i);
            }
            break;
        }

        cv::calcOpticalFlowPyrLK(from.img0, to.img0, from_p2d_plane, tracked_p2d_plane,
                                 mask_tracked, err, cv::Size(31,31), 10,
                                 cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 30, 0.001),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);

    }else
    {

        cv::calcOpticalFlowPyrLK(from.img0, to.img0, from_p2d_plane, tracked_p2d_plane,
                                 mask_tracked, err, cv::Size(31,31), 10,
                                 cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 30, 0.001),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);

    }

    switch(d_camera.cam_type)
    {
    case DEPTH_D435:
        from_p2d_undistort = from_p2d_plane;
        tracked_p2d_undistort = tracked_p2d_plane;
        break;
    case STEREO_RECT:
        from_p2d_undistort = from_p2d_plane;
        tracked_p2d_undistort = tracked_p2d_plane;
        break;
    case STEREO_UNRECT:
        cv::undistortPoints(tracked_p2d_plane,tracked_p2d_undistort,
                            d_camera.K0,d_camera.D0,d_camera.R0,d_camera.P0);
        break;
    }

    //Creat new frame with all successful Optical Flow result
    to.landmarks.clear();
    int w = (to.d_camera.img_w-1);
    int h = (to.d_camera.img_h-1);
    int of_inlier_cnt=0;
    for(int i=from.landmarks.size()-1; i>=0; i--)
    {
        if(mask_tracked.at(i)==1
                &&tracked_p2d_plane.at(i).x>0 && tracked_p2d_plane.at(i).y>0
                &&tracked_p2d_plane.at(i).x<w && tracked_p2d_plane.at(i).y<h)
        {
            of_inlier_cnt++;
            LandMarkInFrame lm=from.landmarks.at(i);
            /// update lm_2d_plane and lm_2d_undistort, other info lm_3d_w and lm_3d_c do not change
            lm.lm_2d_plane=Vec2(tracked_p2d_plane.at(i).x,tracked_p2d_plane.at(i).y);
            lm.lm_2d_undistort=Vec2(tracked_p2d_undistort.at(i).x,tracked_p2d_undistort.at(i).y);
            to.landmarks.push_back(lm);
        }
        else
        {
            outlier.push_back(from_p2d_plane.at(i));
            from_p2d_plane.erase(from_p2d_plane.begin()+i);           
            tracked_p2d_plane.erase(tracked_p2d_plane.begin()+i);
            from_p2d_undistort.erase(from_p2d_undistort.begin()+i);
            tracked_p2d_undistort.erase(tracked_p2d_undistort.begin()+i);
            from_p3d.erase(from_p3d.begin()+i);
        }
    }
    if(of_inlier_cnt < 10)
    {
        ret = false;
        return ret;
    }

    //The date below are aligned:
    //to.landmarks
    //from_cvP2f
    //from_cvP3f
    //tracked_cvP2f

    //STEP2 F matrix check
    vector<unsigned char> mask_F_consistant;
    cv::findFundamentalMat(from_p2d_undistort, tracked_p2d_undistort,
                           cv::FM_RANSAC, 5.0, 0.99, mask_F_consistant);


    //F inconsistance point are mark as outlier
    int F_inlier_cnt=0;
    for(int i = 0; i < mask_F_consistant.size(); i++)
    {
        if(mask_F_consistant.at(i)==0)
        {

            outlier.push_back(from_p2d_plane.at(i));     ///curr_frame->tracking_outlier
            to.landmarks.at(i).is_tracking_inlier=false; /// curr_frame.landmarks: every landmark is_tracking_inlier is true by default.
        }else
        {

            lm2d_from.push_back(from_p2d_plane.at(i));  ///curr_frame->flow_last Every CameraFrame object has two vector<cv::Point2f> to store keypoints
            lm2d_to.push_back(tracked_p2d_plane.at(i)); ///curr_frame->flow_curr
        }
    }

    for(LandMarkInFrame lm : to.landmarks)
    {
        if(lm.is_tracking_inlier==true)
            F_inlier_cnt++;
    }
    if(F_inlier_cnt<10){
        ret=false;
        return ret;
    }
    //STEP 3
    vector<unsigned char> mask_pnp_ransac;
    cv::Mat r_ = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat t_ = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat D;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat inliers;
    vector<cv::Point2f> p2d_un;
    vector<cv::Point3f> p3d;
#if 0
        vector<cv::Point2f> p2d_norm;
        vector<cv::Point2f> p2d_norm_un;
        to.get2Norm_3d(p2d_norm, p3d);
        cv::undistortPoints(p2d_norm, p2d_norm_un,
                            d_camera.K0,d_camera.D0);
        for(int i = 0; i < p3d.size(); i++)
        {
           cout <<  "p2d: " << p2d_norm[i] << " | p2d_norm_un: " << p2d_norm_un[i] <<
                   " | p3d: " << p3d[i] << endl;
        }
#endif
    to.get2dUndistort3dInlierPair_cvPf(p2d_un, p3d);  ///LK tracking successful and Fundamental matrix check consistent landmark

    if(use_guess)
    {
      SE3_to_rvec_tvec(T_c_w_guess, r_ , t_ );

      cv::solvePnPRansac(p3d,p2d_un,this->d_camera.K0_rect,this->d_camera.D0_rect,
                                r_,t_,false, 100, 3.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    }
    else
    {
     cv::solvePnPRansac(p3d,p2d_un,this->d_camera.K0_rect,this->d_camera.D0_rect,
                               r_,t_,false, 100, 3.0, 0.99, inliers, cv::SOLVEPNP_P3P);
    }

    ///inlier masks. Transfer from cv::Mat inliers to vector<uchar> mask_pnp_ransac
    int pnp_inlier_cnt=0;
    for (int i = 0; i < (int)p2d_un.size(); i++)
    {
        mask_pnp_ransac.push_back(0);
    }

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        mask_pnp_ransac[n] = 1;
        pnp_inlier_cnt++;
    }

    to.updateLMState(mask_pnp_ransac);
    to.T_c_w = SE3_from_rvec_tvec(r_,t_);
    ROS_DEBUG_STREAM("tracking: " << pnp_inlier_cnt << "|" << F_inlier_cnt << "|" << of_inlier_cnt );

    if(inliers.rows<10)
    {
        ret = false;
        return ret;
    }else
    {
        ret = true;
        return ret;
    }

}
