#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <thread>

//opencv
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <include/tic_toc_ros.h>
#include <include/common.h>
#include <include/f2f_tracking.h>
#include <include/vi_type.h>
#include <include/rviz_frame.h>
#include <include/rviz_path.h>
#include <include/rviz_pose.h>
#include <include/rviz_odom.h>
#include <include/rviz_mesh.h>


#include <include/yamlRead.h>
#include <include/cv_draw.h>
#include <covis/KeyFrame.h>
#include <covis/CorrectionInf.h>
#include <include/correction_inf_msg.h>
#include <include/keyframe_msg_handler.h>
#include <tf/transform_listener.h>


namespace covis_ns
{


enum TYPEOFIMU{D435I,
               EuRoC_MAV,
               PIXHAWK,
               NONE};

class TrackingNodeletClass : public nodelet::Nodelet
{
public:
  TrackingNodeletClass()  {;}
  ~TrackingNodeletClass() {;}
private:
  bool is_lite_version;
  enum TYPEOFCAMERA cam_type;
  enum TYPEOFIMU imu_type;
  F2FTracking   *cam_tracker;
  DepthCamera dc;
  //Subscribers
  message_filters::Subscriber<sensor_msgs::Image> img0_sub;
  message_filters::Subscriber<sensor_msgs::Image> img1_sub;
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> MyExactSyncPolicy;
  message_filters::Synchronizer<MyExactSyncPolicy> * exactSync_;
  ros::Subscriber imu_sub;
  ros::Subscriber correction_inf_sub;

  //buffer and thread
  queue<sensor_msgs::ImageConstPtr> img0_buf;
  queue<sensor_msgs::ImageConstPtr> img1_buf;
  std::mutex m_img_buf;
  std::shared_ptr <std::thread> process_thread;

  //agent thread
  std::mutex m_agent_kf;
  queue<covis::KeyFrame> agent_keyframe_buf;
  std::shared_ptr <std::thread> kfmsgthread_;

  std::mutex m_agent_img;
  queue<cv_bridge::CvImagePtr> agent_img_buf;
  //Octomap
  //OctomapFeeder* octomap_pub;
  //Visualization
  cv::Mat img0_vis;
  cv::Mat img1_vis;
  image_transport::Publisher img0_pub;
  image_transport::Publisher img1_pub;

  RVIZFrame* frame_pub_agent;

  RVIZPath*  vision_path_pub;
  RVIZPath*  path_lc_pub;

  RVIZPose*  pose_imu_pub;
  RVIZOdom*  odom_imu_pub;
  RVIZPath*  path_imu_pub;

  KeyFrameMsgHandler* kf_Handler;
  RVIZMesh*    drone_pub;
  tf::StampedTransform tranOdomMap;
  tf::TransformListener listenerOdomMap;

  size_t AgentId_;
  string AgentFrameId;

  virtual void onInit()
  {
    ros::NodeHandle& nh = getMTPrivateNodeHandle();
    //cv::startWindowThread(); //Bug report https://github.com/ros-perception/image_pipeline/issues/201
    // Agent data
    int AgentId;
    nh.getParam("AgentId", AgentId);
    AgentId_ = static_cast<size_t>(AgentId);
    ROS_WARN("Agent %lu init " , AgentId_);
    nh.getParam("AgentFrameId", this->AgentFrameId);

    //Publisher
    frame_pub_agent = new RVIZFrame(nh,"/vo_camera_pose","/vo_curr_frame", AgentId_, AgentFrameId);
    vision_path_pub = new RVIZPath(nh,"/vision_path", AgentId_, AgentFrameId);    // pub vio path
    path_lc_pub     = new RVIZPath(nh,"/vision_path_lc", AgentId_, AgentFrameId);
    pose_imu_pub    = new RVIZPose(nh,"/imu_pose", AgentId_, AgentFrameId);   // pub imu pose
    odom_imu_pub    = new RVIZOdom(nh,"/imu_odom", AgentId_, AgentFrameId);   // pub imu odom
    path_imu_pub    = new RVIZPath(nh,"/imu_path", AgentId_, AgentFrameId);   // pub imu path
    drone_pub       = new RVIZMesh(nh, "/drone", AgentId_, AgentFrameId); // visualize by mesh
    kf_Handler      = new KeyFrameMsgHandler(nh, "/vo_kf", AgentId_);

    image_transport::ImageTransport it(nh);
    img0_pub = it.advertise("/Agent" + to_string(AgentId_) + "/vo_img0", 1);
    img1_pub = it.advertise("/Agent" + to_string(AgentId_) + "/vo_img1", 1);

    cam_tracker = new F2FTracking();

    //Load Parameter
    string configFilePath;
    nh.getParam("/yamlconfigfile",   configFilePath);
    ROS_WARN_STREAM("VO YAML configFilePath: " << configFilePath);

    auto severity = getIntVariableFromYaml(configFilePath, "LogLevel");
    if(severity == 0)
      ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);

    is_lite_version = false;
    is_lite_version = getBoolVariableFromYaml(configFilePath,"is_lite_version");
    if(is_lite_version){
      ROS_WARN("flvis run in lite version");
    }

    int vi_type_from_yaml = getIntVariableFromYaml(configFilePath,"type_of_vi");

    Vec3 vi_para1 = Vec3(getDoubleVariableFromYaml(configFilePath,"vifusion_para1"),
                         getDoubleVariableFromYaml(configFilePath,"vifusion_para2"),
                         getDoubleVariableFromYaml(configFilePath,"vifusion_para3"));
    Vec3 vi_para2 = Vec3(getDoubleVariableFromYaml(configFilePath,"vifusion_para4"),
                         getDoubleVariableFromYaml(configFilePath,"vifusion_para5"),
                         getDoubleVariableFromYaml(configFilePath,"vifusion_para6"));
    Vec6 vi_para;
    vi_para.head(3)=vi_para1;
    vi_para.tail(3)=vi_para2;


    Vec3 f_para1 = Vec3(getDoubleVariableFromYaml(configFilePath,"feature_para1"),//max features in a grid
                        getDoubleVariableFromYaml(configFilePath,"feature_para2"),//min features in a grid
                        getDoubleVariableFromYaml(configFilePath,"feature_para3"));//distance of features
    Vec3 f_para2 = Vec3(getDoubleVariableFromYaml(configFilePath,"feature_para4"),//goodFeaturesToTrack detector maxCorners
                        getDoubleVariableFromYaml(configFilePath,"feature_para5"),//goodFeaturesToTrack detector qualityLevel
                        getDoubleVariableFromYaml(configFilePath,"feature_para6"));//goodFeaturesToTrack detector minDistance
    Vec6 f_para;//feature related parameters
    f_para.head(3)=f_para1;
    f_para.tail(3)=f_para2;

    //depth recover parameters
    Vec3 dr_para = Vec3(getDoubleVariableFromYaml(configFilePath,"dr_para1"),//IIR ratio
                        getDoubleVariableFromYaml(configFilePath,"dr_para2"),//range
                        getDoubleVariableFromYaml(configFilePath,"dr_para3"));//enable dummy depth
    //keyframe selection parameters
    Vec3 kf_para = Vec3(getDoubleVariableFromYaml(configFilePath,"kf_para1"),//IIR ratio
                        getDoubleVariableFromYaml(configFilePath,"kf_para2"),//range
                        getDoubleVariableFromYaml(configFilePath,"kf_para3"));//enable dummy depth

    if(vi_type_from_yaml==VI_TYPE_D435I_DEPTH)        {cam_type=DEPTH_D435;    imu_type=D435I;}
    if(vi_type_from_yaml==VI_TYPE_EUROC_MAV)          {cam_type=STEREO_UNRECT; imu_type=EuRoC_MAV;}
    if(vi_type_from_yaml==VI_TYPE_D435_DEPTH_PIXHAWK) {cam_type=DEPTH_D435;    imu_type=PIXHAWK;}
    if(vi_type_from_yaml==VI_TYPE_D435I_STEREO)       {cam_type=STEREO_RECT;   imu_type=D435I;}
    if(vi_type_from_yaml==VI_TYPE_KITTI_STEREO)       {cam_type=STEREO_RECT;   imu_type=NONE;}
    if(vi_type_from_yaml==VI_TYPE_D435_STEREO_PIXHAWK) {cam_type=STEREO_RECT;  imu_type=PIXHAWK;}


    if(vi_type_from_yaml == VI_TYPE_D435I_DEPTH || vi_type_from_yaml == VI_TYPE_D435_DEPTH_PIXHAWK)
    {
      cout << "depth mode " << endl;
      int w = getIntVariableFromYaml(configFilePath,                    "image_width");
      int h = getIntVariableFromYaml(configFilePath,                    "image_height");
      cv::Mat K0_rect = cameraMatrixFromYamlIntrinsics(configFilePath,  "cam0_intrinsics");
      double depth_factor = getDoubleVariableFromYaml(configFilePath,   "depth_factor");
      Mat4x4 mat_imu_cam  = Mat44FromYaml(configFilePath,               "T_imu_cam0");

      dc.setDepthCamInfo(w,
                         h,
                         K0_rect.at<double>(0,0),//fx
                         K0_rect.at<double>(1,1),//fy
                         K0_rect.at<double>(0,2),//cx
                         K0_rect.at<double>(1,2),
                         depth_factor,
                         DEPTH_D435);
      cam_tracker->init(dc,
                        SE3(mat_imu_cam.topLeftCorner(3,3),mat_imu_cam.topRightCorner(3,1)),
                        f_para,
                        vi_para,
                        dr_para,
                        kf_para,
                        50,
                        false);
    }
    if(vi_type_from_yaml == VI_TYPE_D435I_STEREO || vi_type_from_yaml == VI_TYPE_D435_STEREO_PIXHAWK)
    {
      cout << "stereo mode " << endl;
      int w = getIntVariableFromYaml(configFilePath,             "image_width");
      int h = getIntVariableFromYaml(configFilePath,             "image_height");
      cv::Mat K0 = cameraMatrixFromYamlIntrinsics(configFilePath,"cam0_intrinsics");
      cv::Mat D0 = distCoeffsFromYaml(configFilePath,            "cam0_distortion_coeffs");
      cv::Mat K1 = cameraMatrixFromYamlIntrinsics(configFilePath,"cam1_intrinsics");
      cv::Mat D1 = distCoeffsFromYaml(configFilePath,            "cam1_distortion_coeffs");
      Mat4x4  mat_imu_cam  = Mat44FromYaml(configFilePath,       "T_imu_cam0");
      Mat4x4  mat_cam0_cam1  = Mat44FromYaml(configFilePath,     "T_cam0_cam1");
      SE3 T_i_c0 = SE3(mat_imu_cam.topLeftCorner(3,3),
                       mat_imu_cam.topRightCorner(3,1));
      SE3 T_c0_c1 = SE3(mat_cam0_cam1.topLeftCorner(3,3),
                        mat_cam0_cam1.topRightCorner(3,1));
      SE3 T_c1_c0 = T_c0_c1.inverse();

      Mat3x3 R_ = T_c1_c0.rotation_matrix();
      Vec3   T_ = T_c1_c0.translation();
      cv::Mat R__ = (cv::Mat1d(3, 3) <<
                     R_(0,0), R_(0,1), R_(0,2),
                     R_(1,0), R_(1,1), R_(1,2),
                     R_(2,0), R_(2,1), R_(2,2));
      cv::Mat T__ = (cv::Mat1d(3, 1) << T_(0), T_(1), T_(2));
      cv::Mat R0,R1,P0,P1,Q;
      cv::stereoRectify(K0,D0,K1,D1,cv::Size(w,h),R__,T__,
                        R0,R1,P0,P1,Q,
                        CALIB_ZERO_DISPARITY,0,cv::Size(w,h));
      cv::Mat K0_rect = P0.rowRange(0,3).colRange(0,3);
      cv::Mat K1_rect = P1.rowRange(0,3).colRange(0,3);
      cv::Mat D0_rect,D1_rect;
      D1_rect = D0_rect = (cv::Mat1d(4, 1) << 0,0,0,0);

      dc.setSteroCamInfo(w,h,
                         K0, D0, K0_rect, D0_rect, R0, P0,
                         K1, D1, K1_rect, D1_rect, R1, P1,
                         T_c0_c1,STEREO_RECT);
      cam_tracker->init(dc,
                        T_i_c0,
                        f_para,
                        vi_para,
                        dr_para,
                        kf_para,
                        50,
                        false);

    }
    if(vi_type_from_yaml == VI_TYPE_EUROC_MAV)
    {
      cout << "EUROC test " << endl;
      int w = getIntVariableFromYaml(configFilePath,             "image_width");
      int h = getIntVariableFromYaml(configFilePath,             "image_height");
      cv::Mat K0 = cameraMatrixFromYamlIntrinsics(configFilePath,"cam0_intrinsics");
      cv::Mat D0 = distCoeffsFromYaml(configFilePath,            "cam0_distortion_coeffs");
      cv::Mat K1 = cameraMatrixFromYamlIntrinsics(configFilePath,"cam1_intrinsics");
      cv::Mat D1 = distCoeffsFromYaml(configFilePath,            "cam1_distortion_coeffs");
      Mat4x4  mat_mavimu_cam0  = Mat44FromYaml(configFilePath,   "T_mavimu_cam0");
      Mat4x4  mat_mavimu_cam1  = Mat44FromYaml(configFilePath,   "T_mavimu_cam1");
      Mat4x4  mat_i_mavimu  = Mat44FromYaml(configFilePath,      "T_imu_mavimu");
      SE3 T_mavi_c0 = SE3(mat_mavimu_cam0.topLeftCorner(3,3),
                          mat_mavimu_cam0.topRightCorner(3,1));
      SE3 T_mavi_c1 = SE3(mat_mavimu_cam1.topLeftCorner(3,3),
                          mat_mavimu_cam1.topRightCorner(3,1));
      SE3 T_c0_c1 = T_mavi_c0.inverse()*T_mavi_c1;
      SE3 T_c1_c0 = T_c0_c1.inverse();
      SE3 T_i_mavi = SE3(mat_i_mavimu.topLeftCorner(3,3),mat_i_mavimu.topRightCorner(3,1));
      SE3 T_i_c0 = T_i_mavi*T_mavi_c0;
      //cout << "T_i_c0: " << T_i_c0 << endl;
      Mat3x3 R_ = T_c1_c0.rotation_matrix();
      Vec3   T_ = T_c1_c0.translation();
      cv::Mat R__ = (cv::Mat1d(3, 3) <<
                     R_(0,0), R_(0,1), R_(0,2),
                     R_(1,0), R_(1,1), R_(1,2),
                     R_(2,0), R_(2,1), R_(2,2));
      cv::Mat T__ = (cv::Mat1d(3, 1) << T_(0), T_(1), T_(2));
      cv::Mat R0,R1,P0,P1,Q;
      cv::stereoRectify(K0,D0,K1,D1,cv::Size(w,h),R__,T__,
                        R0,R1,P0,P1,Q,
                        CALIB_ZERO_DISPARITY,0,cv::Size(w,h));
      cv::Mat K0_rect = P0.rowRange(0,3).colRange(0,3);
      cv::Mat K1_rect = P1.rowRange(0,3).colRange(0,3);
      cv::Mat D0_rect,D1_rect;
      D1_rect = D0_rect = (cv::Mat1d(4, 1) << 0,0,0,0);

      dc.setSteroCamInfo(w,h,
                         K0, D0, K0_rect, D0_rect, R0, P0,
                         K1, D1, K1_rect, D1_rect, R1, P1,
                         T_c0_c1,STEREO_UNRECT);
      cam_tracker->init(dc,
                        T_i_c0,
                        f_para,
                        vi_para,
                        dr_para,
                        kf_para,
                        0,
                        true);
    }
    if(vi_type_from_yaml == VI_TYPE_KITTI_STEREO)
    {
      cout << "KITTI test " << endl;
      int w = getIntVariableFromYaml(configFilePath,             "image_width");
      int h = getIntVariableFromYaml(configFilePath,             "image_height");
      Mat4x4  P0_ = Mat44FromYaml(configFilePath,"cam0_projection_matrix");
      Mat4x4  P1_ = Mat44FromYaml(configFilePath,"cam1_projection_matrix");
      Mat4x4  K_inverse;
      K_inverse.fill(0);
      Mat3x3 K = P0_.topLeftCorner(3,3);
      K_inverse.topLeftCorner(3,3) = K.inverse();
      Mat4x4 mat_T_c0_c1 = K_inverse*P1_;
      mat_T_c0_c1.topLeftCorner(3,3).setIdentity();
      SE3 T_c0_c1(mat_T_c0_c1.topLeftCorner(3,3),mat_T_c0_c1.topRightCorner(3,1));

      cv::Mat P0 = (cv::Mat1d(3, 4) <<
            P0_(0,0), P0_(0,1), P0_(0,2), P0_(0,3),
            P0_(1,0), P0_(1,1), P0_(1,2), P0_(1,3),
            P0_(2,0), P0_(2,1), P0_(2,2), P0_(2,3));
      cv::Mat P1 = (cv::Mat1d(3, 4) <<
            P1_(0,0), P1_(0,1), P1_(0,2), P1_(0,3),
            P1_(1,0), P1_(1,1), P1_(1,2), P1_(1,3),
            P1_(2,0), P1_(2,1), P1_(2,2), P1_(2,3));
      cv::Mat K0,K1,K0_rect,K1_rect;
      cv::Mat D0,D1,D0_rect,D1_rect;
      D1_rect = D0_rect = D1 = D0 =(cv::Mat1d(4, 1) << 0,0,0,0);
      K0 = K1 = K0_rect = P0.rowRange(0,3).colRange(0,3);
      K1_rect = P1.rowRange(0,3).colRange(0,3);

      dc.setSteroCamInfo(w,h,
                         K0, D0, K0_rect, D0_rect, (cv::Mat1d(3, 3) << 1,0,0,0,1,0,0,0,1), P0,
                         K1, D1, K1_rect, D1_rect, (cv::Mat1d(3, 3) << 1,0,0,0,1,0,0,0,1), P1,
                         T_c0_c1,STEREO_RECT);
      cam_tracker->init(dc,
                        SE3(),//dummy parameter
                        f_para,
                        vi_para,
                        dr_para,
                        kf_para,
                        0,
                        false);

    }

    img0_sub.subscribe(nh, "/vo/input_image_0", 3);
    img1_sub.subscribe(nh, "/vo/input_image_1", 3);

    imu_sub = nh.subscribe<sensor_msgs::Imu>(
          "/imu",
          10,
          boost::bind(&TrackingNodeletClass::imu_callback, this, _1));
    exactSync_ = new message_filters::Synchronizer<MyExactSyncPolicy>(MyExactSyncPolicy(2), img0_sub, img1_sub);
    exactSync_->registerCallback(boost::bind(&TrackingNodeletClass::image_input_callback, this, _1, _2));

    process_thread = std::shared_ptr<std::thread>
            (new std::thread(&TrackingNodeletClass::process, this));
    kfmsgthread_ = std::shared_ptr<std::thread>
            (new std::thread(&TrackingNodeletClass::Agentkfprocess, this));


    ROS_WARN("VIO %lu start tracking thread", AgentId_);
    //    correction_inf_sub = nh.subscribe<covis::CorrectionInf>(
    //          "/vo_localmap_feedback",
    //          2,
    //          boost::bind(&TrackingNodeletClass::correction_feedback_callback, this, _1));

  }


  void imu_callback(const sensor_msgs::ImuConstPtr& msg)
  {
    //std::thread::id this_id = std::this_thread::get_id();
    //std::cout << "thread " << this_id << " imu callback ...\n";
    //SETP1: TO ENU Frame
    Vec3 acc,gyro;
    ros::Time tstamp = msg->header.stamp;
    //ROS_WARN("Agent %lu, IMU time %lf ", AgentId_, msg->header.stamp.toSec());
    if(imu_type==D435I)
    {
      acc = Vec3(-msg->linear_acceleration.z,
                 msg->linear_acceleration.x,
                 msg->linear_acceleration.y);
      gyro = Vec3(msg->angular_velocity.z,
                  -msg->angular_velocity.x,
                  -msg->angular_velocity.y);
    }
    if(imu_type==EuRoC_MAV)
    {
      gyro = Vec3(msg->angular_velocity.z,
                  -msg->angular_velocity.y,
                  msg->angular_velocity.x);
      acc = Vec3(-msg->linear_acceleration.z,
                 msg->linear_acceleration.y,
                 -msg->linear_acceleration.x);
    }
    if(imu_type==PIXHAWK)
    {
      acc = Vec3(-msg->linear_acceleration.x,
                 -msg->linear_acceleration.y,
                 -msg->linear_acceleration.z);
      gyro = Vec3(msg->angular_velocity.x,
                  msg->angular_velocity.y,
                  msg->angular_velocity.z);
    }



    //nur for test
    //acc = Vec3(4.55,0.3,-7.88);
    //gyro =  Vec3(0.001,0.002,0.003);
    Quaterniond q_w_i;
    Vec3        pos_w_i, vel_w_i;
    cam_tracker->imu_feed(tstamp.toSec(),acc,gyro,
                          q_w_i,pos_w_i,vel_w_i);
    pose_imu_pub->pubPose(q_w_i,pos_w_i,tstamp);
    odom_imu_pub->pubOdom(q_w_i,pos_w_i,vel_w_i,tstamp);
    path_imu_pub->pubPathT_w_c(SE3(q_w_i,pos_w_i),tstamp, AgentId_);
    drone_pub->PubT_w_i(SE3(q_w_i,pos_w_i), tstamp, AgentId_);
  }
/*
  void correction_feedback_callback(const covis::CorrectionInf::ConstPtr& msg)
  {
    //unpacking and update the structure
    CorrectionInfStruct correction_inf;
    CorrectionInfMsg::unpack(msg,
                             correction_inf.frame_id,
                             correction_inf.T_c_w,
                             correction_inf.lm_count,
                             correction_inf.lm_id,
                             correction_inf.lm_3d,
                             correction_inf.lm_outlier_count,
                             correction_inf.lm_outlier_id);
  }
*/
  void image_input_callback(const sensor_msgs::ImageConstPtr & img0_Ptr,
                            const sensor_msgs::ImageConstPtr & img1_Ptr)
  {

    //ROS_WARN("Agent %lu, img0 time %lf ", AgentId_, img0_Ptr->header.stamp.toSec());
    //ROS_WARN("Agent %lu, img1 time %lf ", AgentId_, img0_Ptr->header.stamp.toSec());

    m_img_buf.lock();
    //std::thread::id this_id = std::this_thread::get_id();
    //std::cout << "thread " << this_id << " camera callback ...\n";
    img0_buf.push(img0_Ptr);
    img1_buf.push(img1_Ptr);
    m_img_buf.unlock();

  }
// second thread to sync img msg
  void process()
  {
      while(true)
      {
          m_img_buf.lock();
          sensor_msgs::ImageConstPtr img0_Ptr;
          sensor_msgs::ImageConstPtr img1_Ptr;
          if(!img0_buf.empty() && !img1_buf.empty())
          {
              img0_Ptr = img0_buf.front();
              img0_buf.pop();
              img1_Ptr = img1_buf.front();
              img1_buf.pop();

          }
          m_img_buf.unlock();
          if(img0_Ptr && img1_Ptr)
          {
              //tic_toc_ros tt_cb;
              static int count=1;
              count ++;
              auto seq = img0_Ptr->header.seq;
              //ROS_DEBUG("VO receive img data: %d ", count++);
              //ROS_DEBUG("VO receive img data seq: %d ", seq);
              ros::Time tstamp = img0_Ptr->header.stamp;
              cv_bridge::CvImagePtr cvbridge_img0  = cv_bridge::toCvCopy(img0_Ptr, img0_Ptr->encoding);
              cv_bridge::CvImagePtr cvbridge_img1  = cv_bridge::toCvCopy(img1_Ptr, img1_Ptr->encoding);
              bool newkf;//new key frame
              bool reset_cmd;//reset command to localmap node
              this->cam_tracker->image_feed(tstamp.toSec(),
                                            cvbridge_img0->image,
                                            cvbridge_img1->image,
                                            newkf,
                                            reset_cmd);

              if(newkf)
              {
                covis::KeyFrame agent_keyframe_msg;
                kf_Handler->preProcessKeyFramemsg(cam_tracker->curr_frame, cam_tracker->vimotion, agent_keyframe_msg, tstamp);
                m_agent_kf.lock();
                //printf("push kf msg to buf \n");
                agent_keyframe_buf.push(agent_keyframe_msg);
                m_agent_kf.unlock();

                m_agent_img.lock();
                agent_img_buf.push(cvbridge_img0);
                m_agent_img.unlock();
              }

              //if(reset_cmd) kf_Handler->cmdLMResetPub(ros::Time(tstamp));
              frame_pub_agent->pubFramePtsPoseT_c_w(this->cam_tracker->curr_frame->getValid3dPts(),
                                              this->cam_tracker->curr_frame->T_c_w,
                                              tstamp);
              vision_path_pub->pubPathT_c_w(this->cam_tracker->curr_frame->T_c_w, tstamp, AgentId_);


              SE3 T_map_c =SE3();
              try{
                listenerOdomMap.lookupTransform("map","odom",ros::Time(0), tranOdomMap);
                tf::Vector3 tf_t= tranOdomMap.getOrigin();
                tf::Quaternion tf_q = tranOdomMap.getRotation();
                SE3 T_map_odom(Quaterniond(tf_q.w(),tf_q.x(),tf_q.y(),tf_q.z()),
                               Vec3(tf_t.x(),tf_t.y(),tf_t.z()));
                T_map_c = T_map_odom*this->cam_tracker->curr_frame->T_c_w.inverse();
                path_lc_pub->pubPathT_w_c(T_map_c,tstamp, AgentId_);
              }
              catch (tf::TransformException ex)
              {
                //cout<<"no transform between map and odom yet."<<endl;
              }
              if(!is_lite_version)
              {
                cvtColor(cam_tracker->curr_frame->img0,img0_vis,CV_GRAY2BGR);
                if(cam_type==DEPTH_D435)
                {
                  drawFrame(img0_vis,*this->cam_tracker->curr_frame,1,6);
                  visualizeDepthImg(img1_vis,*this->cam_tracker->curr_frame);
                }
                if(cam_type==STEREO_UNRECT || cam_type==STEREO_RECT)
                {
                  drawFrame(img0_vis,*this->cam_tracker->curr_frame,1,11);
                  cvtColor(cam_tracker->curr_frame->img1,img1_vis,CV_GRAY2BGR);
                  drawFlow(img0_vis,
                           this->cam_tracker->curr_frame->flow_last,
                           this->cam_tracker->curr_frame->flow_curr);
                  drawFlow(img1_vis,
                           this->cam_tracker->curr_frame->flow_0,
                           this->cam_tracker->curr_frame->flow_1);
                }
                sensor_msgs::ImagePtr img0_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img0_vis).toImageMsg();
                sensor_msgs::ImagePtr img1_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img1_vis).toImageMsg();
                img0_pub.publish(img0_msg);
                img1_pub.publish(img1_msg);
              }


          }
          std::this_thread::sleep_for(std::chrono::milliseconds(2));

      }

  }

  void Agentkfprocess()
  {
    while(1)
    {
      covis::KeyFrame tmp_msg;
      bool pub_flag = false;
      m_agent_kf.lock();
      if(!agent_keyframe_buf.empty())
      {
          tmp_msg = agent_keyframe_buf.front();
          agent_keyframe_buf.pop();
          pub_flag = true;
          //printf("pop kf msg from buf, buff size: %lu \n", agent_keyframe_buf.size());
      }
      m_agent_kf.unlock();
      if(pub_flag)
      {
        cv::Mat img;
        m_agent_img.lock();
        if(!agent_img_buf.empty())
        {
//           cout << setprecision(20) << endl;
//           cout << "img ros time: " << agent_img_buf.front()->header.stamp.toSec() << endl;
//           cout << "kf ros time: " << tmp_msg.header.stamp.toSec() << endl;
           img = agent_img_buf.front()->image;
           agent_img_buf.pop();
//           cv::imshow("img0 ", agent_img_buf.back()->image);
//           cv::waitKey(20);
        }

        m_agent_img.unlock();
        if(!img.empty())
        {
          kf_Handler->ProcessKeyFramemsg(tmp_msg, img);

        }
        else {
          printf("Warning: img0 empty \n");
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(5));

    }

  }


};//class TrackingNodeletClass
}//namespace covis_ns

PLUGINLIB_EXPORT_CLASS(covis_ns::TrackingNodeletClass, nodelet::Nodelet)


