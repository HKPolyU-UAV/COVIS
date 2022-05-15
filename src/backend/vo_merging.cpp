//ROS
#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>

#include "include/merging.h"

namespace covis_ns
{

class MergingNodeletClass : public nodelet::Nodelet
{
public:
  MergingNodeletClass()  {;}
  ~MergingNodeletClass() {;}

private:
  ros::Subscriber kf_sub[10];

  Merging * merger;
  DepthCamera dc;
  LC_PARAS lc_paras;
  //DBow related para
  Vocabulary voc;
  Database db;// faster search
  int ClientNum = 1;
  bool IntraLoop = true;
  std::shared_ptr <std::thread> kfmsgthread_;
  queue<covis::KeyFrameConstPtr> msg_queue;
  std::mutex m_buf;


  void merge_callback(const covis::KeyFrameConstPtr& msg)
  {
    static int count = 0;
    m_buf.lock();
    ROS_DEBUG("Server received: %d ", count++);
    msg_queue.push(msg);
    m_buf.unlock();
  }

  void kfmsgProcess()
  {
    while(true)
    {
      covis::KeyFrameConstPtr msg;
      m_buf.lock();
      if(!msg_queue.empty())
      {
        msg = msg_queue.front();
        msg_queue.pop();
      }
      m_buf.unlock();

      if(msg)
      {
        merger->setKeyFrameMerge(msg);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  virtual void onInit()
  {

    ros::NodeHandle nh = getPrivateNodeHandle();

    // Server data
    string configFilePath;
    nh.getParam("/yamlconfigfile", configFilePath);
    nh.getParam("/ClientNum", ClientNum);
    ROS_WARN_STREAM("Server YAML configFilePath: " << configFilePath);
    lc_paras.lcKFStart    = getIntVariableFromYaml(configFilePath,"lcKFStart");
    lc_paras.lcKFDist     = getIntVariableFromYaml(configFilePath,"lcKFDist");
    lc_paras.lcKFMaxDist  = getIntVariableFromYaml(configFilePath,"lcKFMaxDist");
    lc_paras.lcKFLast     = getIntVariableFromYaml(configFilePath,"lcKFLast");
    lc_paras.lcNKFClosest = getIntVariableFromYaml(configFilePath,"lcNKFClosest");
    lc_paras.ratioMax     = getDoubleVariableFromYaml(configFilePath,"ratioMax");
    lc_paras.ratioRansac  = getDoubleVariableFromYaml(configFilePath,"ratioRansac");
    lc_paras.minPts       = getIntVariableFromYaml(configFilePath,"minPts");
    lc_paras.minScore     = getDoubleVariableFromYaml(configFilePath,"minScore");
    int saveImage         = getIntVariableFromYaml(configFilePath, "saveImage");
    IntraLoop             = getBoolVariableFromYaml(configFilePath, "IntraLoop");
    string path           = getstringVariableFromYaml(configFilePath, "ResultPath");
    cout << path << endl;
    std::vector<string> ResultPaths;
    for (int i = 0; i< 3; i++)
    {
      ResultPaths.push_back(path +  + "loop_" + to_string(i) + "_path.txt");
    }
    lc_paras.ResultPaths = ResultPaths;
    for(auto p:ResultPaths)
      cout << p << endl;

    auto severity = getIntVariableFromYaml(configFilePath, "LogLevel");
    if(severity == 0)
      ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);




    //        #define VI_TYPE_D435I_DEPTH        (0)
    //        #define VI_TYPE_EUROC_MAV          (1)
    //        #define VI_TYPE_D435_DEPTH_PIXHAWK (2)
    //        #define VI_TYPE_D435I_STEREO       (3)
    //        #define VI_TYPE_KITTI_STEREO       (4)
    int vi_type_from_yaml = getIntVariableFromYaml(configFilePath,"type_of_vi");

    if(vi_type_from_yaml == VI_TYPE_D435I_DEPTH || vi_type_from_yaml == VI_TYPE_D435_DEPTH_PIXHAWK)
    {
      ROS_WARN("D435 DEPTH test");
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
    }
    if(vi_type_from_yaml == VI_TYPE_D435I_STEREO)
    {
      ROS_WARN("D435 stereo test");
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
                        cv::CALIB_ZERO_DISPARITY,0,cv::Size(w,h));
      cv::Mat K0_rect = P0.rowRange(0,3).colRange(0,3);
      cv::Mat K1_rect = P1.rowRange(0,3).colRange(0,3);
      cv::Mat D0_rect,D1_rect;
      D1_rect = D0_rect = (cv::Mat1d(4, 1) << 0,0,0,0);
      dc.setSteroCamInfo(w,h,
                         K0, D0, K0_rect, D0_rect, R0, P0,
                         K1, D1, K1_rect, D1_rect, R1, P1,
                         T_c0_c1,STEREO_RECT);
    }
    if(vi_type_from_yaml == VI_TYPE_EUROC_MAV)
    {
      ROS_WARN("EUROC test");
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
                        cv::CALIB_ZERO_DISPARITY,0,cv::Size(w,h));
      cv::Mat K0_rect = P0.rowRange(0,3).colRange(0,3);
      cv::Mat K1_rect = P1.rowRange(0,3).colRange(0,3);
      cv::Mat D0_rect,D1_rect;
      D1_rect = D0_rect = (cv::Mat1d(4, 1) << 0,0,0,0);
      dc.setSteroCamInfo(w,h,
                         K0, D0, K0_rect, D0_rect, R0, P0,
                         K1, D1, K1_rect, D1_rect, R1, P1,
                         T_c0_c1,STEREO_UNRECT);
    }
    if(vi_type_from_yaml == VI_TYPE_KITTI_STEREO)
    {
      ROS_WARN("KITTI test");
      int w = getIntVariableFromYaml(configFilePath,             "image_width");
      int h = getIntVariableFromYaml(configFilePath,             "image_height");
      Mat4x4  P0_ = Mat44FromYaml(configFilePath,"cam0_projection_matrix");
      Mat4x4  P1_ = Mat44FromYaml(configFilePath,"cam1_projection_matrix");
      Mat4x4  K_inverse;
      K_inverse.fill(0);
      Mat3x3 K = P0_.topLeftCorner(3,3);
      K_inverse.topLeftCorner(3,3) = K.inverse();
      //cout << "K_inverse" << endl << K_inverse << endl;
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
    }

    string vocFile;
    nh.getParam("/voc", vocFile);
    Vocabulary vocTmp(vocFile);
    voc = vocTmp;
    Database dbTmp(voc, true, 0);
    db = dbTmp;

    ROS_WARN("ClientNum: %d ", ClientNum);
    merger = new Merging(nh, lc_paras, voc, db, dc, saveImage, ClientNum, IntraLoop);
    //


    for(int i = 0; i < ClientNum; i++)
    {
      kf_sub[i] = nh.subscribe<covis::KeyFrame>
          ("/Agent" + to_string(i) + "/Map_kf", 2000,
           &MergingNodeletClass::merge_callback, this, ros::TransportHints().tcpNoDelay());

    }
    kfmsgthread_ = std::shared_ptr<std::thread>
        (new std::thread(&MergingNodeletClass::kfmsgProcess, this));


  }



};//class MergingNodeletClass
}//namespace covis_ns



PLUGINLIB_EXPORT_CLASS(covis_ns::MergingNodeletClass, nodelet::Nodelet)
