#include "include/keyframe_msg_handler.h"
inline static bool sortbyresponse(const cv::KeyPoint &a,
                          const cv::KeyPoint &b )
{
    return a.response > b.response;
}
KeyFrameMsgHandler::KeyFrameMsgHandler()
{

}

KeyFrameMsgHandler::KeyFrameMsgHandler(ros::NodeHandle &nh, string topic_name, size_t AgentId_, int buffersize)
{
    this->AgentId_ = AgentId_;
    //kf_pub = nh.advertise<covis::KeyFrame>("/Agent" +to_string(AgentId_)+topic_name,1);      // deprecated api for local map
    kf_pub = nh.advertise<covis::KeyFrame>("/Agent" +to_string(AgentId_)+"/Map_kf",1); // for map merge and loop closure
    kf_odom_pub = nh.advertise<nav_msgs::Odometry>("/Agent" +to_string(AgentId_)+"/keyframe_pose", 1000);
    string brief_pattern_file;
    nh.getParam("/briefpath", brief_pattern_file);
    ROS_WARN("loading BRIEF pattern %s ", brief_pattern_file.c_str());
    BRIEF_ = std::make_shared<BriefDescriptorExtractor> ();
    BRIEF_->create(brief_pattern_file);

}

void BriefDescriptorExtractor::create(const string& pattern_path)
{
    cv::FileStorage fs(pattern_path, cv::FileStorage::READ);
    if(!fs.isOpened()) {
      throw string("Could not open BRIEF file")+ pattern_path;}

    vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;
    m_brief.importPairs(x1,y1,x2,y2);


}
void BriefDescriptorExtractor::extract(const cv::Mat& img, vector<cv::KeyPoint>& keypoints, vector<DVision::BRIEF::bitset>& brief_descriptors)
{
  m_brief.compute(img, keypoints, brief_descriptors);
}

//void KeyFrameMsgHandler::cmdLMResetPub(ros::Time stamp)
//{
//    cout << "KeyFrameMsg::cmdLMResetPub: " << endl;
//    covis::KeyFrame kf;
//    kf.header.stamp = stamp;
//    kf.frame_id = 0;
//    kf.lm_count = 0;
//    kf.command = KFMSG_CMD_RESET_LM;
//    kf.img0 = sensor_msgs::Image();
//    kf.img1 = sensor_msgs::Image();
//    kf.lm_count = 0;
//    kf.lm_id_data = std_msgs::Int64MultiArray();
//    kf.lm_2d_data.clear();
//    kf.lm_3d_data.clear();
//    kf.lm_descriptor_data = std_msgs::UInt8MultiArray();
//    kf.T_c_w = geometry_msgs::Transform();
//    kf_pub.publish(kf);
//}

/**
 * prepare landmark and pose
 * @param frame current frame
 * @param msg user-defined ros message
 * @param stamp ros Time
 */
void KeyFrameMsgHandler::preProcessKeyFramemsg(CameraFrame::Ptr& frame, VIMOTION::Ptr& vimotion, covis::KeyFrame& msg, ros::Time stamp)
{

  static int count = 0 ;
  ROS_DEBUG("Agent %lu sent: %d ", this->AgentId_, count++);
  msg.header.stamp = stamp;
  msg.frame_id = frame->frame_id;
  msg.AgentId_ = this->AgentId_;     //AgentId_
  msg.command = KFMSG_CMD_NONE;
  msg.T_c_w.translation.x = frame->T_c_w.translation()[0];
  msg.T_c_w.translation.y = frame->T_c_w.translation()[1];
  msg.T_c_w.translation.z = frame->T_c_w.translation()[2];
  msg.T_c_w.rotation.w    = frame->T_c_w.unit_quaternion().w();
  msg.T_c_w.rotation.x    = frame->T_c_w.unit_quaternion().x();
  msg.T_c_w.rotation.y    = frame->T_c_w.unit_quaternion().y();
  msg.T_c_w.rotation.z    = frame->T_c_w.unit_quaternion().z();

  msg.T_c_i.translation.x = vimotion->T_c_i.translation()[0];
  msg.T_c_i.translation.y = vimotion->T_c_i.translation()[1];
  msg.T_c_i.translation.z = vimotion->T_c_i.translation()[2];
  msg.T_c_i.rotation.w    = vimotion->T_c_i.unit_quaternion().w();
  msg.T_c_i.rotation.x    = vimotion->T_c_i.unit_quaternion().x();
  msg.T_c_i.rotation.y    = vimotion->T_c_i.unit_quaternion().y();
  msg.T_c_i.rotation.z    = vimotion->T_c_i.unit_quaternion().z();


  vector<int64_t> lm_id;
  vector<Vec2> lm_2d;
  vector<Vec3> lm_3d_c;
  frame->getKeyFrameInf(lm_id, lm_2d, lm_3d_c); // lm_2d is undistort but not normed | lm_3d in camera frame
//  cout << "lm_id size: " << lm_id.size() << endl;
//  cout << "lm_2d size: " << lm_2d.size() << endl;
//  cout << "lm_3d size: " << lm_3d.size() << endl;

  msg.lm_count =  static_cast<int32_t>(lm_id.size());
//  for(size_t i=0; i<lm_id.size(); i++)
//  {
//    //cout << "New getKeyFrame: " << lm_id.at(i) << " p2d_u "  << lm_2d.at(i).transpose() << " p3d " << lm_3d_c.at(i).transpose() << endl;
//  }
  for(size_t i=0; i < lm_id.size(); i++)
  {
    geometry_msgs::Point32 p2d, p3d;
    p2d.x= lm_2d[i](0);
    p2d.y= lm_2d[i](1);
    p2d.z = 1;
    msg.lm_2d.push_back(p2d);

    p3d.x= lm_3d_c[i](0);
    p3d.y= lm_3d_c[i](1);
    p3d.z= lm_3d_c[i](2);
    msg.lm_3d.push_back(p3d);

  }

  nav_msgs::Odometry odometry;
  odometry.header.stamp = stamp;
  odometry.header.frame_id = "map";
//  odometry.pose.pose.position.x = vimotion->T_w_i.translation()[0];
//  odometry.pose.pose.position.y = vimotion->T_w_i.translation()[1];
//  odometry.pose.pose.position.z = vimotion->T_w_i.translation()[2];
//  odometry.pose.pose.orientation.x = vimotion->T_w_i.unit_quaternion().x();
//  odometry.pose.pose.orientation.y = vimotion->T_w_i.unit_quaternion().y();
//  odometry.pose.pose.orientation.z = vimotion->T_w_i.unit_quaternion().z();
//  odometry.pose.pose.orientation.w = vimotion->T_w_i.unit_quaternion().w();
  odometry.pose.pose.position.x = frame->T_c_w.inverse().translation()[0];
  odometry.pose.pose.position.y = frame->T_c_w.inverse().translation()[1];
  odometry.pose.pose.position.z = frame->T_c_w.inverse().translation()[2];
  odometry.pose.pose.orientation.x = frame->T_c_w.inverse().unit_quaternion().x();
  odometry.pose.pose.orientation.y = frame->T_c_w.inverse().unit_quaternion().y();
  odometry.pose.pose.orientation.z = frame->T_c_w.inverse().unit_quaternion().z();
  odometry.pose.pose.orientation.w = frame->T_c_w.inverse().unit_quaternion().w();
  kf_odom_pub.publish(odometry);
  //ROS_WARN("pose size %zu \n", sizeof(msg.T_c_w));

}

/**
 * prepare landmark and pose
 * @param msg user-defined ros message
 * @param img0
 */
void KeyFrameMsgHandler::ProcessKeyFramemsg(covis::KeyFrame& msg, const cv::Mat& img0)
{
  //cout << "process KF " << endl;

  cv_bridge::CvImage cvimg0(std_msgs::Header(), "mono8", img0);
  try
  {
    if(0)
      cvimg0.toImageMsg(msg.img0);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("convert cv::Mat to ROS image: %s", e.what());
    return;
  }


  auto start = std::chrono::high_resolution_clock::now();

  vector<cv::Point2f> p2d;
  vector <cv::KeyPoint> old_keypoints, keypoints;
  vector<DVision::BRIEF::bitset> old_brief_descriptors, brief_descriptors;

  if(0)
  {
    cv::goodFeaturesToTrack(img0, p2d, 1000, 0.01, 5);
    cv::KeyPoint::convert(p2d, keypoints);

//    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000,1.2f,8,31,0,2, cv::ORB::HARRIS_SCORE,31,20);
//    orb->detect(img0, keypoints);

  }
  else
  {
    cv::FAST(img0, keypoints, 20, true);
  }

  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
  //ROS_DEBUG("fast feature detection time: %f s \n", (duration.count()/1000.0));

  for (size_t i = 0; i <keypoints.size(); i++)
  {
    geometry_msgs::Point32 p2d_post;
    p2d_post.x = keypoints.at(i).pt.x;
    p2d_post.y = keypoints.at(i).pt.y;
    p2d_post.z = 1;
    msg.lm_2d_post.push_back(p2d_post);
    //    Vec2 p2d_post_norm = d_camera.pixel2norm(Vec2(keypoints[i].pt.x, keypoints[i].pt.y));
    //    p2d_post.x = p2d_post_norm(0);
    //    p2d_post.y = p2d_post_norm(1);
  }
  //cout << "kf id " << msg.frame_id << endl;
  //cout << "lm2d_post: " << msg.lm_2d_post.size() << endl;
//  for(size_t i =0; i < msg.lm_2d_post.size(); i++)
//  {

//    //if(msg.lm_2d_post.at(i).x <= 0 || msg.lm_2d_post.at(i).y <=0 || msg.lm_2d_post.at(i).z <=0)
//     cout << cv::Point2f(msg.lm_2d_post.at(i).x, msg.lm_2d_post.at(i).y) << endl;

//  }

  BRIEF_->extract(img0, keypoints, brief_descriptors);


  for(int i = 0; i < brief_descriptors.size(); i++)
  {
    for (int k = 0; k < 4 ; k++)
    {
      unsigned long long int tmp_int = 0;
      for(int j = 255 - 64*k; j > 255 - 64*k - 64; j--)
      { 
          tmp_int <<= 1;
          tmp_int += brief_descriptors[i][j];
      }
      msg.lm_2d_post_descriptor.push_back(tmp_int);
    }

  }
  //cout << "kf id " << msg.frame_id << endl;
 // cout << "lm2d: " << msg.lm_2d.size() << " | " << "lm3d:" << msg.lm_3d.size() << endl;
  for(size_t i = 0; i < msg.lm_2d.size(); i++)
  {
    cv::KeyPoint kpt;
    kpt.pt = cv::Point2f(msg.lm_2d.at(i).x, msg.lm_2d.at(i).y);
    old_keypoints.push_back(kpt);
    //cout << cv::Point2f(msg.lm_2d.at(i).x, msg.lm_2d.at(i).y) << " | " << cv::Point3f(msg.lm_3d.at(i).x, msg.lm_3d.at(i).y, msg.lm_3d.at(i).z) << endl;

  }
  BRIEF_->extract(img0, old_keypoints, old_brief_descriptors);

  for(int i = 0; i < old_brief_descriptors.size(); i++)
  {
    for (int k = 0; k < 4 ; k++)
    {
      unsigned long long int tmp_int = 0;
      for(int j = 255 - 64*k; j > 255 - 64*k - 64; j--)
      {
          tmp_int <<= 1;
          tmp_int += old_brief_descriptors[i][j];
      }
      msg.lm_2d_descriptor.push_back(tmp_int);
    }

  }
//  cout << "lm_2d size: " << msg.lm_2d.size() << endl;
//  cout << "brief_descriptors size: " << old_brief_descriptors.size() << endl;
//  cout << "x4 lm_2d_des: size "<< msg.lm_2d_descriptor.size() << endl;


  kf_pub.publish(msg);


}
void KeyFrameMsgHandler::unpack(covis::KeyFrameConstPtr kf_const_ptr,
                                int64_t         &frame_id,
                                size_t          &AgentId_,
                                SE3             &T_c_w,
                                SE3             &T_c_i,
                                cv::Mat         &img0,
                                int             &lm_count,
                                vector<Vec3>    &lm_3d,
                                vector<Vec2>    &lm_2d,
                                vector<Vec2>    &lm_2d_post,
                                vector<DVision::BRIEF::bitset> &lm_2d_descriptor,
                                vector<DVision::BRIEF::bitset> &lm_2d_post_descriptor,
                                //vector<cv::Mat> &lm_2d_descriptor,  // descriptors of img0
                                //vector<cv::Mat> &lm_2d_post_descriptor,  // descriptors of img0
                                ros::Time       &time)
{
  frame_id = kf_const_ptr->frame_id;
  AgentId_ = kf_const_ptr->AgentId_;
  Vec3 t;
  Quaterniond uq;
  t(0) = kf_const_ptr->T_c_w.translation.x;
  t(1) = kf_const_ptr->T_c_w.translation.y;
  t(2) = kf_const_ptr->T_c_w.translation.z;
  uq.w() = kf_const_ptr->T_c_w.rotation.w;
  uq.x() = kf_const_ptr->T_c_w.rotation.x;
  uq.y() = kf_const_ptr->T_c_w.rotation.y;
  uq.z() = kf_const_ptr->T_c_w.rotation.z;
  T_c_w = SE3(uq,t);

  Vec3 t_ci;
  Quaterniond uq_ci;
  t_ci(0) = kf_const_ptr->T_c_i.translation.x;
  t_ci(1) = kf_const_ptr->T_c_i.translation.y;
  t_ci(2) = kf_const_ptr->T_c_i.translation.z;
  uq_ci.w() = kf_const_ptr->T_c_i.rotation.w;
  uq_ci.x() = kf_const_ptr->T_c_i.rotation.x;
  uq_ci.y() = kf_const_ptr->T_c_i.rotation.y;
  uq_ci.z() = kf_const_ptr->T_c_i.rotation.z;
  T_c_i = SE3(uq_ci,t_ci);

  time = kf_const_ptr->header.stamp;

  img0.release();
  lm_3d.clear();
  lm_2d.clear();
  lm_2d_post.clear();
  lm_2d_descriptor.clear();
  lm_2d_post_descriptor.clear();

  if(0){
      cv_bridge::CvImagePtr cvbridge_image  = cv_bridge::toCvCopy(kf_const_ptr->img0, kf_const_ptr->img0.encoding);
      img0 = cvbridge_image->image;
  }


  lm_count = kf_const_ptr->lm_count;

  for(size_t i=0; i< kf_const_ptr->lm_2d.size(); i++)
  {
//        cv::Mat descriptor = cv::Mat(1,32,CV_8U);
//        for(auto j=0; j<32; j++)
//        {
//            descriptor.at<uint8_t>(0,j)= kf_const_ptr->lm_descriptor_data.data.at(i*32+j);
//        }
//        lm_descriptors.push_back(descriptor);
      //cout << lm_descriptors.at(i) << endl;
      //lm_id.push_back(kf_const_ptr->lm_id_data.data[i]);
      Vec2 p2d(kf_const_ptr->lm_2d.at(i).x, kf_const_ptr->lm_2d.at(i).y);
      Vec3 p3d(kf_const_ptr->lm_3d.at(i).x, kf_const_ptr->lm_3d.at(i).y, kf_const_ptr->lm_3d.at(i).z);
      lm_2d.push_back(p2d);
      lm_3d.push_back(p3d);
  }

  for(size_t i = 0; i < kf_const_ptr->lm_2d_post.size(); i++)
  {
      Vec2 p2d_post(kf_const_ptr->lm_2d_post.at(i).x, kf_const_ptr->lm_2d_post.at(i).y);
      lm_2d_post.push_back(p2d_post);
  }


  for (int i = 0; i< (int)kf_const_ptr->lm_2d_descriptor.size(); i += 4)
  {
     boost::dynamic_bitset<> bitset(256);
     for (int k = 0; k< 4; k++)
     {
       unsigned long long int tmp = kf_const_ptr->lm_2d_descriptor[i+k];

       for(int j = 0; j < 64 ; j++, tmp >>=1)
       {
           bitset[256 - 64 * (k + 1) + j] = (tmp & 1);

       }
     }
     //cout << "i: " << i << " bitset: " << bitset << endl;
     //cout << "i: " << i << " brief_des: " << brief_des << endl;
     lm_2d_descriptor.push_back(bitset);
  }

  for (int i = 0; i< (int)kf_const_ptr->lm_2d_post_descriptor.size(); i += 4)
  {
     boost::dynamic_bitset<> bitset(256);
     for (int k = 0; k< 4; k++)
     {
       unsigned long long int tmp = kf_const_ptr->lm_2d_post_descriptor[i+k];

       for(int j = 0; j < 64 ; j++, tmp >>=1)
       {
           bitset[256 - 64 * (k + 1) + j] = (tmp & 1);

       }

     }
     //cout << "i: " << i << " bitset: " << bitset << endl;
     lm_2d_post_descriptor.push_back(bitset);
  }

}
/*
void KeyFrameMsgHandler::pub(CameraFrame& frame, ros::Time stamp)
{

    static int count = 0;
    ROS_DEBUG("Agent %lu sent: %d ", this->AgentId_, count++);
    covis::KeyFrame kf;

    kf.header.stamp = stamp;
    kf.frame_id = frame.frame_id;
    kf.AgentId_ = this->AgentId_;     //AgentId_
    kf.command = KFMSG_CMD_NONE;
    cv_bridge::CvImage cvimg0(std_msgs::Header(), "mono8", frame.img0);
    cvimg0.toImageMsg(kf.img0);
    if(frame.d_camera.cam_type==DEPTH_D435)
    {
        cv_bridge::CvImage cv_d_img(std_msgs::Header(), "16UC1", frame.d_img);
        cv_d_img.toImageMsg(kf.img1);
    }

    if(frame.d_camera.cam_type==STEREO_UNRECT || frame.d_camera.cam_type==STEREO_RECT)
    {
        cv_bridge::CvImage cvimg1(std_msgs::Header(), "mono8", frame.img1);
        cvimg1.toImageMsg(kf.img1);
    }

    vector<int64_t> lm_id;
    vector<Vec2> lm_2d;
    vector<Vec3> lm_3d;

    frame.getKeyFrameInf(lm_id,lm_2d,lm_3d); // lm_2d is undistort but not normed | lm_3d in world frame
    kf.lm_count =  static_cast<int32_t>(lm_id.size());

    for(size_t i=0; i<lm_id.size(); i++)
    {
      //cout << " New getKeyFrame: " << lm_id.at(i) << " " << lm_2d.at(i).transpose() << " "  << lm_3d.at(i).transpose() << endl;
    }
    kf.lm_id_data.layout.dim.push_back(std_msgs::MultiArrayDimension());
    kf.lm_id_data.layout.dim[0].label = "lm_id";
    kf.lm_id_data.layout.dim[0].size = static_cast<uint32_t>(lm_id.size());
    kf.lm_id_data.layout.dim[0].stride = static_cast<uint32_t>(lm_id.size());

    kf.lm_id_data.data.clear();
    kf.lm_id_data.data.insert(kf.lm_id_data.data.end(),lm_id.begin(),lm_id.end());

    kf.lm_descriptor_data.layout.dim.push_back(std_msgs::MultiArrayDimension());
    kf.lm_descriptor_data.layout.dim.push_back(std_msgs::MultiArrayDimension());

    kf.lm_descriptor_data.layout.dim[0].label = "lm_descriptor";
    kf.lm_descriptor_data.layout.dim[0].size = 0;
    kf.lm_descriptor_data.layout.dim[0].stride = 0;
//    kf.lm_descriptor_data.layout.dim[0].size = static_cast<uint32_t>(lm_id.size());
//    kf.lm_descriptor_data.layout.dim[0].stride = static_cast<uint32_t>(32*lm_id.size());
    kf.lm_descriptor_data.layout.dim[1].label = "32uint_descriptor";
    kf.lm_descriptor_data.layout.dim[1].size = 0;
    kf.lm_descriptor_data.layout.dim[1].stride = 0;
//    kf.lm_descriptor_data.layout.dim[1].size = 1;
//    kf.lm_descriptor_data.layout.dim[1].stride = 32;


//    for(size_t i=0; i<lm_id.size(); i++)
//    {
//        //cout << "i:" << lm_descriptors.at(i).type() << "  " << "size:" << lm_descriptors.at(i).size << endl;
//        //cout << "i " << lm_descriptors.at(i) << endl;
//        for(int j=0; j<32; j++)
//        {
//            kf.lm_descriptor_data.data.push_back(lm_descriptors.at(i).at<uint8_t>(0,j));
//            //cout << unsigned(lm_descriptors.at(i).at<uint8_t>(0,j)) << " ";
//        }
//    }

    for(size_t i=0; i<lm_id.size(); i++)
    {
        Vec2 p2d=lm_2d.at(i);
        Vec3 p3d=lm_3d.at(i);
        geometry_msgs::Vector3 vp2d;
        geometry_msgs::Vector3 vp3d;
        vp2d.x = p2d[0];
        vp2d.y = p2d[1];
        kf.lm_2d_data.push_back(vp2d);
        vp3d.x = p3d[0];
        vp3d.y = p3d[1];
        vp3d.z = p3d[2];
        kf.lm_3d_data.push_back(vp3d);
    }
    //cout << "SE3 T_c_w: " << frame.T_c_w << endl;
    Vec3 t=frame.T_c_w.translation();
    Quaterniond uq= frame.T_c_w.unit_quaternion();
    kf.T_c_w.translation.x=t[0];
    kf.T_c_w.translation.y=t[1];
    kf.T_c_w.translation.z=t[2];
    kf.T_c_w.rotation.w=uq.w();
    kf.T_c_w.rotation.x=uq.x();
    kf.T_c_w.rotation.y=uq.y();
    kf.T_c_w.rotation.z=uq.z();
    kf.command = KFMSG_CMD_NONE;

    kf_pub.publish(kf);
}

void KeyFrameMsgHandler::unpack(covis::KeyFrameConstPtr kf_const_ptr,
                         int64_t         &frame_id,
                         size_t          &AgentId_,
                         cv::Mat         &img0,
                         cv::Mat         &img1,
                         int             &lm_count,
                         vector<int64_t> &lm_id,
                         vector<Vec2>    &lm_2d,
                         vector<Vec3>    &lm_3d,
                         vector<cv::Mat> &lm_descriptors,
                         SE3             &T_c_w,
                         ros::Time       &time)
{
    img0.release();
    img1.release();
    lm_id.clear();
    lm_2d.clear();
    lm_3d.clear();
    lm_descriptors.clear();

    frame_id = kf_const_ptr->frame_id;
    AgentId_ = kf_const_ptr->AgentId_;
    cv_bridge::CvImagePtr cvbridge_image  = cv_bridge::toCvCopy(kf_const_ptr->img0, kf_const_ptr->img0.encoding);
    img0=cvbridge_image->image;
    cv_bridge::CvImagePtr cvbridge_d_image = cv_bridge::toCvCopy(kf_const_ptr->img1, kf_const_ptr->img1.encoding);
    img1 = cvbridge_d_image->image;
    lm_count = kf_const_ptr->lm_count;
    int count =  kf_const_ptr->lm_count;
    for(auto i=0; i<count; i++)
    {
//        cv::Mat descriptor = cv::Mat(1,32,CV_8U);
//        for(auto j=0; j<32; j++)
//        {
//            descriptor.at<uint8_t>(0,j)=kf_const_ptr->lm_descriptor_data.data.at(i*32+j);
//        }
//        lm_descriptors.push_back(descriptor);
        //cout << lm_descriptors.at(i) << endl;
        lm_id.push_back(kf_const_ptr->lm_id_data.data[i]);
        Vec2 p2d(kf_const_ptr->lm_2d_data.at(i).x,kf_const_ptr->lm_2d_data.at(i).y);
        Vec3 p3d(kf_const_ptr->lm_3d_data.at(i).x,kf_const_ptr->lm_3d_data.at(i).y,kf_const_ptr->lm_3d_data.at(i).z);
        lm_2d.push_back(p2d);
        lm_3d.push_back(p3d);
    }
    Vec3 t;
    Quaterniond uq;
    t(0) = kf_const_ptr->T_c_w.translation.x;
    t(1) = kf_const_ptr->T_c_w.translation.y;
    t(2) = kf_const_ptr->T_c_w.translation.z;
    uq.w() = kf_const_ptr->T_c_w.rotation.w;
    uq.x() = kf_const_ptr->T_c_w.rotation.x;
    uq.y() = kf_const_ptr->T_c_w.rotation.y;
    uq.z() = kf_const_ptr->T_c_w.rotation.z;
    T_c_w = SE3(uq,t);
    time = kf_const_ptr->header.stamp;

}
*/

