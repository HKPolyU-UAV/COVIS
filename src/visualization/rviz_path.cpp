#include <include/rviz_path.h>

/** @brief Publish in Server frame

*/
RVIZPath::RVIZPath(ros::NodeHandle& nh, string topic_name, string GlobalFrameId, int bufferCount, int maxNumOfPose)
{
  path_pub = nh.advertise<nav_msgs::Path>(topic_name, bufferCount);
  if(maxNumOfPose==-1)
  {
    numOfPose = DEFAULT_NUM_OF_POSE;
  }else
  {
    numOfPose = maxNumOfPose;
  }
  this->GlobalFrameId = GlobalFrameId;
  path.header.frame_id = this->GlobalFrameId;
  is_Server = true;
}
/** @brief Publish in Agent frame

*/

RVIZPath::RVIZPath(ros::NodeHandle& nh, string topic_name, size_t AgentId_, string AgentFrameId, int bufferCount, int maxNumOfPose)
{

  this->AgentId_ = AgentId_;
  this->AgentFrameId = AgentFrameId;

  stringstream* ss;
  ss = new stringstream;
  *ss << "/Agent" << this->AgentId_ << topic_name;

  path_pub = nh.advertise<nav_msgs::Path>(ss->str(), bufferCount);
  if(maxNumOfPose==-1)
  {
    numOfPose = DEFAULT_NUM_OF_POSE;
  }else
  {
    numOfPose = maxNumOfPose;
  }
  path.header.frame_id = this->AgentFrameId;
  delete ss;
  is_Server = false;
}


void RVIZPath::pubPathT_c_w(const SE3 T_c_w, const ros::Time stamp, size_t AgentId)
{
  pubPathT_w_c(T_c_w.inverse(),stamp, AgentId);
}

void RVIZPath::pubPathT_w_c(const SE3 T_w_c, const ros::Time stamp, size_t AgentId)
{
  geometry_msgs::PoseStamped poseStamped;
  if(is_Server)
     poseStamped.header.frame_id = this->GlobalFrameId;
  else
     poseStamped.header.frame_id = this->AgentFrameId;

  poseStamped.header.stamp    = stamp;

  Quaterniond q = T_w_c.so3().unit_quaternion();
  Vec3        t = T_w_c.translation();

  poseStamped.pose.orientation.w = q.w();
  poseStamped.pose.orientation.x = q.x();
  poseStamped.pose.orientation.y = q.y();
  poseStamped.pose.orientation.z = q.z();
  poseStamped.pose.position.x = t[0];
  poseStamped.pose.position.y = t[1];
  poseStamped.pose.position.z = t[2];

  path.header.stamp = stamp;
  path.poses.push_back(poseStamped);

  if(path.poses.size()>=numOfPose)
  {
    path.poses.erase(path.poses.begin());
  }

  path_pub.publish(path);
}

void RVIZPath::clearPath()
{
  path.poses.clear();
}

int RVIZPath::getSize()
{
  int size = path.poses.size();
  return size;
}
