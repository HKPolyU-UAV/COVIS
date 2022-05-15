#ifndef RVIZPATH_H
#define RVIZPATH_H

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <eigen3/Eigen/Dense>
#include <include/common.h>


#define DEFAULT_NUM_OF_POSE (10000)

class RVIZPath
{
private:
  ros::Publisher path_pub;
  nav_msgs::Path path;
  unsigned int numOfPose;
  string GlobalFrameId;
  size_t AgentId_;
  string AgentFrameId;
  bool is_Server = false;
public:

  RVIZPath(ros::NodeHandle& nh, string topic_name, string GlobalFrameId, int bufferCount=1, int maxNumOfPose=-1);
  RVIZPath(ros::NodeHandle& nh, string topic_name, size_t AgentId_, string AgentFrameId, int bufferCount=1, int maxNumOfPose=-1);

  ~RVIZPath();

  void pubPathT_c_w(const SE3 T_c_w, const ros::Time stamp=ros::Time::now(), size_t AgentId=0);
  void pubPathT_w_c(const SE3 T_w_c, const ros::Time stamp=ros::Time::now(), size_t AgentId=0);
  void clearPath();
  int getSize();
};//class RVIZPath



#endif // RVIZPATH_H
