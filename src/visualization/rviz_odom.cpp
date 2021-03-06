#include "include/rviz_odom.h"

RVIZOdom::RVIZOdom()
{

}
/** @brief Publish in Server frame

*/
RVIZOdom::RVIZOdom(ros::NodeHandle& nh,
                   string topicName, string frameId,
                   int bufferSize)
{
  odom_pub = nh.advertise<nav_msgs::Odometry>(topicName, bufferSize);
  this->odom.header.frame_id = frameId;
}
/** @brief Publish in Agent frame

*/
RVIZOdom::RVIZOdom(ros::NodeHandle& nh,
                   string topicName,size_t &AgentId_, string &AgentFrameId,
                   int bufferSize)
{
  this->AgentId_ = AgentId_;
  this->AgentFrameId = AgentFrameId;
  stringstream* ss;
  ss = new stringstream;
  *ss << "/Agent" << this->AgentId_ << topicName;
  odom_pub = nh.advertise<nav_msgs::Odometry>(ss->str(), bufferSize);
  this->odom.header.frame_id = this->AgentFrameId;
}
void RVIZOdom::pubOdom(const Quaterniond q, const Vec3 t, const Vec3 v, const ros::Time stamp)
{
  odom.header.stamp = stamp;
  odom.child_frame_id = "imuframe";
  odom.pose.pose.position.x=t(0);
  odom.pose.pose.position.y=t(1);
  odom.pose.pose.position.z=t(2);
  odom.twist.twist.linear.x = v(0);
  odom.twist.twist.linear.y = v(1);
  odom.twist.twist.linear.z = v(2);
  odom.pose.pose.orientation.w=q.w();
  odom.pose.pose.orientation.x=q.x();
  odom.pose.pose.orientation.y=q.y();
  odom.pose.pose.orientation.z=q.z();
  odom_pub.publish(odom);
}
