#include "include/rviz_mesh.h"


RVIZMesh::RVIZMesh(){

}
RVIZMesh::~RVIZMesh(){

}
/** @brief Publish in Server frame, deprecated

*/
RVIZMesh::RVIZMesh(ros::NodeHandle& nh, string topicName, string GlobalFrameId, int bufferSize)
{
  mesh_pub = nh.advertise<visualization_msgs::Marker>(topicName, bufferSize);
  this->GlobalFrameId = GlobalFrameId; //"/map"//
  is_Server = true;

}

/** @brief Publish in Agent frame

*/
RVIZMesh::RVIZMesh(ros::NodeHandle& nh,
                   string topicName, size_t AgentId_, string AgentFrameId,
                   int bufferSize)
{
  this->AgentId_ = AgentId_;
  this->AgentFrameId = AgentFrameId;
  mesh_pub = nh.advertise<visualization_msgs::Marker>("/Agent" + to_string(this->AgentId_) + topicName, bufferSize);
  is_Server = false;


}


void RVIZMesh::PubT_w_c(SE3 cam_pose, ros::Time time, size_t AgentId)
{

  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;
  //marker.header
  if(is_Server)
     marker.header.frame_id = this->GlobalFrameId;
  else
     marker.header.frame_id = this->AgentFrameId;
  marker.header.stamp = time;
  marker.ns = "Mesh";

  if(is_Server)
    marker.id = AgentId;  //unique identifier of drone
  else
    marker.id = this->AgentId_;  //unique identifier of drone

  marker.type = visualization_msgs::Marker::MESH_RESOURCE;
  marker.mesh_resource = "package://covis/models/hummingbird.mesh";

  marker.action = visualization_msgs::Marker::ADD;

  double scale = 1.0;
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;

  std_msgs::ColorRGBA white, green, red, yellow, blue;
  white.r = 1.0;
  white.g = 1.0;
  white.b = 1.0;
  white.a = 1.0;

  green.r = 0.0;
  green.g = 1.0;
  green.b = 0.0;
  green.a = 1.0;

  red.r = 1.0;
  red.g = 0.0;
  red.b = 0.0;
  red.a = 1.0;

  blue.r = 0.0;
  blue.g = 0.0;
  blue.b = 1.0;
  blue.a = 1.0;

  yellow.r = 1.0;
  yellow.g = 1.0;
  yellow.b = 0.0;
  yellow.a = 1.0;

  if(marker.id == 0)
  {
    marker.color = red;
  }
  else if(marker.id == 1)
  {
    marker.color = green;
  }
  else if(marker.id == 2)
  {
    marker.color = blue;
  }
  else if(marker.id == 3)
  {
    marker.color = yellow;
  }

  Vec3 t = cam_pose.translation();;
  Quaterniond r = cam_pose.so3().unit_quaternion();

  marker.pose.position.x = t.x();
  marker.pose.position.y = t.y();
  marker.pose.position.z = t.z();
  marker.pose.orientation.w = r.w();
  marker.pose.orientation.x = r.x();
  marker.pose.orientation.y = r.y();
  marker.pose.orientation.z = r.z();
  mesh_pub.publish(marker);

}

void RVIZMesh::PubT_w_i(SE3 drone_pose, ros::Time time, size_t AgentId)
{
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;
  //marker.header
  if(is_Server)
     marker.header.frame_id = this->GlobalFrameId;
  else
     marker.header.frame_id = this->AgentFrameId;
  marker.header.stamp = time;
  marker.ns = "Mesh";

  if(is_Server)
    marker.id = AgentId;  //unique identifier of drone
  else
    marker.id = this->AgentId_;  //unique identifier of drone

  marker.type = visualization_msgs::Marker::MESH_RESOURCE;
  marker.mesh_resource = "package://covis/models/hummingbird.mesh";

  marker.action = visualization_msgs::Marker::ADD;

  double scale = 1.0;
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;

  std_msgs::ColorRGBA white, green, red, yellow, blue;
  white.r = 1.0;
  white.g = 1.0;
  white.b = 1.0;
  white.a = 1.0;

  green.r = 0.0;
  green.g = 1.0;
  green.b = 0.0;
  green.a = 1.0;

  red.r = 1.0;
  red.g = 0.0;
  red.b = 0.0;
  red.a = 1.0;

  blue.r = 0.0;
  blue.g = 0.0;
  blue.b = 1.0;
  blue.a = 1.0;

  yellow.r = 1.0;
  yellow.g = 1.0;
  yellow.b = 0.0;
  yellow.a = 1.0;

  if(marker.id == 0)
  {
    marker.color = red;
  }
  else if(marker.id == 1)
  {
    marker.color = green;
  }
  else if(marker.id == 2)
  {
    marker.color = blue;
  }
  else if(marker.id == 3)
  {
    marker.color = yellow;
  }

  Vec3 t = drone_pose.translation();;
  Quaterniond r = drone_pose.so3().unit_quaternion();

  marker.pose.position.x = t.x();
  marker.pose.position.y = t.y();
  marker.pose.position.z = t.z();
  marker.pose.orientation.w = r.w();
  marker.pose.orientation.x = r.x();
  marker.pose.orientation.y = r.y();
  marker.pose.orientation.z = r.z();
  mesh_pub.publish(marker);

}

#if 0
void RVIZMesh::PubMarkerArray()
{
  visualization_msgs::MarkerArray marker_array;
  for(auto &marker:markers)
  {
    marker_array.markers.push_back(marker);
  }

  mesh_pub.publish(marker_array);
  //cout << "pub marker size: " << markers.size() << endl;

}
void RVIZMesh::clearMarkerArray()
{
  markers.clear();

}
#endif
