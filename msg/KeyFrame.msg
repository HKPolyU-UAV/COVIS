Header header
int64  frame_id
uint64 AgentId_
int8   command
geometry_msgs/Transform  T_c_w
geometry_msgs/Transform  T_c_i
int32 lm_count
geometry_msgs/Point32[]  lm_3d       # p3d in cameraframe
geometry_msgs/Point32[]  lm_2d       # p2d 
geometry_msgs/Point32[]  lm_2d_post  # p2d FAST
int64[] lm_2d_descriptor
int64[] lm_2d_post_descriptor
sensor_msgs/Image img0              # visualize matched img only

# lm_3d in world frame
# lm_2d in pixel frame
# lm_2d_post in pixel frame, redetected by FAST or GoodFeaturesTotrack
# lm_2d_descriptor lm_2d_post_descriptor is corresponding BRIEF descriptor



