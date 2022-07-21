#include "include/merging.h"
template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

Merging::Merging(ros::NodeHandle &nh, LC_PARAS &paras, const vector<DepthCamera> &d_cameras, int save_flag_, int number, bool Intra):
  lc_paras(paras), d_cameras(d_cameras), SAVE_IMG(save_flag_), IntraLoop(Intra)
{
  number_of_Agent = static_cast<size_t>(number);
  ROS_WARN("Total number of Agent: %lu ", number_of_Agent);
  ROS_WARN("loop detection threshold: %f ", lc_paras.minScore);
  ROS_WARN("loop detection begin ID: %d ", lc_paras.lcKFStart);
  ROS_WARN("SaveImg: %d ", SAVE_IMG);
  ROS_WARN("IntraLoop: %d ", IntraLoop);
  ROS_WARN("d_cameras: %lu ", d_cameras.size());
  string vocFile;
  nh.getParam("/voc", vocFile);
  ROS_WARN("voc file: %s ", vocFile.c_str());
  voc = new BriefVocabulary(vocFile);
  db.setVocabulary(*voc, false, 0);

  // global path
  globalpaths_.resize(number_of_Agent);
  dronepub_.resize(number_of_Agent);
  for(size_t i = 0; i < number_of_Agent; i++){
    globalpaths_[i] = new RVIZPath(nh, "/Server/global_path_"+to_string(i), "map", 1, 10000);
    dronepub_[i]    = new RVIZMesh(nh, "/Server/global_drone_"+to_string(i), "map", 1000);
  }

  // inter-and-intra agent edge
  edge_merge_pub  = new RVIZEdge(nh, "/Server/merge_edge", "map", 10000);

  // inter-agent matched image
  image_transport::ImageTransport it(nh);
  merge_Img_pub = it.advertise("/Server/merge_img",1);

  m_drift.lock();
  for (size_t i = 0; i < number_of_Agent; i++){
    merged_agent[i] = false;
    T_local_global[i] = SE3();
    drifts[i] = SE3();
  }
  m_drift.unlock();

  Loop_running_ = true;
  pgo_running_ = false;
  for(size_t i = 0; i < number_of_Agent; i++)
  {
    Loopthreads_.push_back(std::thread(&Merging::runLC, this));
    ROS_WARN("%lu loop detection thread init ", i);

  }


  ROS_WARN("GBA thread start");
  pgothread_ = std::thread(&Merging::startPGO, this);

}
Merging::~Merging()
{
   cv::destroyAllWindows();
}

void Merging::setKeyFrame(const covis::KeyFrameConstPtr& msg)
{
  tic_toc_ros msg_tic;
  //STEP1.1 Unpack
  // [1]kf.frame_id
  // [2]kf.T_c_w_odom

  KeyFrameMerge kf;

  cv::Mat img0, img1_unpack;
  //vector<int64_t> lm_id_unpack;
//  vector<Vec2> lm_2d_unpack, lm_2d_post_unpack;
//  vector<Vec3> lm_3d_unpack;
  //vector<cv::Mat> lm_descriptor_unpack, lm_descriptor_post_unpack;
  //vector<DVision::BRIEF::bitset> lm_descriptor_unpack, lm_descriptor_post_unpack;

  KeyFrameMsgHandler::unpack(msg, kf.frame_id, kf.AgentId_, kf.T_c_w_odom, kf.T_c_i, kf.img0,
                             kf.lm_count, kf.lm_3d, kf.lm_2d, kf.lm_2d_post,
                            kf.lm_2d_descriptor, kf.lm_2d_post_descriptor, kf.t);


  //STEP1.2 Construct KeyFrameLC
  // [1]kf.T_c_w
  // [2]kf.keyframe_id

  BowVector kf_bv;

  kf.keyframe_id = kf_id++;

  if(kf.keyframe_id==0)
  {
    merged_agent[kf.AgentId_] = true;
    ROS_WARN("first register Agent: %lu \n", kf.AgentId_);

  }

  m_imgs.lock();
  idpair idp = make_pair(kf.AgentId_, kf.keyframe_id);
  idpairImg.insert(make_pair(idp, kf.img0));
  m_imgs.unlock();


  //STEP1.4 Construct KeyFrameLC
  //Compute bow vector
  //BowVector size: < 500      <WordId, WordValue>
  // [1]kf.kf_bv
  //voc.transform(BriefDescriptors,kf.kf_bv);


  //STEP2 add kf to list
  m_vector.lock();
  kf.T_c_w =  kf.T_c_w_odom;

  m_drift.lock();
  kf.T_c_w_global = kf.T_c_w * T_local_global[kf.AgentId_] * drifts[kf.AgentId_];
  kf.T_w_i = kf.T_c_w_global.inverse() * kf.T_c_i;
  m_drift.unlock();

  shared_ptr<KeyFrameMerge> kf_ptr =std::make_shared<KeyFrameMerge>(kf);
  //cout << "latest kf: " << kf_ptr->keyframe_id << endl;

  m_path.lock();
  globalpaths_[kf_ptr->AgentId_]->pubPathT_w_c((kf.T_c_w_global).inverse(),kf.t, kf_ptr->AgentId_);
  dronepub_[kf_ptr->AgentId_]->PubT_w_i(kf.T_w_i, kf_ptr->t, kf_ptr->AgentId_);

  m_path.unlock();

  kfs_all.push_back(kf_ptr);
  kf_queue.push(kf_ptr);

#if 0
  if(kf.AgentId_ == this->AgentId_)
  {
    //cout << "\033[1;32m this agent keyframe: \033[0m" << endl;
    kfs_this.push_back(kf_ptr);
  }
  else
  {
    // cout << "\033[1;32m merge keyframe: \033[0m" << endl;
    kfs_other.push_back(kf_ptr);
    kfbv_map.push_back(kf_bv);

  }
#endif
  m_vector.unlock();
  //ROS_DEBUG("\033[1;31m msg time: %lf \033[0m", msg_tic.dT_ms());


}
void Merging::runLC()
{
  while (Loop_running_)
  {
    shared_ptr<KeyFrameMerge> kf_ptr;
    m_vector.lock();
    if(!kf_queue.empty())
    {
      kf_ptr = kf_queue.front();
      kf_queue.pop();
    }
    m_vector.unlock();
    if(kf_ptr)
    {
      //STEP 2.1 Add des to voc and Detect Loop
      tic_toc_ros t_detect;
      int loop_index = -1;
      loop_index = AddandDetectLoop(kf_ptr);
      //ROS_DEBUG("\033[1;32m detect time: %lf \033[0m ", t_detect.dT_ms());

      if(loop_index != -1)
      {
        //ROS_DEBUG("\033[1;32m find loop candidate \033[0m ");
        tic_toc_ros t_match;
        SE3 loop_pose;
        bool is_lc = isLoopClosureKF(kfs_all.at(loop_index), kf_ptr, loop_pose);
        //ROS_DEBUG("\033[1;34m match time: %lf \033[0m ", t_match.dT_ms());

        if(is_lc)
        {
          updateGlobalPose(kfs_all.at(loop_index), kf_ptr, loop_pose);

          bool is_close = add_Loop_check(kfs_all.at(loop_index), kf_ptr, loop_pose);

          if(!is_close)
          {
            m_pgo.lock();
            pgo_running_ = true;
            graph_queue.push(kf_ptr->keyframe_id);
            m_pgo.unlock();
          }

        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }


}

int Merging::AddandDetectLoop(shared_ptr<KeyFrameMerge> kf)
{
  cv::Mat compressed_image;
#if 0
  if(SAVE_IMG)
  {
    int feature_num = static_cast<int>(kf->lm_2d_post.size());
    idpair ip = make_pair(kf->AgentId_, kf->keyframe_id);
    m_imgs.lock();
    cv::resize(idpairImg.find(ip)->second, compressed_image, cv::Size(360, 240));
    m_imgs.unlock();
    cv::cvtColor(compressed_image, compressed_image, CV_GRAY2RGB);
    cv::putText(compressed_image, "AgentID: " + to_string(kf->AgentId_) + " Index: " + to_string(kf->keyframe_id) + " feature num:" + to_string(feature_num), cv::Point2f(15, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,255));
    LoopImgs.insert(make_pair(kf->keyframe_id, compressed_image));
  }
#endif
  //m_db.lock();
  DBoW2::QueryResults ret;

  tic_toc_ros t_query;
  //cout << "keyframe brief descriptors size: " << kf->lm_2d_post_descriptor.size() << endl;
  db.query(kf->lm_2d_post_descriptor, ret, 4, kf->keyframe_id - 50);
//  cout << "query: " << endl;
//  cout << ret << endl;
  //ROS_DEBUG("\033[1;32m query time: %lf \033[0m ", t_query.dT_ms());

  tic_toc_ros t_add;

  db.add(kf->lm_2d_post_descriptor);
  //ROS_DEBUG("\033[1;32m add time: %lf \033[0m ", t_add.dT_ms());

  //m_db.unlock();

  bool find_loop = false;
  cv::Mat loop_result;
#if 0
  if(SAVE_IMG)
  {
    loop_result = compressed_image.clone();
    for(int i = 0; i < ret.size(); i++)
    {
      int tmp_idx = ret[i].Id;
      auto it = LoopImgs.find(tmp_idx);
      cv::Mat tmp_img = (it->second).clone();
      cv::putText(tmp_img, "Score: " + to_string(ret[i].Score) , cv::Point2f(15, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,255));
      cv::hconcat(loop_result, tmp_img, loop_result);
    }
    std::string image_path;
    image_path = "/home/yurong/Loop/COVIS/4loop_candidate/" + to_string(kf->keyframe_id) + ".png";

    cv::imwrite(image_path, loop_result);
  }
#endif
#if 0
  if(SAVE_IMG)
  {
    cv::imshow("loop_candidate", loop_result);
    cv::waitKey(20);
  }

#endif

  if(ret.size() >= 1 && ret[0].Score > lc_paras.minScore)
  {
    for(int i = 1; i < ret.size(); i++)
    {
      if(ret[i].Score > 0.015)
      {
        find_loop = true;
      }
    }
  }

  if(find_loop && kf->keyframe_id > lc_paras.lcKFStart)
  {
    int min_index = -1;
    for(int i = 0; i < ret.size(); i++)
    {
      if(min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
        min_index = ret[i].Id;
      //cout << i << ", min_index: " << min_index << endl;
    }
    return min_index;
  }
  else
    return -1;

}


bool Merging::isLoopClosureKF(shared_ptr<KeyFrameMerge> kf_old, shared_ptr<KeyFrameMerge> kf_cur, SE3 &se_ji)
{

  //  cout << "kf loop: " << kf0->AgentId_ << " " << kf0->keyframe_id << endl;
  //  cout << "kf curr: " << kf1->AgentId_ << " " << kf1->keyframe_id << endl;
  //std::unique_lock<std::mutex> MatchLock(m_match);

  bool is_lc = false;

  vector<cv::Point3f> matched_3d;   // in camera frame
  vector<cv::Point2f> matched_2d_cur, matched_2d_old;
  vector<cv::Point2f> kf_old_lm_2d;
  matched_3d.clear();
  matched_2d_cur.clear();
  matched_2d_old.clear();
  kf_old_lm_2d.clear();

  std::vector<uchar> status;

  for(int i = 0; i < kf_cur->lm_3d.size(); i++)
  {
      cv::Point3f pt(kf_cur->lm_3d[i](0), kf_cur->lm_3d[i](1), kf_cur->lm_3d[i][2]);
      matched_3d.push_back(pt);
  }
  for(int i = 0; i < kf_cur->lm_2d.size(); i++)
  {
      cv::Point2f pt(kf_cur->lm_2d[i](0), kf_cur->lm_2d[i](1));
      matched_2d_cur.push_back(pt);
  }

  for(int i = 0 ; i < kf_old->lm_2d_post.size(); i++)
  {
      cv::Point2f pt(kf_old->lm_2d_post.at(i)[0], kf_old->lm_2d_post.at(i)[1]);
      kf_old_lm_2d.push_back(pt);
  }

#if 0
  if(SAVE_IMG)
  {
    cv::Mat gray_img, loop_match_img;
    cv::Mat old_img = kf_old->img0;
    cv::hconcat(kf_cur->img0, old_img, gray_img);
    cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
    for(int i = 0;i < (int)kf_cur->lm_2d.size(); i++)
    {
        cv::Point2f cur_pt(kf_cur->lm_2d.at(i).x(),kf_cur->lm_2d.at(i).y());
        cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
    }
    for(int i = 0;i < (int)kf_old->lm_2d_post.size(); i++)
    {
        cv::Point2f new_pt(kf_old->lm_2d_post.at(i).x(), kf_old->lm_2d_post.at(i).y());
        new_pt.x += kf_cur->img0.cols;
        cv::circle(loop_match_img, new_pt, 5, cv::Scalar(0, 255, 0));
    }
    cv::Mat info = cv::Mat::zeros(cv::Size(loop_match_img.cols, 4*15), CV_8UC3);

    cv::putText(info, "VIO feature: " + to_string(kf_cur->lm_2d.size()), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
    cv::putText(info, "Fast feature: " + to_string(kf_old->lm_2d_post.size()), cv::Point2f(20 + kf_cur->img0.cols, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
    cv::vconcat(info, loop_match_img, loop_match_img);
    std::string save_path;
    save_path = "/home/yurong/Loop/COVIS/loop_image/" + to_string(kf_cur->keyframe_id) + "-" + to_string(kf_old->keyframe_id)
        + "_0raw_point.jpg";
    cv::imwrite(save_path, loop_match_img);


  }
#endif

  tic_toc_ros t_search;
  searchByBRIEFDes(matched_2d_old, status, kf_cur->lm_2d_descriptor,
                   kf_old->lm_2d_post_descriptor, kf_old_lm_2d); // matched_2d_old
  //printf("\033[1;34m vio brief size: %lu \033[0m  \n", kf_cur->lm_2d_descriptor.size());
  //printf("\033[1;34m fast brief size: %lu \033[0m  \n", kf_old->lm_2d_post_descriptor.size());
  //printf("\033[1;34m total num : %lu \033[0m  \n", kf_cur->lm_2d_descriptor.size() * kf_old->lm_2d_post_descriptor.size());
  //ROS_DEBUG("\033[1;34m search time: %lf \033[0m ", t_search.dT_ms());
#if 0
  if(t_search.dT_ms() >= 30.0)
  {
    if(SAVE_IMG)
    {
      cv::Mat gray_img, loop_match_img;
      cv::Mat old_img = kf_old->img0;
      cv::hconcat(kf_cur->img0, old_img, gray_img);
      cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
      for(int i = 0;i < (int)kf_cur->lm_2d.size(); i++)
      {
          cv::Point2f cur_pt(kf_cur->lm_2d.at(i).x(),kf_cur->lm_2d.at(i).y());
          cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
      }
      for(int i = 0;i < (int)kf_old->lm_2d_post.size(); i++)
      {
          cv::Point2f new_pt(kf_old->lm_2d_post.at(i).x(), kf_old->lm_2d_post.at(i).y());
          new_pt.x += kf_cur->img0.cols;
          cv::circle(loop_match_img, new_pt, 5, cv::Scalar(0, 255, 0));
      }
      cv::Mat info = cv::Mat::zeros(cv::Size(loop_match_img.cols, 4*15), CV_8UC3);

      cv::putText(info, "VIO feature: " + to_string(kf_cur->lm_2d.size()), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
      cv::putText(info, "Fast feature: " + to_string(kf_old->lm_2d_post.size()), cv::Point2f(20 + kf_cur->img0.cols, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
      cv::vconcat(info, loop_match_img, loop_match_img);
      std::string save_path;
      save_path = "/home/yurong/Loop/COVIS/slow/" + to_string(kf_cur->frame_id) + "-" + to_string(kf_old->frame_id)
          + "_searchslow.jpg";
      cv::imwrite(save_path, loop_match_img);

    }

  }
#endif


  reduceVector(matched_2d_old, status);
  reduceVector(matched_2d_cur, status);
  reduceVector(matched_3d, status);

#if 0
  /// visualize matching result
  if(SAVE_IMG)
  {
    ///publish loop closure frame

    idpair kf0_idpair = make_pair(kf_old->AgentId_,kf_old->keyframe_id);
    idpair kf1_idpair = make_pair(kf_cur->AgentId_,kf_cur->keyframe_id);

    //m_imgs.lock();
    cv::Mat img_1 = idpairImg.find(kf1_idpair)->second;
    cv::Mat img_2 = idpairImg.find(kf0_idpair)->second;
    // m_imgs.unlock();

    int gap = 15;
    int offset = img_1.cols + gap;

    cv::Mat gap_img(img_1.rows, gap, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::Mat out_img, tmp_img;
    cv::hconcat(img_1, gap_img, gap_img);
    cv::hconcat(gap_img, img_2, out_img);
    cv::cvtColor(out_img, out_img, CV_GRAY2RGB);

    for(int i = 0; i < matched_2d_cur.size(); i++) // kf1 lm2d
    {
      cv::Point2f cur_pt = matched_2d_cur.at(i);
      cv::circle(out_img, cur_pt, 5, cv::Scalar(0, 255, 0));
    }
    for(int i = 0; i < matched_2d_old.size(); i++)  //kf0 lm2d_post
    {
      cv::Point2f old_pt = matched_2d_old.at(i);
      old_pt.x += offset;
      cv::circle(out_img, old_pt, 5, cv::Scalar(0, 255, 0));
    }
    for(int i = 0; i < matched_2d_cur.size(); i++)
    {
      cv::Point2f old_pt = matched_2d_old.at(i);
      old_pt.x += offset;
      cv::line(out_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
    }
    std::string save_path;
    save_path = "/home/yurong/Loop/COVIS/loop_image/"
        + to_string(kf_cur->keyframe_id) + "-" + to_string(kf_old->keyframe_id) + "_1matched.png";
    cv::imwrite(save_path, out_img);

  }
#endif

  if(matched_2d_cur.size() < lc_paras.minPts)
  {
    return is_lc;
  }


  status.clear();
  if(matched_2d_cur.size() >= lc_paras.minPts)
  {
    status.clear();
    cv::Mat r_ = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat t_ = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat inliers;

    tic_toc_ros t_ransac;
    size_t cam_id = kf_old->AgentId_;
    cv::solvePnPRansac(matched_3d, matched_2d_old, d_cameras[cam_id].K0_rect, d_cameras[cam_id].D0_rect,
                         r_, t_, false, 100, 2.0, 0.99, inliers, cv::SOLVEPNP_P3P);

    //ROS_DEBUG("\033[1;34m ransac time: %lf \033[0m ", t_ransac.dT_ms());


    for(int i = 0; i < matched_2d_old.size(); i++)
      status.push_back(0);

    for(int i = 0; i < inliers.rows; i++)
    {
      int n = inliers.at<int>(i);
      status[n] = 1;
    }

    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_3d, status);

    static int count = 0;
    static int loop_count = 0;
#if 0
    if (SAVE_IMG && !matched_2d_cur.empty())
    {
          //ROS_DEBUG("\033[1;32m potential loop count %d \033[0m ", count ++ );
          int gap = 10;
          cv::Mat gap_image(kf_cur->img0.rows, gap, CV_8UC1, cv::Scalar(255, 255, 255));
          cv::Mat gray_img, loop_match_img;
          cv::Mat old_img = kf_old->img0;
          cv::hconcat(kf_cur->img0, gap_image, gap_image);
          cv::hconcat(gap_image, old_img, gray_img);
          cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
          for(int i = 0; i< (int)matched_2d_cur.size(); i++)
          {
              cv::Point2f cur_pt = matched_2d_cur[i];
              cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
          }
          for(int i = 0; i< (int)matched_2d_old.size(); i++)
          {
              cv::Point2f old_pt = matched_2d_old[i];
              old_pt.x += (kf_cur->img0.cols + gap);
              cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
          }
          for (int i = 0; i< (int)matched_2d_cur.size(); i++)
          {
              cv::Point2f old_pt = matched_2d_old[i];
              old_pt.x += (kf_cur->img0.cols + gap) ;
              cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
          }
          cv::Mat notation(50, kf_cur->img0.cols + gap + kf_cur->img0.cols, CV_8UC3, cv::Scalar(255, 255, 255));
          putText(notation, "current frame: " + to_string(kf_cur->keyframe_id), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
          putText(notation, "previous frame: " + to_string(kf_old->keyframe_id), cv::Point2f(20 + kf_cur->img0.cols + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
          cv::vconcat(notation, loop_match_img, loop_match_img);


          string save_path;
          save_path = "/home/yurong/Loop/COVIS/pnp/"
              + to_string(kf_cur->keyframe_id) + "-" + to_string(kf_old->keyframe_id) + "_2pnp.png";
          cv::imwrite(save_path, loop_match_img);
    }
#endif
    if(inliers.rows < lc_paras.minPts)
    {
      //ROS_DEBUG("\033[1;31m inliers not enough kf: %ld num: %d \033[0m " ,(kf_cur->keyframe_id), inliers.rows);
      return is_lc;
    }


    //ROS_WARN("\033[1;31m found more than %d matched points %ld \033[0m ",lc_paras.minPts, (kf_cur->keyframe_id));
    //ROS_WARN("\033[1;31m effective loop %d \033[0m ", loop_count++);

    //SE3 se_ij

    se_ji = SE3_from_rvec_tvec(r_,t_).inverse();

    Vec3 rpy;
    rpy= Q2rpy(se_ji.unit_quaternion());

    if(se_ji.translation().norm() < 20.0 && se_ji.so3().log().norm() < 1.5)
      is_lc = true;


  }

//#if 0
  if(is_lc)
  {
    if(SAVE_IMG)
    {
      ///publish loop closure frame
      cv::Mat img_0 = kf_cur->img0;
      cv::Mat img_1 = kf_old->img0;
      if(img_0.size != img_1.size)
      {
         cout << "hconcat cannot be used " << endl;
      }
      int gap = 20;
      int offset = img_0.cols + gap;

      cv::Mat gap_img(img_0.rows, gap, CV_8UC1, cv::Scalar(0, 0, 0));
      cv::Mat out_img, tmp_img;
      cv::hconcat(img_0, gap_img, gap_img);
      cv::hconcat(gap_img, img_1, out_img);
      cv::cvtColor(out_img, out_img, CV_GRAY2RGB);

      for(int i = 0; i < matched_2d_cur.size(); i++)
      {
        cv::Point2f cur_pt = matched_2d_cur.at(i);
        cv::circle(out_img, cur_pt, 5, cv::Scalar(0, 255, 0));
      }

      for(int i = 0; i < matched_2d_old.size(); i++)
      {
        cv::Point2f old_pt = matched_2d_old.at(i);
        old_pt.x += offset;
        cv::circle(out_img, old_pt, 5, cv::Scalar(0, 255, 0));
      }

      for(int i = 0; i < matched_2d_cur.size(); i++)
      {
        cv::Point2f old_pt = matched_2d_old.at(i);
        old_pt.x += offset;
        cv::line(out_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
      }


      cv::Mat info = cv::Mat::zeros(cv::Size(img_0.cols + gap + img_1.cols, 4*gap), CV_8UC3);
      Vec3 rpy, t;
      rpy= Q2rpy(se_ji.unit_quaternion());
      t = se_ji.translation();

      cv::putText(info, "yaw: " + to_string(rpy[2]*57.2958), cv::Point2f(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 3);
      cv::putText(info, "current frame: " + to_string(kf_cur->keyframe_id) + " Agent: " + to_string(kf_cur->AgentId_), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
      cv::putText(info, "previous frame: " + to_string(kf_old->keyframe_id) + " Agent: " + to_string(kf_old->AgentId_), cv::Point2f(20 + offset, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
      cv::vconcat(info, out_img, out_img);
#if 1
      std::string save_path;
      save_path = "/home/yurong/Loop/COVIS/loop/"
          + to_string(kf_cur->keyframe_id) + "-" + to_string(kf_old->keyframe_id) + "_3loop.png";
      cv::imwrite(save_path, out_img);
#endif
      cv::Mat vis_img;
      cv::resize(out_img, vis_img, cv::Size(out_img.cols/2, out_img.rows/2));
      sensor_msgs::ImagePtr lc_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_img).toImageMsg();
      merge_Img_pub.publish(lc_msg);

    }
  }
//# endif
  return is_lc;

}

void Merging::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                      std::vector<uchar> &status,
                      const std::vector<BRIEF::bitset> &point_des,
                      const std::vector<BRIEF::bitset> &descriptors_old,
                      const std::vector<cv::Point2f> &feature_2d_old)
{
  for(int i = 0; i < (int)point_des.size(); i++)
  {
      cv::Point2f pt(0.f, 0.f);
      if (searchInAera(point_des[i], descriptors_old, feature_2d_old, pt))
        status.push_back(1);
      else
        status.push_back(0);
      matched_2d_old.push_back(pt);
  }
}
bool Merging::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::Point2f> &feature_2d_old,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
      best_match_norm = feature_2d_old[bestIndex];
      return true;
    }
    else
      return false;
}

int Merging::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void Merging::updateGlobalPose(shared_ptr<KeyFrameMerge> kf_loop, shared_ptr<KeyFrameMerge> kf_curr, SE3& se_ji)
{
  bool need_update = false;
  //cout << "kf loop: " << kf_loop->AgentId_ << " " << kf_loop->keyframe_id << endl;
  //cout << "kf curr: " << kf_curr->AgentId_ << " " << kf_curr->keyframe_id << endl;
  m_drift.lock();
  /* local:curr -> global: loop */
  if(kf_loop->AgentId_ != kf_curr->AgentId_ && merged_agent[kf_loop->AgentId_] == true && merged_agent[kf_curr->AgentId_] == false)
  {
    ROS_WARN_STREAM("Merge " << kf_curr->AgentId_ << " to " << kf_loop->AgentId_ );
    merged_agent[kf_curr->AgentId_] = true;
    SE3 T_loop, T_curr;
    T_loop = kf_loop->T_c_w;
    T_curr = kf_curr->T_c_w;

    SE3 T_loop_curr = T_curr.inverse() * se_ji* T_loop;
    T_local_global[kf_curr->AgentId_] = T_loop_curr;

    m_vector.lock();
    for(size_t i = 0; i < kfs_all.size(); i++)
    {
      if(kfs_all[i]->AgentId_ == kf_curr->AgentId_)
      {
        SE3 T_local = kfs_all[i]->T_c_w;
        kfs_all[i]->T_c_w_global = T_local * T_loop_curr;
        need_update = true;
      }
    }
    m_vector.unlock();

  }
  /*local: loop -> global: curr */
  if(kf_loop->AgentId_ != kf_curr->AgentId_ && merged_agent[kf_loop->AgentId_] == false && merged_agent[kf_curr->AgentId_] == true)
  {
    ROS_WARN_STREAM("Merge " << kf_loop->AgentId_ << " to " << kf_curr->AgentId_);
    merged_agent[kf_loop->AgentId_] = true;
    SE3 T_loop, T_curr;
    T_loop = kf_loop->T_c_w;
    T_curr = kf_curr->T_c_w;

    SE3 T_curr_loop = T_loop.inverse() * se_ji.inverse() * T_curr ; // T_global_local
    T_local_global[kf_loop->AgentId_] = T_curr_loop;

    m_vector.lock();
    for(size_t i = 0; i < kfs_all.size(); i++)
    {
      if(kfs_all[i]->AgentId_ == kf_loop->AgentId_)
      {
        SE3 T_local = kfs_all[i]->T_c_w;
        kfs_all[i]->T_c_w_global =  T_local * T_curr_loop;
        need_update = true;
      }
    }
    m_vector.unlock();

  }

  if(need_update)
  {
    m_path.lock();
    // clear old pose and pub shifted poses
    for(size_t i = 0; i < globalpaths_.size(); i++)
      globalpaths_[i]->clearPath();

    for(size_t i = 0; i < kfs_all.size(); i++)
    {
      auto kf = kfs_all[i];
      globalpaths_[kf->AgentId_]->pubPathT_w_c((kf->T_c_w_global).inverse(),kf->t, kf->AgentId_);
    }
    m_path.unlock();

  }
  m_drift.unlock();



}


bool Merging::add_Loop_check(shared_ptr<KeyFrameMerge> kf_loop, shared_ptr<KeyFrameMerge> kf_curr, SE3 &loop_pose)
{
  std::unique_lock<std::mutex> LoopLock(m_loop);
  if (!IntraLoop)
  {
    if(kf_loop->AgentId_ != kf_curr->AgentId_)  //Disable intra_LOOP_Closure
    {
      this->loop_ids.push_back(Vec3I(static_cast<int>(kf_loop->keyframe_id), static_cast<int>(kf_curr->keyframe_id), 1));
      this->loop_poses.push_back(loop_pose);
      //int thre = 1;
      int thre = static_cast<int>((static_cast<double>(kf_id)/100)*2);
      //  cout << "thre " << thre << endl;
      //  cout << "curr_id: " << (kf_curr->keyframe_id) << endl;
      //  cout << "last_pgo_id: " << last_pgo_id << endl;
      //  cout << "equal: " << static_cast<int>(kf_curr->keyframe_id) - static_cast<int>(last_pgo_id) << endl;
      //  cout << endl;
      if(static_cast<int>(kf_curr->keyframe_id) - static_cast<int>(last_pgo_id) < thre)
      {
        ROS_DEBUG("Last loop is too close");
        return true;
      }
      else
      {
        last_pgo_id = static_cast<int>(kf_curr->keyframe_id);
        return false;
      }
    }
    else
    {
      return true;
    }


  }
  else
  {
    this->loop_ids.push_back(Vec3I(static_cast<int>(kf_loop->keyframe_id), static_cast<int>(kf_curr->keyframe_id), 1));
    this->loop_poses.push_back(loop_pose);
    //int thre = 1;
    int thre = static_cast<int>((static_cast<double>(kf_id)/100)*2);
    //  cout << "thre " << thre << endl;
    //  cout << "curr_id: " << (kf_curr->keyframe_id) << endl;
    //  cout << "last_pgo_id: " << last_pgo_id << endl;
    //  cout << "equal: " << static_cast<int>(kf_curr->keyframe_id) - static_cast<int>(last_pgo_id) << endl;
    //  cout << endl;
    if(static_cast<int>(kf_curr->keyframe_id) - static_cast<int>(last_pgo_id) < thre)
    {
      ROS_DEBUG("Last loop is too close");
      return true;
    }
    else
    {
      last_pgo_id = static_cast<int>(kf_curr->keyframe_id);
      return false;
    }
  }


}


void Merging::startPGO()
{
  while(true)
  {
    int64_t id = -1;
    static int cnt = 0;
    m_pgo.lock();
    if(!graph_queue.empty())
    {
      id = graph_queue.front();
      graph_queue.pop();

    }
    m_pgo.unlock();
    if(id != -1)
    {
      //      std::unique_lock<std::mutex> lock(m_pgo);
      //      ROS_INFO("\033[1;32m waiting. \033[0m");

      //      cv_pgo.wait(lock, [&]{return (pgo_running_);});
      ROS_WARN("\033[1;32m Start optimize. %ld \033[0m", id);
      //#if 0
      if(pgo_running_)
      {
        int time_interval = 1000;
        std::this_thread::sleep_for(std::chrono::milliseconds(time_interval));
        auto opt_start = chrono::high_resolution_clock::now();
        PoseGraphOptimization();
        auto opt_end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(opt_end - opt_start);
        ROS_WARN("\033[1;32m optimize time: %f s \033[0m \n", (duration.count()/1000.0));
        pgo_running_ = false;
        ROS_WARN("\033[1;32m Finish optimize. Optimize %d times \033[0m", ++cnt);
      }
      //#endif
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

}

void Merging::PoseGraphOptimization()
{
  m_vector.lock();

  uint64_t kf_prev_idx = 2 * kfs_all.size();
  uint64_t kf_curr_idx = 0;
  //  cout << "kf_prev_idx: 2 times size: " << kf_prev_idx << endl;
  //  cout << "kf_curr_idx: " << kf_curr_idx << endl;
  m_loop.lock();
  //cout << "loop_ids: size is " << loop_ids.size() << endl;

  //  kf_prev_idx: the smallest idx in loop_ids
  //  kf_curr_idx: the largest  idx in loop_ids
  for(size_t i = 0; i < loop_ids.size(); i++)
  {
    //cout << i << endl;
    //cout << loop_ids[i].transpose() << endl;
    if(loop_ids[i](0) < static_cast<int>(kf_prev_idx))
    {
      //cout << loop_ids[i](0) << "<" << kf_prev_idx << endl;
      kf_prev_idx = static_cast<uint64_t>(loop_ids[i](0));
    }

    if(loop_ids[i](1) > static_cast<int>(kf_curr_idx))
    {
      //cout << loop_ids[i](1) << ">" << kf_curr_idx << endl;
      kf_curr_idx = static_cast<uint64_t>(loop_ids[i](1));
    }

  }
  for(auto id:loop_ids){
    cout << id.transpose() << endl;
  }
  //  cout << "earliest index: " << kf_prev_idx << endl;
  //  cout << "current index: " << kf_curr_idx << endl;


  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);


  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver(new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>());
  std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr(new g2o::BlockSolver_6_3(std::move(linearSolver)));
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

  solver->setUserLambdaInit(1e-10);
  optimizer.setAlgorithm(solver);

  //noticed g2o pgo use swi swj as vertices, where swi transform p from i to w
  //use sij as edge
  //error function is sij^-1 * (swi^-1*swj)
  //so we need to use Tcw.inverse() as vertices, and sji.inverse as edge

  // grab the KFs included in the optimization

  //cout << "start to insert vertices "<<endl;
  vector<int> kf_list;
  for (size_t i = kf_prev_idx; i <= kf_curr_idx; i++)
  {
    if (kfs_all[i] != nullptr)
    {
      // check if it is a LC vertex
      //bool is_lc_i = false;
      bool is_lc_j = false;
      int id = 0; // not used
      for (auto it = loop_ids.begin(); it != loop_ids.end(); it++, id++)
      {
        //check if the first element equal to kf_prev_idx
        if ((*it)(0) == static_cast<int>(i))
        {
          //is_lc_i = true;
          break;
        }
        if ((*it)(1) == static_cast<int>(i))
        {

          is_lc_j = true;
          break;
        }
      }
      kf_list.push_back(static_cast<int>(i));
      // create SE3 vertex
      g2o::VertexSE3 *v_se3 = new g2o::VertexSE3();
      v_se3->setId(static_cast<int>(i));
      v_se3->setMarginalized(false);

      // _inverseMeasurement * from->estimate().inverse() * to->estimate();
      // z^-1 * (x_i^-1 * x_j)

      SE3 siw = kfs_all[i]->T_c_w_global.inverse();

      if (is_lc_j)
      {
        // update pose of LC vertex
        v_se3->setFixed(false);
        v_se3->setEstimate(SE3_to_g2o(siw));

      }
      else
      {
        v_se3->setEstimate(SE3_to_g2o(siw));
        //if ((is_lc_i && loop_ids.back()[0] == i) || i == 0)
        if(i == 0 || i == kf_prev_idx) /// first pose set fix
        {
          v_se3->setFixed(true);
        }
        else
        {
          v_se3->setFixed(false);
        }
      }
      optimizer.addVertex(v_se3);
    }
  }
  //cout<<"start to insert adjacent edge "<<endl;


  // introduce edges
  for (size_t i = kf_prev_idx; i <= kf_curr_idx; i++)
  {
    int cnt = 0;
    for (size_t j = i + 1; j <= kf_curr_idx;  j++)
    {
      if (kfs_all[i] != nullptr && kfs_all[j] != nullptr && cnt<5 && kfs_all[i]->AgentId_ == kfs_all[j]->AgentId_)
      {
        // kf2kf constraint
        SE3 sji = kfs_all[j]->T_c_w_global*kfs_all[i]->T_c_w_global.inverse();//se_jw*se_iw.inverse();
        SE3 sij = sji.inverse();
        // add edge
        g2o::EdgeSE3* e_se3 = new g2o::EdgeSE3();

        e_se3->setVertex(0, optimizer.vertex(static_cast<int>(i)));
        e_se3->setVertex(1, optimizer.vertex(static_cast<int>(j)));
        e_se3->setMeasurement(SE3_to_g2o(sij));
        e_se3->setInformation(Matrix6d::Identity());
        e_se3->setRobustKernel(new g2o::RobustKernelCauchy());
        //RobustKernelHuber());
        optimizer.addEdge(e_se3);
        cnt++;
      }
    }
  }

  //cout<<"start to insert loop edge "<<endl;

  // introduce loop closure edges
  uint64_t id = 0;
  for (auto it = loop_ids.begin(); it != loop_ids.end(); it++, id++)
  {
    // add edge
    g2o::EdgeSE3 *e_se3 = new g2o::EdgeSE3();
    e_se3->setVertex(0, optimizer.vertex((*it)(0)));
    e_se3->setVertex(1, optimizer.vertex((*it)(1)));

    //edge_merge_pub->pubLoopEdge(kfs_all[(*it)(0)]->T_c_w_global.inverse(), kfs_all[(*it)(1)]->T_c_w_global.inverse());

    SE3 loop_pose = loop_poses[id].inverse();
    //    cout << "LC vertex i index: " << (*it)(0) << endl;
    //    cout << "LC vertex j index: " << (*it)(1) << endl;
    //    auto result = loop_pose.inverse() * (kfs_all[(*it)(0)]->T_c_w_global * kfs_all[(*it)(1)]->T_c_w_global.inverse());
    //    cout << "z^-1:" << loop_pose.inverse() << endl;
    //    cout << "Mul : " << kfs_all[(*it)(0)]->T_c_w_global * kfs_all[(*it)(1)]->T_c_w_global.inverse() << endl;
    //    cout << "result: " << result << endl;
    //    cout << "translation: " << result.translation().norm() << " " << result.so3().log().norm() << endl;
    e_se3->setMeasurement(SE3_to_g2o(loop_pose));
    e_se3->information() = Matrix6d::Identity();
    e_se3->setRobustKernel(new g2o::RobustKernelCauchy());
    optimizer.addEdge(e_se3);
  }
  m_loop.unlock();
  m_vector.unlock();


  //optimizer.edges();
  optimizer.save("/home/yurong/covis_before.g2o");
  // optimize graph

  optimizer.initializeOptimization();
  optimizer.computeInitialGuess();
  optimizer.computeActiveErrors();
  optimizer.optimize(5);

  optimizer.save("/home/yurong/covis_after.g2o");


  m_vector.lock();
  // calculate drift
  map<size_t, size_t> last_AgentKfIds;
  for(auto kf_it = kf_prev_idx; kf_it <= kf_curr_idx; kf_it++)
  {
    shared_ptr<KeyFrameMerge> kf = kfs_all[kf_it];
    last_AgentKfIds[kf->AgentId_] = kf_it;
  }
  //  cout << "last agentId: " << endl;
  //  for(auto it:last_AgentKfIds)
  //  {
  //      cout << it.first << " " << it.second << endl;
  //  }
  map<size_t, size_t>::iterator it;
  for(it = last_AgentKfIds.begin(); it != last_AgentKfIds.end(); it++)
  {
    size_t AgentId = it->first;
    size_t kf_id = it->second;
    SE3 Tcw1 = kfs_all[kf_id]->T_c_w_global;
    g2o::VertexSE3 * v_se3 = static_cast<g2o::VertexSE3 *>(optimizer.vertex(static_cast<int>(kf_id)));
    g2o::SE3Quat Twc_g2o = v_se3->estimateAsSE3Quat();
    SE3 Twc2 = SE3_from_g2o(Twc_g2o);
    SE3 Tw2_w1 = Twc2*Tcw1;
    SE3 Tw1_w2 = Tw2_w1.inverse();
    m_drift.lock();
    drifts[AgentId] = drifts[AgentId] * Tw1_w2;
    m_drift.unlock();
    //cout << "Agent " << AgentId << " " << kf_id << " drift " << drifts[AgentId].translation().transpose() << endl;
  }
  m_vector.unlock();


  m_vector.lock();
  for (auto kf_it = kf_prev_idx; kf_it <= kf_curr_idx; kf_it++)
  {
    //cout << "update: " << kf_it << endl;
    g2o::VertexSE3 *v_se3 = static_cast<g2o::VertexSE3 *>(optimizer.vertex(static_cast<int>(kf_it)));
    g2o::SE3Quat Twc_g2o = v_se3->estimateAsSE3Quat();
    SE3 Twc2 = SE3_from_g2o(Twc_g2o);  // after optimize

    shared_ptr<KeyFrameMerge> kf = kfs_all[kf_it];
    SE3 Tcw1 = kf->T_c_w_global; // before optimize

    //loop candidate frame fix, Tw2_w1 should be identity matrix
    SE3 Tw2_w1 =Twc2*Tcw1;// transform from previous to current from odom to map
    SE3 Tw1_w2 = Tw2_w1.inverse();

    kf->T_c_w_global = Twc2.inverse();

    if(kf_it == kf_prev_idx || kf_it == kf_curr_idx)
    {
      //      cout << "before optimize: " << Tcw1 << endl;
      //      cout << "after  optimize: " << Twc2.inverse() << endl;
    }

  }
  m_vector.unlock();

  // recover pose and update map
  m_vector.lock();
  m_path.lock();
  for(size_t i = 0; i < globalpaths_.size(); i++)
    globalpaths_[i]->clearPath();

  m_drift.lock();
  for(size_t i = kf_curr_idx+1; i < kfs_all.size(); i++)
  {
    //cout << "update new coming frame: " << i << endl;
    kfs_all[i]->T_c_w_global = kfs_all[i]->T_c_w_odom * T_local_global[kfs_all[i]->AgentId_] * drifts[kfs_all[i]->AgentId_];
  }
  m_drift.unlock();
  for(size_t i = 0; i < kfs_all.size(); i++)
  {
    shared_ptr<KeyFrameMerge> kf = kfs_all[i];
    globalpaths_[kf->AgentId_]->pubPathT_w_c((kf->T_c_w_global).inverse(),kf->t, kf->AgentId_);
    dronepub_[kf->AgentId_]->PubT_w_i(kf->T_w_i, kf->t, kf->AgentId_);

  }
  m_path.unlock();
  m_vector.unlock();

  //publish Loop edge

  m_vector.lock();
  m_loop.lock();
  edge_merge_pub->clearEdge();
  for (auto it = loop_ids.begin(); it != loop_ids.end(); it++)
  {
    auto kf_from = kfs_all[(*it)(0)];
    auto kf_to = kfs_all[(*it)(1)];
    edge_merge_pub->AddLoopEdge(kf_from->T_c_w_global.inverse(), kf_to->T_c_w_global.inverse(), kf_to->t);
  }
  edge_merge_pub->PubEdge();
  m_loop.unlock();
  m_vector.unlock();


}

#if 0

void Merging::expandGraph()
{

  if(!kfs_this.empty() &&!kfs_other.empty()&& !kfs_all.empty())
  {
    size_t m = kfs_this.size();
    size_t n = kfs_other.size();
    size_t l = kfs_all.size();

    //  cout << "this number: "  << m << endl;
    //  cout << "other number: "   << n << endl;
    //  cout << "all number: "    << l << endl;
    //  cout << endl;

    //sim_vec.resize(n);
    sim_vec.clear();
    sim_vec.resize(l);
    sim_vec_covis.resize(static_cast<size_t>(lc_paras.lcKFDist));
    sim_vec_covis.clear();
    // sim matrix l rows l cols
    sim_matrix.resize(l);
    for (size_t i = 0; i < l; i++)
      sim_matrix[i].resize(l);

  }


}

//sim_vec: [s(kf0,kf_curr) s(kf1,kf_curr) ...  s(kf_curr, kf_curr)]

void Merging::sim_mat_update()
{
  tic_toc_ros simgraph_tic;
  //STEP3 search similar image in the list
  for(int i = 0; i < kfs_all.size(); i++)
  {
    if(kfs_all[i] != nullptr && kfs_all[i]->AgentId_ != kfs_all.back()->AgentId_)
    {
      double score = voc.score(kfs_all[i]->kf_bv, kfs_all.back()->kf_bv);
      sim_matrix[kfs_all.size()-1][i] = score;
      sim_matrix[i][kfs_all.size()-1] = score;

      sim_vec[i] = score;
    }
    else
    {
      sim_matrix[kfs_all.size()-1][i] = 0;
      sim_matrix[i][kfs_all.size()-1] = 0;
      sim_vec[i] = 0;
    }

  }
  printf("\033[1;32m sim graph time: %lf \033[0m \n", simgraph_tic.dT_ms());

  tic_toc_ros vec_tic;
  for(int i = kfs_this.size() - lc_paras.lcKFDist; i < kfs_this.size(); i++)
  {
    if(!kfs_this.empty() && kfs_this.size() >= sim_vec_covis.size())
    {
      double score = voc.score(kfs_this.back()->kf_bv, kfs_this[i]->kf_bv);
      sim_vec_covis.push_back(score);
    }
  }
  printf("\033[1;32m vec_tic time: %lf \033[0m \n", vec_tic.dT_ms());



  // print new sim_matrix
  //  cout << "matrix: " << endl;
  //  for(uint64_t i = 0; i < kfs_all.size(); i++)
  //  {
  //    for(uint64_t j = 0; j < kfs_all.size(); j++)
  //    {
  //      cout<<sim_matrix[i][j]<<" ";
  //    }
  //    cout<<endl;
  //  }
  //  cout << "vector: " << endl;
  //  for(auto v:sim_vec){
  //    cout << v << " ";
  //  }
}



bool Merging::isMergeCandidate()
{
  cv::Mat compressed_image;
  auto kf = this->kfs_all.back();

  // Visualization: All images are insert to LoopImgs
  if(DEBUG_IMG)
  {
    int feature_num = kf->lm_count;
    idpair ip = make_pair(kf->AgentId_,kf->keyframe_id);
    cv::resize(idpairImg.find(ip)->second, compressed_image, cv::Size(360, 240));
    cv::cvtColor(compressed_image, compressed_image, CV_GRAY2RGB);
    cv::putText(compressed_image,"AgentID: " + to_string(kf->AgentId_) + " Index: " + to_string(kf->keyframe_id) + " feature num:" + to_string(feature_num), cv::Point2f(15, 15), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,255));
    LoopImgs.insert(make_pair(kf->keyframe_id, compressed_image));
  }

  bool is_merge_candidate = false;
  size_t g_size = kfs_all.size();
  if(g_size < 2 * lc_paras.lcKFStart)
  {
    cout << "kf number less than 150. Return " << endl;
    return is_merge_candidate;
  }

  uint64_t sort_index;
  if((g_size - lc_paras.lcKFDist) > 5000)
    sort_index = g_size - lc_paras.lcKFDist - 5000;
  else
    sort_index = 0;

  vector<std::pair<double, shared_ptr<KeyFrameMerge>>> vScoreKFs; // score of agent's CurrKF and other agents' KF
  for (uint64_t j = sort_index; j < static_cast<uint64_t>(g_size); j++)
  {
    if(kfs_all[j]!= nullptr && sim_vec[j] != 0.0)
    {
      vScoreKFs.push_back(make_pair(sim_vec[j], kfs_all[j]));
    }
  }
  if(vScoreKFs.empty()) // all score is 0 (same AgentId or score is -0)
  {
    return is_merge_candidate;
  }
  sort(vScoreKFs.rbegin(),vScoreKFs.rend()); //sort container by score

  cv::Mat loop_result;
  if(DEBUG_IMG)
  {
    loop_result = compressed_image.clone();
    for(int i = 0; i < 4; i++)
    {
      idpair idpair = make_pair(kf->AgentId_,kf->keyframe_id);
      int tmp_idx = vScoreKFs[i].second->keyframe_id;
      auto it = LoopImgs.find(tmp_idx);
      cv::Mat tmp_img = it->second.clone();
      cv::putText(tmp_img, "Score: " + to_string(vScoreKFs[i].first) , cv::Point2f(15, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,255));
      cv::hconcat(loop_result, tmp_img, loop_result);
    }
    std::string image_path;
    image_path = "/home/yurong/LoopCandidate/" + to_string(kf->keyframe_id) + ".png";
    cv::imwrite(image_path, loop_result);
  }
  if(DEBUG_IMG)
  {
    cv::imshow("loop_candidate", loop_result);
    cv::waitKey(20);
  }

  // find the minimum score in the covisibility graph (and/or 3 previous keyframes)
  double lc_min_score = 1.0;
  for (int i = 0; i < sim_vec_covis.size(); i++)
  {
    double score = sim_vec_covis[i];

    if (score < lc_min_score && score > 0.001) lc_min_score = score;
  }
  lc_min_score = min(lc_min_score, 0.4);


  if( vScoreKFs[0].first < max(lc_paras.minScore, lc_min_score))
  {
    return is_merge_candidate;
  }
  else
  {

  }

  int idx_max = int(vScoreKFs[0].second->keyframe_id);
  if(idx_max == this->kf_prev_idx_)
  {
    //cout << "\033[1;31m repetitive LC detection, ignore  \033[0m" << endl;
    return is_merge_candidate;
  }
  int nkf_closest = 0;
  if(vScoreKFs[0].first >= lc_min_score)
  {
    for (uint64_t j = 1; j < vScoreKFs.size(); j++)
    {
      int idx = int(vScoreKFs[j].second->keyframe_id);
      if(abs(idx - idx_max) <= lc_paras.lcKFMaxDist && vScoreKFs[j].first >= lc_min_score * 0.8)
      {
        //cout << "idx: " << idx << endl;
        nkf_closest++;
      }
    }
  }
  // update in case of being loop closure candidate
  if (nkf_closest >= lc_paras.lcNKFClosest && vScoreKFs[0].first > lc_paras.minScore)
  {
    is_merge_candidate = true;
    this->kf_prev_idx_ = static_cast<uint64_t>(idx_max);

  }
  // ****************************************************************** //
  //  cout << endl
  //       << "\033[1;32m lc_min_score: \033[0m " << lc_min_score;
  //  cout << endl
  //       << "\033[1;32m Nkf_closest: \033[0m  " << nkf_closest;
  //  cout << endl
  //       << "\033[1;32m idx_max: \033[0m " << idx_max << endl;
  //  cout << "\033[1;32m max score of previous kfs: \033[0m "<< vScoreKFs[0].first << endl;
  //  cout << endl;
  return is_merge_candidate;
}
bool Merging::check_lastLC_close()
{
  int thre = static_cast<int>((static_cast<double>(kf_id)/100)*2);
  //  cout << "thre " << thre << endl;
  //  cout << "last_pgo_id: " << last_pgo_id << endl;
  //  cout << "equal: " << kf_curr_idx_ - static_cast<size_t>(last_pgo_id) << endl;
  //  cout << endl;
  if(kf_curr_idx_ - static_cast<size_t>(last_pgo_id) < thre)
  {
    cout<<"\033[1;31m Last loop is too close \033[0m " <<endl;
    return true;
  }
  else
    return false;
}
#endif


