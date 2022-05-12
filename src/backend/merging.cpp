#include "include/merging.h"

Merging::Merging(ros::NodeHandle &nh, LC_PARAS &lc_paras,DBoW3::Vocabulary &voc, DBoW3::Database &db,
                 DepthCamera &dc, int saveimg_flag, int number, bool Intra):
  lc_paras(lc_paras), voc(voc), db(db), dc(dc), SAVE_IMG(saveimg_flag), IntraLoop(Intra)
{
  number_of_Agent = static_cast<size_t>(number);
  ROS_WARN("Total number of Agent: %lu ", number_of_Agent);
  ROS_WARN("loop detection threshold: %f ", lc_paras.minScore);
  ROS_WARN("SaveImg: %d ", SAVE_IMG);
  ROS_WARN("IntraLoop: %d ", IntraLoop);

  // global path
  globalpaths_.resize(number_of_Agent);
  for(size_t i = 0; i < number_of_Agent; i++)
    globalpaths_[i] = new RVIZPath(nh, "/global_path_"+to_string(i), "map", 1, 10000);

  // inter-agent edge
  edge_merge_pub  = new RVIZEdge(nh, "/merge_edge", "map", 10000);

  // inter-agent matched image
  image_transport::ImageTransport it(nh);
  merge_Img_pub = it.advertise("/merge_img",1);

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


void Merging::setKeyFrameMerge(const covis::KeyFrameConstPtr& msg)
{
  tic_toc_ros msg_tic;
  //STEP1.1 Unpack
  // [1]kf.frame_id
  // [2]kf.T_c_w_odom

  KeyFrameMerge kf;

  cv::Mat img0_unpack, img1_unpack;
  vector<int64_t> lm_id_unpack;
  vector<Vec2> lm_2d_unpack;
  vector<Vec3> lm_3d_unpack;
  vector<cv::Mat> lm_descriptor_unpack;
  int lm_count_unpack;
  KeyFrameMsg::unpack(msg, kf.frame_id, kf.AgentId_, img0_unpack, img1_unpack, lm_count_unpack,
                      lm_id_unpack,lm_2d_unpack, lm_3d_unpack, lm_descriptor_unpack, kf.T_c_w_odom, kf.t);




  //STEP1.2 Construct KeyFrameLC
  // [1]kf.T_c_w
  // [2]kf.keyframe_id

  BowVector kf_bv;

  kf.keyframe_id = kf_id++;

  std::string train_path;
  train_path = "/home/yurong/Pre_training/" + to_string(kf.keyframe_id) + ".png";
  //cout << "writing " << train_path << endl;
  //cv::imwrite(train_path, img0_unpack);

  if(kf.keyframe_id==0)
  {
    merged_agent[kf.AgentId_] = true;
    ROS_WARN("first register Agent: %lu \n", kf.AgentId_);

  }

  m_imgs.lock();
  idpair idp = make_pair(kf.AgentId_, kf.keyframe_id);
  idpairImg.insert(make_pair(idp, img0_unpack));
  m_imgs.unlock();

  //  if(SAVE_LOOP)
  //  {
  //    for(int i = 0; i < 3; i++)
  //    {
  //      ofstream loop_path_file(lc_paras.ResultPaths[i], ios::app);
  //      loop_path_file.close();
  //    }

  //  }

  //STEP1.3 Extract-Compute_descriptors
  // ORB feature size: 500
  // Descriptors size: 500 x 32


  vector<cv::KeyPoint> ORBFeatures;
  cv::Mat ORBDescriptorsL;
  vector<cv::Mat> ORBDescriptors;
  ORBFeatures.clear();
  ORBDescriptors.clear();
  cv::Ptr<cv::ORB> orb = cv::ORB::create(500,1.2f,8,31,0,2, cv::ORB::HARRIS_SCORE,31,20);
  orb->detectAndCompute(img0_unpack,cv::Mat(),ORBFeatures,ORBDescriptorsL);
  descriptors_to_vecDesciptor(ORBDescriptorsL,ORBDescriptors);

  kf.lm_descriptor = ORBDescriptors;

  //STEP1.4 Construct KeyFrameLC
  //Compute bow vector
  //BowVector size: < 500      <WordId, WordValue>
  // [1]kf.kf_bv
  voc.transform(kf.lm_descriptor,kf_bv);
  kf.kf_bv = kf_bv;


  //STEP1.5 Construct KeyFrameLC
  //Recover 3d information
  // [1]kf.kf_bv
  vector<Vec3> lm_3d;
  vector<Vec2> lm_2d;
  vector<double> lm_d;
  vector<bool> lm_3d_mask;
  switch(dc.cam_type)
  {
  case STEREO_RECT:
  {
    //track to another image
    std::vector<cv::Point2f> lm_img0, lm_img1;
    vector<float>   err;
    vector<unsigned char> status;
    cv::KeyPoint::convert(ORBFeatures,lm_img0);
    lm_img1 = lm_img0;
    cv::calcOpticalFlowPyrLK(img0_unpack, img1_unpack,
                             lm_img0, lm_img1,
                             status, err, cv::Size(31,31),5,
                             cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.001),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    //triangulation
    for(size_t i=0; i<status.size(); i++)
    {
      if(status.at(i)==1)
      {
        Vec3 pt3d_c;
        if(Triangulation::trignaulationPtFromStereo(Vec2(lm_img0.at(i).x,lm_img0.at(i).y),
                                                    Vec2(lm_img1.at(i).x,lm_img1.at(i).y),
                                                    dc.P0_,
                                                    dc.P1_,
                                                    pt3d_c))
        {
          lm_2d.push_back(Vec2(lm_img0.at(i).x,lm_img0.at(i).y));
          lm_d.push_back(0.0);
          lm_3d.push_back(pt3d_c);
          lm_3d_mask.push_back(true);
          continue;
        }
        else
        {

          lm_2d.push_back(Vec2(0,0));
          lm_d.push_back(0.0);
          lm_3d.push_back(Vec3(0,0,0));
          lm_3d_mask.push_back(false);
          continue;
        }
      }else
      {
        lm_2d.push_back(Vec2(0,0));
        lm_d.push_back(0.0);
        lm_3d.push_back(Vec3(0,0,0));
        lm_3d_mask.push_back(false);
        continue;
      }
    }
    break;
  }
  case STEREO_UNRECT:
  {
    //track to another image
    std::vector<cv::Point2f> lm_img0, lm_img1;
    std::vector<cv::Point2f> lm_img0_undistort, lm_img1_undistort;
    vector<float>   err;
    vector<unsigned char> status;
    cv::KeyPoint::convert(ORBFeatures,lm_img0);
    lm_img1 = lm_img0;
    cv::calcOpticalFlowPyrLK(img0_unpack, img1_unpack,
                             lm_img0, lm_img1,
                             status, err, cv::Size(31,31),5,
                             cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.001),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    //go to undistor plane
    cv::undistortPoints(lm_img0, lm_img0_undistort, dc.K0, dc.D0, dc.R0, dc.P0);
    cv::undistortPoints(lm_img1, lm_img1_undistort, dc.K1, dc.D1, dc.R1, dc.P1);

    //triangulation
    for(size_t i=0; i<status.size(); i++)
    {
      if(status.at(i)==1)
      {
        Vec3 pt3d_c;
        if(Triangulation::trignaulationPtFromStereo(Vec2(lm_img0_undistort.at(i).x,lm_img0_undistort.at(i).y),
                                                    Vec2(lm_img1_undistort.at(i).x,lm_img1_undistort.at(i).y),
                                                    dc.P0_,
                                                    dc.P1_,
                                                    pt3d_c))
        {
          lm_2d.push_back(Vec2(lm_img0.at(i).x,lm_img0.at(i).y));
          lm_d.push_back(0.0);
          lm_3d.push_back(pt3d_c);
          lm_3d_mask.push_back(true);
          continue;
        }
        else
        {

          lm_2d.push_back(Vec2(0,0));
          lm_d.push_back(0.0);
          lm_3d.push_back(Vec3(0,0,0));
          lm_3d_mask.push_back(false);
          continue;
        }
      }else
      {
        lm_2d.push_back(Vec2(0,0));
        lm_d.push_back(0.0);
        lm_3d.push_back(Vec3(0,0,0));
        lm_3d_mask.push_back(false);
        continue;
      }
    }

    break;
  }
  case DEPTH_D435:
  {
    for(size_t i = 0; i<ORBFeatures.size();i++)
    {
      cv::Point2f cvtmp = ORBFeatures[i].pt;
      Vec2 tmp(cvtmp.x,cvtmp.y);
      double d = (img1_unpack.at<ushort>(cvtmp))/1000.0;
      if(d>=0.3&&d<=10)
      {
        Vec3 p3d((tmp.x()-dc.cam0_cx)/dc.cam0_fx*d,
                 (tmp.y()-dc.cam0_cy)/dc.cam0_fy*d,
                 d);
        lm_2d.push_back(tmp);
        lm_d.push_back(d);
        lm_3d.push_back(p3d);
        lm_3d_mask.push_back(true);
      }else
      {
        lm_2d.push_back(Vec2(0,0));
        lm_d.push_back(0.0);
        lm_3d.push_back(Vec3(0,0,0));
        lm_3d_mask.push_back(false);
      }
    }
    break;
  }
  }
  //STEP1.6 Construct KeyFrameLC
  // [1]kf.lm_2d
  // [2]kf.lm_d
  // [3]kf.lm_descriptor
  // [4]kf.lm_count
  //pass feature and descriptor
  kf.lm_2d = lm_2d;
  kf.lm_3d = lm_3d;
  kf.lm_depth = lm_d;
  for(int i=lm_3d_mask.size()-1; i>=0; i--)
  {
    if(lm_3d_mask.at(i)==false)
    {
      kf.lm_descriptor.erase(kf.lm_descriptor.begin()+i);
      kf.lm_2d.erase(kf.lm_2d.begin()+i);
      kf.lm_3d.erase(kf.lm_3d.begin()+i);
    }
  }

  kf.lm_count = static_cast<int>(kf.lm_2d.size());
  lm_2d.clear();
  lm_3d.clear();
  lm_d.clear();



  //STEP2 add kf to list
  m_vector.lock();
  kf.T_c_w =  kf.T_c_w_odom;

  m_drift.lock();
  kf.T_c_w_global = kf.T_c_w * T_local_global[kf.AgentId_] * drifts[kf.AgentId_];
  m_drift.unlock();

  shared_ptr<KeyFrameMerge> kf_ptr =std::make_shared<KeyFrameMerge>(kf);
  //cout << "latest kf: " << kf_ptr->keyframe_id << endl;

  m_path.lock();
  pubGlobalPath(kf_ptr);
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
  ROS_DEBUG("\033[1;31m msg time: %lf \033[0m", msg_tic.dT_ms());


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
      tic_toc_ros loop_time;
      int loop_index = -1;
      loop_index = AddandDetectLoop(kf_ptr);
      ROS_DEBUG("\033[1;32m detectloop time: %lf \033[0m ", loop_time.dT_ms());

      if(loop_index != -1)
      {
        //ROS_INFO("\033[1;32m find loop \033[0m ");

        SE3 loop_pose;
        bool is_lc = isLoopClosureKF(kfs_all.at(loop_index), kf_ptr, loop_pose);

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
  if(SAVE_IMG)
  {
    int feature_num = kf->lm_count;
    idpair ip = make_pair(kf->AgentId_, kf->keyframe_id);
    m_imgs.lock();
    cv::resize(idpairImg.find(ip)->second, compressed_image, cv::Size(360, 240));
    m_imgs.unlock();
    cv::cvtColor(compressed_image, compressed_image, CV_GRAY2RGB);
    cv::putText(compressed_image, "AgentID: " + to_string(kf->AgentId_) + " Index: " + to_string(kf->keyframe_id) + " feature num:" + to_string(feature_num), cv::Point2f(15, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,255));
    LoopImgs.insert(make_pair(kf->keyframe_id, compressed_image));
  }

  m_db.lock();
  DBoW3::QueryResults ret;
  db.query(kf->lm_descriptor, ret, 5, kf->keyframe_id - 30);
  //cout << "query: " << endl;
  //cout << ret << endl;

  db.add(kf->lm_descriptor);
  m_db.unlock();

  bool find_loop = false;
  cv::Mat loop_result;
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
    image_path = "/home/yurong/LoopCandidate/" + to_string(kf->keyframe_id) + ".png";

    cv::imwrite(image_path, loop_result);
  }

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

  if(find_loop && kf->keyframe_id > 0)
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


bool Merging::isLoopClosureKF(shared_ptr<KeyFrameMerge> kf0, shared_ptr<KeyFrameMerge> kf1, SE3 &se_ji)
{

  //  cout << "kf loop: " << kf0->AgentId_ << " " << kf0->keyframe_id << endl;
  //  cout << "kf curr: " << kf1->AgentId_ << " " << kf1->keyframe_id << endl;
  std::unique_lock<std::mutex> MatchLock(m_match);

  select_match.clear();
  bool is_lc = false;
  int common_pt = 0;

  vector<cv::DMatch> selected_matches_;
  if (!(kf1->lm_descriptor.size() == 0) && !(kf0->lm_descriptor.size() == 0))
  {

    cv::BFMatcher *bfm = new cv::BFMatcher(cv::NORM_HAMMING, false); // cross-check
    cv::Mat pdesc_l1= cv::Mat::zeros(cv::Size(32,static_cast<int>(kf0->lm_descriptor.size())),CV_8U);
    cv::Mat pdesc_l2= cv::Mat::zeros(cv::Size(32,static_cast<int>(kf1->lm_descriptor.size())),CV_8U);

    vector<vector<cv::DMatch>> pmatches_12, pmatches_21;
    // 12 and 21 matches
    vecDesciptor_to_descriptors(kf0->lm_descriptor,pdesc_l1);
    vecDesciptor_to_descriptors(kf1->lm_descriptor,pdesc_l2);

    bfm->knnMatch(pdesc_l1, pdesc_l2, pmatches_12, 2);
    bfm->knnMatch(pdesc_l2, pdesc_l1, pmatches_21, 2);


    // resort according to the queryIdx
    sort(pmatches_12.begin(), pmatches_12.end(), sort_descriptor_by_queryIdx());
    sort(pmatches_21.begin(), pmatches_21.end(), sort_descriptor_by_queryIdx());

    // bucle around pmatches

    vector<cv::Point3f> p3d;   // in camera frame
    vector<cv::Point2f> p2d;   // in pixel frame
    p3d.clear();
    p2d.clear();

    for (size_t i = 0; i < pmatches_12.size(); i++)
    {
      /// check if they are mutual best matches
      size_t lr_qdx = static_cast<size_t>(pmatches_12[i][0].queryIdx);
      size_t lr_tdx = static_cast<size_t>(pmatches_12[i][0].trainIdx);
      size_t rl_tdx = static_cast<size_t>(pmatches_21[lr_tdx][0].trainIdx);

      // check if they are mutual best matches and satisfy the distance ratio test
      if (lr_qdx == rl_tdx)
      {
        /// if closest distance is less than half of the second cloeset distance
        if(pmatches_12[i][0].distance * 1.0/pmatches_12[i][1].distance < static_cast<float>(lc_paras.ratioMax))
        {
          common_pt++;
          // save data for optimization
          Vector3d P0 = kf0->lm_3d[lr_qdx];
          //                        double d = kf0->lm_depth[lr_qdx];
          //                        Vector2d pl_map = kf0->lm_2d[lr_qdx];
          //                        double x = (pl_map(0)-dc.cam0_cx)/dc.cam0_fx*d;
          //                        double y = (pl_map(1)-dc.cam0_cy)/dc.cam0_fy*d;
          //                        Vector3d P0(x,y,d);
          Vector3f P = P0.cast<float>();
          Vector2f pl_obs = kf1->lm_2d[lr_tdx].cast<float>();
          cv::Point3f p3(P(0),P(1),P(2));
          cv::Point2f p2(pl_obs(0),pl_obs(1));
          p3d.push_back(p3);
          p2d.push_back(p2);
          selected_matches_.push_back(pmatches_12[i][0]);
        }

      }

    }
    cv::Mat r_ = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat t_ = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat inliers;

    if(p3d.size() < 5)
    {
      ROS_DEBUG("\033[1;31m p3d less 5 %ld \033[0m ",(kf1->keyframe_id));

      return is_lc;
    }

    cv::solvePnPRansac(p3d, p2d, dc.K0_rect, dc.D0_rect, r_, t_, false, 100, 2.0, 0.99, inliers, cv::SOLVEPNP_P3P);
    for( int i = 0; i < inliers.rows; i++){
      int n = inliers.at<int>(i);
      select_match.push_back(selected_matches_[n]);
    }
    //cout<<"selected points size: "<<p3d.size()<<" inliers size: "<<inliers.rows<<" unseletced size: "<<pmatches_12.size()<<endl;

    if(inliers.rows*1.0/p3d.size() < lc_paras.ratioRansac || inliers.rows < lc_paras.minPts ) //return is_lc;
    {
      ROS_DEBUG("\033[1;31m inliers not enough %ld \033[0m " ,(kf1->keyframe_id));
      return is_lc;
    }
    //SE3 se_ij
    se_ji = SE3_from_rvec_tvec(r_,t_);

    if(se_ji.translation().norm() < 10 && se_ji.so3().log().norm() < 10)
    {
      is_lc = true;
    }
    else
    {
      ROS_DEBUG("\033[1;31m translation or rotation too large %ld \033[0m ", (kf1->keyframe_id));
    }

    if(is_lc)
    {
      /*publish loop closure frame*/

      idpair kf0_idpair = make_pair(kf0->AgentId_,kf0->keyframe_id);
      idpair kf1_idpair = make_pair(kf1->AgentId_,kf1->keyframe_id);

      //m_imgs.lock();
      cv::Mat img_1 = idpairImg.find(kf0_idpair)->second;
      cv::Mat img_2 = idpairImg.find(kf1_idpair)->second;
      // m_imgs.unlock();

      cv::cvtColor(img_1, img_1, CV_GRAY2RGB);
      cv::cvtColor(img_2, img_2, CV_GRAY2RGB);

      /* check how many rows are necessary for output matrix */
      int totalRows = img_1.rows >= img_2.rows ? img_1.rows : img_2.rows;
      int gap = 15;
      int offset = img_1.cols + gap;
      cv::Mat outImg = cv::Mat::zeros(cv::Size(img_1.cols + gap + img_2.cols, totalRows), CV_8UC3);
      cv::Mat roi_left( outImg, cv::Rect( 0, 0, img_1.cols, img_1.rows) );
      cv::Mat roi_right( outImg, cv::Rect( offset, 0, img_2.cols, img_2.rows) );
      img_1.copyTo( roi_left );
      img_2.copyTo( roi_right );


      for ( size_t counter = 0; counter < select_match.size(); counter++ )
      {
        size_t lr_qdx = static_cast<size_t>(select_match[counter].queryIdx);
        size_t lr_tdx = static_cast<size_t>(select_match[counter].trainIdx);
        cv::Scalar matchColorRGB;
        matchColorRGB = cv::Scalar( 0, 255, 0 );
        Vector2f P_left  = kf0->lm_2d[lr_qdx].cast<float>();
        Vector2f P_right = kf1->lm_2d[lr_tdx].cast<float>();
        cv::Point2f p_left(P_left(0),P_left(1));
        cv::Point2f p_right(P_right(0) + offset, P_right(1));
        cv::line( outImg, p_left, p_right, matchColorRGB, 2 );
      }

      cv::Mat info = cv::Mat::zeros(cv::Size(img_1.cols + gap + img_2.cols, 4*gap), CV_8UC3);
      cv::putText(info, "inliers: " + to_string(inliers.rows), cv::Point2f(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 3);
      cv::putText(info, "se_ji translation norm: " + to_string(se_ji.translation().norm()), cv::Point2f(20 + offset, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 3);

      cv::putText(info, "current frame: " + to_string(kf1->keyframe_id) + " Agent: " + to_string(kf1->AgentId_), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
      cv::putText(info, "previous frame: " + to_string(kf0->keyframe_id) + " Agent: " + to_string(kf0->AgentId_), cv::Point2f(20 + offset, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 3);
      cv::vconcat(info, outImg, outImg);
      std::string image_path;
      image_path = "/home/yurong/LoopResult/" + to_string(kf1->keyframe_id) + ".png";
      cv::imwrite(image_path, outImg);
      sensor_msgs::ImagePtr lc_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", outImg).toImageMsg();
      merge_Img_pub.publish(lc_msg);
    }
  }

  return is_lc;

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
      pubGlobalPath(kf);
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
        last_pgo_id = static_cast<int>(kf_curr->keyframe_id);
      return false;
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
      last_pgo_id = static_cast<int>(kf_curr->keyframe_id);
    return false;
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
        ROS_WARN("\033[1;32m Finish optimize. Optimize %d times \033[0m", cnt++);
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
  //  for(auto id:loop_ids){
  //    cout << id.transpose() << endl;
  //  }
  //  cout << "earliest index: " << kf_prev_idx << endl;
  //  cout << "current index: " << kf_curr_idx << endl;


  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(true);


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
    //cout << "recover: " << i << endl;
    shared_ptr<KeyFrameMerge> kf = kfs_all[i];
    pubGlobalPath(kf);

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



void Merging::pubGlobalPath(shared_ptr<KeyFrameMerge> kf)
{
  //  for(int i = 0; i<globalpaths_.size(); i++)
  //  {

  //  }
  switch(kf->AgentId_)
  {
  case(0):
  {
    globalpaths_[0]->pubPathT_w_c((kf->T_c_w_global).inverse(),kf->t);
    break;
  }
  case(1):
  {
    globalpaths_[1]->pubPathT_w_c((kf->T_c_w_global).inverse(),kf->t);
    break;
  }
  case(2):
  {
    globalpaths_[2]->pubPathT_w_c((kf->T_c_w_global).inverse(),kf->t);
    break;
  }
  }


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


