/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */
//SYSTEM HEAD FILE
#include <dirent.h>
#include <stdio.h>
#include <string>
#include <fstream>

#include <iostream>
#include <vector>

// DBoW3
#include "../3rdPartLib/DBow3/src/DBoW3.h"
#include "../3rdPartLib/DBow3/src/DescManip.h"
#include "../3rdPartLib/DBow3/src/Vocabulary.h"
#include "../3rdPartLib/DBow3/src/BowVector.h"
#include "../3rdPartLib/DBow3/src/ScoringObject.h"
#include "../3rdPartLib/DBow3/src/Database.h"
#include "../3rdPartLib/DLib/DVision/DVision.h"
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif


using namespace DBoW3;
using namespace std;
using namespace DVision;

//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}


vector<string> readImagePaths(int argc,char **argv,int start){
  vector<string> paths;
  for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
  return paths;
}


vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") throw (std::exception){
  //select detector
  cv::Ptr<cv::Feature2D> fdetector;
  if (descriptor=="orb")        fdetector=cv::ORB::create();
  else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
  else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
  else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

  else throw std::runtime_error("Invalid descriptor");
  assert(!descriptor.empty());
  vector<cv::Mat>    features;


  cout << "Extracting   features..." << endl;
  for(size_t i = 0; i < path_to_images.size(); ++i)
  {
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cout<<"reading image: "<<path_to_images[i]<<endl;
    cv::Mat image = cv::imread(path_to_images[i], 0);
    if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
    cout<<"extracting features"<<endl;
    fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    features.push_back(descriptors);
    cout<<"done detecting features"<<endl;
  }
  return features;
}

// ----------------------------------------------------------------------------

void NewVocCreation(const vector<cv::Mat> &features)
{
  // branching factor and depth levels
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  DBoW3::Vocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
       << voc << endl << endl;

  // lets do something with this vocabulary
//  cout << "Matching images against themselves (0 low, 1 high): " << endl;
//  BowVector v1, v2;
//  for(size_t i = 0; i < features.size(); i++)
//  {
//    voc.transform(features[i], v1);
//    for(size_t j = 0; j < features.size(); j++)
//    {
//      voc.transform(features[j], v2);

//      double score = voc.score(v1, v2);
//      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
//    }
//  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("new_voc.dbow3");
  cout << "Done" << endl;
}

void AddfeaturesExistingVoc(DBoW3::Vocabulary &old_voc, const vector<cv::Mat> &features)
{
  cout << "old voc has " << old_voc.size() << " words" << endl;
  Database db(old_voc, true, 0);
  cout << "old Database information: " << endl << db << endl;
  for(size_t i = 0; i < features.size(); i++)
    db.add(features[i]);
  db.save("merge_voc.dbow3");
//  const Vocabulary * new_voc = db.getVocabulary();
//  cout << "new voc has " << new_voc->size() << " words" << endl;
//  cout << "new Database information: " << endl << db << endl;
}

////// ----------------------------------------------------------------------------

void testDatabase(const  vector<cv::Mat > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  Vocabulary voc("small_voc.yml.gz");

  Database db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(size_t i = 0; i < features.size(); i++)
    db.add(features[i]);

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(size_t i = 0; i < features.size(); i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;

  // once saved, we can load it again
  cout << "Retrieving database once again..." << endl;
  Database db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}


// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{
   size_t n_bits = 8;
   boost::dynamic_bitset<> B1(n_bits);
   boost::dynamic_bitset<> B2(n_bits);
   boost::dynamic_bitset<> temp(4, 3);

   cout << "Binary representation of 8: "
           << B1 << endl;
      cout << "Binary representation of 7: "
           << B2 << endl
           << endl;


      unsigned long long int tmp_int = 14;



      cout << B1 << endl;

      for(int i = 0; i < 4 ; i++, tmp_int >>= 1)
      {

          cout << "i: " << i << " tmp_int: " << tmp_int << endl;
          B1[i] = (tmp_int & 1);


      }
      cout << "final tmp: " << tmp_int << endl;
      cout << B1 << endl;


//  string descriptor= "orb";
//  string folder = "/home/yurong/Training/";

//  int len;
//  int img_cnt_total = 0;
//  struct dirent *pDirent;
//  DIR *pDir;
//  pDir = opendir(folder.c_str());
//  if (pDir != NULL) {
//      while ((pDirent = readdir(pDir)) != NULL) {
//          len = strlen (pDirent->d_name);
//          if (len >= 4) {
//              if (strcmp (".png", &(pDirent->d_name[len - 4])) == 0) {
//                  img_cnt_total++;
//                  //printf ("%s\n", pDirent->d_name);
//              }
//          }
//      }
//      closedir (pDir);
//  }
//  cout << "contain " << img_cnt_total << " images" << endl;
//  int img_cnt = 0;
//  vector<string> paths;
//  while(img_cnt < img_cnt_total)
//  {
//    string img_path = folder+to_string(img_cnt)+".png";
//    paths.push_back(img_path);
//    img_cnt ++;

//  }
//  vector< cv::Mat> features= loadFeatures(paths, descriptor);
//  //NewVocCreation(features);



//  Vocabulary old_voc("/home/yurong/new_ws/src/CO-VISLAM/voc/merge_voc.dbow3");
//  cout << "Vocabulary information: " << endl
//       << old_voc << endl << endl;
  //AddfeaturesExistingVoc(old_voc, features);

  //  testDatabase(features);


  return 0;
}

