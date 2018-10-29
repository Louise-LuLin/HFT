#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "vector"
#include "math.h"
#include "string.h"
#include <string>
#include <iostream>
#include "omp.h"
#include "map"
#include "set"
#include <fstream>
#include "vector"
#include "common.hpp"
#include "algorithm"
#include "lbfgs.h"
#include "sstream"
#include <assert.h>

//#include "gzstream.h"

/// Safely open a file
FILE* fopen_(const char* p, const char* m)
{
  FILE* f = fopen(p, m);
  if (!f)
  {
    printf("Failed to open %s\n", p);
    exit(1);
  }
  return f;
}

/// Data associated with a rating
struct vote
{
  int user; // ID of the user
  int item; // ID of the item
  float value; // Rating

  int voteTime; // Unix-time of the rating
  std::vector<int> words; // IDs of the words in the review
};

typedef struct vote vote;

/// To sort words by frequency in a corpus
bool wordCountCompare(std::pair<std::string, int> p1, std::pair<std::string, int> p2)
{
  return p1.second > p2.second;
}

/// To sort votes by product ID
bool voteCompare(vote* v1, vote* v2)
{
  return v1->item > v2->item;
}

/// Sign (-1, 0, or 1)
template<typename T> int sgn(T val)
{
  return (val > T(0)) - (val < T(0));
}

class corpus
{
public:
  corpus(std::string prefix, std::string source, int max)
  {
    std::string voteFile = prefix + "/HFT/data.tsv";
    std::string CVIdxFile = prefix + "/" + source + "CVIndex.txt";

    std::map<std::string, int> uCounts;
    std::map<std::string, int> bCounts;

    std::string uName;
    std::string bName;
    float value;
    int voteTime;
    int nw;
    int nRead = 0;

    std::fstream in;
    //igzstream in;

    std::string line;
    std::string sWord;
    
    // Read the input file. 
    // The first time the file is read it is only to compute word counts, in order to select the top "maxWords" words to include in the dictionary
    in.open(voteFile.c_str());
    while (std::getline(in, line))
    {
      std::stringstream ss(line);
      ss >> uName >> bName >> value >> voteTime >> nw;
      if (value > 5 or value < 0)
      { // Ratings should be in the range [0,5]
        printf("Got bad value of %f\nOther fields were %s %s %d\n", value, uName.c_str(), bName.c_str(), voteTime);
        exit(0);
      }
      for (int w = 0; w < nw; w++)
      {
        ss >> sWord;
        if (wordCount.find(sWord) == wordCount.end())
          wordCount[sWord] = 0;
        wordCount[sWord]++;
      }

      if (uCounts.find(uName) == uCounts.end())
        uCounts[uName] = 0;
      if (bCounts.find(bName) == bCounts.end())
        bCounts[bName] = 0;
      uCounts[uName]++;
      bCounts[bName]++;

      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush(stdout);
      }

      if (max > 0 and (int) nRead >= max)
        break;
    }
    in.close();

    printf("\nnUsers = %d, nItems = %d, nRatings = %d\n", (int) uCounts.size(), (int) bCounts.size(), nRead);

    V = new std::vector<vote*>();
    vote* v = new vote();
    std::map<std::string, int> userIds;
    std::map<std::string, int> beerIds;
    std::map<std::string, int> rV;
    std::map<std::string, std::vector<std::string>> mapByUserIds;
    std::map<int, int> CVIndex;

    nUsers = 0;
    nBeers = 0;
    // Comment this block to include all users, otherwise only users/items with userMin/beerMin ratings will be considered
    //    nUsers = 1;
    //    nBeers = 1;
    //    userIds["NOT_ENOUGH_VOTES"] = 0;
    //    beerIds["NOT_ENOUGH_VOTES"] = 0;
    //    rUserIds[0] = "NOT_ENOUGH_VOTES";
    //    rBeerIds[0] = "NOT_ENOUGH_VOTES";
    //    vote* v_ = new vote();
    //    v_->user = 0;
    //    v_->item = 0;
    //    v_->value = 0;
    //    v_->voteTime = 0;
    //    V->push_back(v_);

    int userMin = 0;
    int beerMin = 0;

    int maxWords = 5000; // Dictionary size
    std::vector < std::pair<std::string, int> > whichWords;
    for (std::map<std::string, int>::iterator it = wordCount.begin(); it != wordCount.end(); it++)
      whichWords.push_back(*it);
    sort(whichWords.begin(), whichWords.end(), wordCountCompare);
    if ((int) whichWords.size() < maxWords)
      maxWords = (int) whichWords.size();
    nWords = maxWords;
    for (int w = 0; w < maxWords; w++)
    {
      wordId[whichWords[w].first] = w;
      idWord[w] = whichWords[w].first;
    }

    // Re-read the entire file, this time building structures from those words in the dictionary
    std::fstream in2;
    //igzstream in2;
    in2.open(voteFile.c_str());
    nRead = 0;
    while (std::getline(in2, line))
    {
      std::stringstream ss(line);
      ss >> uName >> bName >> value >> voteTime >> nw;

      for (int w = 0; w < nw; w++)
      {
        ss >> sWord;
        if (wordId.find(sWord) != wordId.end())
          v->words.push_back(wordId[sWord]);
      }

      int user_idx=0;
      int beer_idx=0;
      if (uCounts[uName] >= userMin)
      {
        if (userIds.find(uName) == userIds.end())
        {
          rUserIds[nUsers] = uName;
          userIds[uName] = nUsers++;
        }
        user_idx = userIds[uName];
      }
      v->user = user_idx;

      if (bCounts[bName] >= beerMin)
      {
        if (beerIds.find(bName) == beerIds.end())
        {
          rBeerIds[nBeers] = bName;
          beerIds[bName] = nBeers++;
        }
        beer_idx = beerIds[bName];
      }
      v->item = beer_idx;

      // add by Lu
      mapByUserIds[uName].push_back(bName);
      rV[std::to_string(user_idx) + "_" + std::to_string(mapByUserIds[uName].size()-1)] = V->size();

      v->value = value;
      v->voteTime = voteTime;

      V->push_back(v);
      v = new vote();

      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush( stdout);
      }

      if (max > 0 and (int) nRead >= max)
        break;
    }
    printf("\n[info]rV = %d\n", (int)rV.size());
    delete v;

    in2.close();

    //read CVIndex file
    int cur_Id = 0;
    int user_idx = 0;
    int beer_idx = 0;
    int mask = 0;
    std::string token;
    std::fstream in3;
    in3.open(CVIdxFile.c_str());
    nRead = 0;
    while (std::getline(in3, line))
    {
      std::replace( line.begin(), line.end(), ',', ' ');
      std::stringstream ss(line);
      ss >> uName >> cur_Id >> mask;
      
      if(userIds.find(uName) == userIds.end()){
        printf("[err]User %s not exist\n", uName.c_str());
        // assert(userIds.find(uName) != userIds.end());
      }
      else
        user_idx = userIds[uName];

      std::string voteKey = std::to_string(user_idx) + "_" + std::to_string(cur_Id);
      if(rV.find(voteKey) == rV.end()){
        printf("[err]User-id pair %s(%s, %d) not exist\n", voteKey.c_str(), uName.c_str(), cur_Id);
        // assert(rV.find(voteKey) != rV.end());
      }
      else
        CVIndex[rV[voteKey]] = mask;
      
      nRead++;
      if (nRead % 100000 == 0)
      {
        printf(".");
        fflush( stdout);
      }
    }
    in3.close();
    printf("nRead = %d, CVIndex size=%d, mapByUser size=%d\n", nRead, (int)CVIndex.size(), (int)mapByUserIds.size());

  }

  ~corpus()
  {
    for (std::vector<vote*>::iterator it = V->begin(); it != V->end(); it++)
      delete *it;
    delete V;
  }

  std::vector<vote*>* V;
  std::map<std::string, int> rV;

  int nUsers; // Number of users
  int nBeers; // Number of items
  int nWords; // Number of words

  std::map<std::string, int> userIds; // Maps a user's string-valued ID to an integer
  std::map<std::string, int> beerIds; // Maps an item's string-valued ID to an integer

  std::map<std::string, std::vector<std::string>> mapByUserIds; // map a user's string ID to a list of item's string ID (userID->{itemID})
  std::map<int, int> CVIndex; // map vote index to a CV index (uIdx_iIdx -> CVIndex)

  std::map<int, std::string> rUserIds; // Inverse of the above map
  std::map<int, std::string> rBeerIds;

  std::map<std::string, int> wordCount; // Frequency of each word in the corpus
  std::map<std::string, int> wordId; // Map each word to its integer ID
  std::map<int, std::string> idWord; // Inverse of the above map
};
