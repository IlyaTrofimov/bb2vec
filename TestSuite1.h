#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <thread>

#include <boost/program_options.hpp>
#include <cxxtest/TestSuite.h>

#include "acc-rec.h"

extern int g_item_count;
extern vector<double> g_scores;

class MyTestSuite1 : public CxxTest::TestSuite
{
public:

	void testReadBasket(void)
	{
		vector<Basket> baskets;
		vector<string> item_ids;
		map<string, int> item_nums;

		ReadBaskets(string("test/baskets.train.10p.txt"), &baskets, &item_ids, &item_nums, nullptr);

		TS_ASSERT_EQUALS(baskets.size(), 34753);
		TS_ASSERT_EQUALS(item_ids.size(), 6083);
		TS_ASSERT_EQUALS(item_nums.size(), 6083);
	}

	void testReadViews(void)
	{
		vector<Basket> baskets;
		vector<string> item_ids;
		map<string, int> item_nums;
		map<string, int> token_nums;
		vector<ViewObject> views;

		ReadViews("test/views.txt", &item_ids, &item_nums, &token_nums, &views);

		TS_ASSERT_EQUALS(views.size(), 945660);
		TS_ASSERT_EQUALS(item_ids.size(), 7121);
		TS_ASSERT_EQUALS(item_nums.size(), 7121);
		TS_ASSERT_EQUALS(token_nums.size(), 0);
	}

	void testReadItemFeatures(void)
	{
		vector<Basket> baskets;
		vector<string> item_ids;
		map<string, int> item_nums;

		ReadBaskets(string("test/mm_baskets.val.txt"), &baskets, &item_ids, &item_nums, nullptr);

		TS_ASSERT_EQUALS(baskets.size(), 8812);
		TS_ASSERT_EQUALS(item_ids.size(), 2900);
		TS_ASSERT_EQUALS(item_nums.size(), 2900);

		vector<vector<int> > item_features;
		map<string, int> token_nums;
		g_item_count = item_ids.size();

		ReadItemFeatures("test/items_features.txt", item_nums, &token_nums, &item_features);

		TS_ASSERT_EQUALS(token_nums.size(), 2175);
		TS_ASSERT_EQUALS(item_features.size(), 2900);
	}

	void testRun(void)
	{
		g_scores.clear();
		char *argv[] = {"PATH", "-b", "test/baskets.train.10p.txt", "--val", "test/baskets.val.txt", "-t", "8", "--iter", "5"};
		run_main(sizeof(argv) / sizeof(argv[0]), argv);

		TS_ASSERT_DELTA(g_scores[0], 0.283173, 5e-3);
		TS_ASSERT_DELTA(g_scores[1], 0.332549, 5e-3);
		TS_ASSERT_DELTA(g_scores[2], 0.349154, 5e-3);
		TS_ASSERT_DELTA(g_scores[3], 0.356232, 5e-3);
		TS_ASSERT_DELTA(g_scores[4], 0.359371, 5e-3);
	}

	void testRunRank(void)
	{
		g_scores.clear();
		char *argv[] = {"PATH", "-b", "test/baskets.train.10p.txt", "--val", "test/baskets.val.txt", "-t", "8", "--iter", "5", "--rank"};
		run_main(sizeof(argv) / sizeof(argv[0]), argv);

		TS_ASSERT_DELTA(g_scores[0], 0.3497, 5e-3);
		TS_ASSERT_DELTA(g_scores[1], 0.3640, 5e-3);
		TS_ASSERT_DELTA(g_scores[2], 0.3708, 5e-3);
		TS_ASSERT_DELTA(g_scores[3], 0.3731, 5e-3);
		TS_ASSERT_DELTA(g_scores[4], 0.3758, 5e-3);
	}
};
