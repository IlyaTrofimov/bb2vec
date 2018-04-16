#include <cstring>
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
#include <cassert>
#include <climits>

#include <boost/program_options.hpp>

#include "acc-rec.h"
#include "global.h"

using std::cout;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::string;
using std::map;
using std::set;
using std::pair;
using std::make_pair;
using std::endl;
using std::thread;
using std::to_string;
using std::max;

namespace po = boost::program_options;

/********************************************************************************************************
*
* Simple function for string splitting
*
********************************************************************************************************/
vector<string> Split(const string& line, char separator)
{
	vector<string> parts;
	int prev_position = 0;

	for (int i = 0; i <= (int)line.length(); ++i) {
		if (i == (int)line.length() || line.at(i) == separator) {
			parts.push_back(line.substr(prev_position, i - prev_position));
			prev_position = i + 1;
		}
	}

	return parts;
}

/********************************************************************************************************
*
* Force vector to unit L2-norm.
*
********************************************************************************************************/
void norm(double **v, int item) {

	double n = 0.0;

	for (int i = 0; i < g_dim; ++i)
		n += SQUARE(v[item][i]);

	n = sqrt(n);

	for (int i = 0; i < g_dim; ++i)
		v[item][i] /= n; 
}

/****************************************************************
*
* Simple function for string splitting.
*
****************************************************************/
void ReadUnaryCount(const string& fileName, const map<string, int>& item_nums, vector<double> *item_weights)
{
	ifstream ifs(fileName.c_str());

	cout << "STREAM1" << endl;

	while(ifs.good()) { 

		string s;
		std::getline(ifs, s);

		if(s.length() == 0)
			continue;

		vector<string> parts = Split(s, '\t');

		cout << parts[0] << endl;
			
		if (item_nums.find(parts[0]) == item_nums.end())
			continue;

		int item_num = item_nums.at(parts[0]);
		int count = std::stod(parts[1]);

		item_weights->at(item_num) = count;

		cout << item_num << " " << count << endl;
	}
	ifs.close();
}

/****************************************************************
*
* Enumerate items features from a string.
*
****************************************************************/
void EnumFeatures(const vector<string>& tokens, map<string, int> *token_nums, vector<int>* features)
{
	for (auto token : tokens) {
		if (token.size()) {
			int token_idx = 0;			

			if (token_nums->find(token) == token_nums->end()) {
				token_idx = token_nums->size();
				token_nums->insert(make_pair(token, token_idx));	
				token_ids.push_back(token);
			}
			else {
				token_idx = token_nums->find(token)->second;
			}

			features->push_back(token_idx);
		}
	}
}

/****************************************************************
*
* Reads features of items from file.
*
****************************************************************/
void ReadItemFeatures(const string& fileName, const map<string, int>& item_nums, map<string, int>* token_nums, vector<vector<int> >* item_features)
{
	ifstream ifs(fileName.c_str());	
	item_features->resize(g_item_count);

	while(ifs.good()) { 

		string s;
		std::getline(ifs, s);

		if(s.length() == 0)
			continue;

		vector<string> parts = Split(s, '\t');
			
		if (item_nums.find(parts[0]) == item_nums.end())  // ITEM NOT FOUND
			continue;

		int item_num = item_nums.at(parts[0]);
		item_features->at(item_num) = vector<int>();

		vector<string> tokens = Split(parts[1], ',');
		EnumFeatures(tokens, token_nums, &(item_features->at(item_num)));
	}
	ifs.close();
}

/****************************************************************
*
* Remove bias from views PMI.
*
****************************************************************/
void RemoveBias(vector<ViewObject> *views)
{
	double sum = 0.0;

	for (auto view : *views)
		sum += view.PMI;

	double avg = sum / views->size();

	for (auto& view : *views)
		view.PMI -= avg;
}

/****************************************************************
*
* Remove bias from file.
*
****************************************************************/
void ReadViews(const string& fileName, vector<string> *item_ids, map<string, int> *item_nums, map<string, int> *token_nums, vector<ViewObject> *views)
{
	ifstream ifs(fileName.c_str());

	while(ifs.good()) { 

		string s, item_id;
		std::getline(ifs, s);

		if(s.length() == 0)
			continue;

		vector<string> parts = Split(s, '\t');
		ViewObject view;
			
		if (not g_views_features) {
			item_id = parts[0];

			if (item_nums->find(item_id) == item_nums->end()) {
				int idx = item_nums->size();
				item_nums->insert(make_pair(item_id, idx));
			}

			item_id = parts[1];

			if (item_nums->find(item_id) == item_nums->end()) {
				int idx = item_nums->size();
				item_nums->insert(make_pair(item_id, idx));
			}

			view.Item1 = item_nums->at(parts[0]);
			view.Item2 = item_nums->at(parts[1]);
		}
		else {
			vector<string> tokens = Split(parts[0], ',');
			EnumFeatures(tokens, token_nums, &view.features1);

			tokens = Split(parts[1], ',');
			EnumFeatures(tokens, token_nums, &view.features2);
		}
		
		view.PMI = std::stod(parts[2]);

		views->push_back(view);
	}
	ifs.close();

	item_ids->resize(item_nums->size());

	for (auto item_iter : *item_nums)
		item_ids->at(item_iter.second) = item_iter.first;
}

/****************************************************************
*
* Read baskets from file
*
****************************************************************/
void ReadBaskets(const string& fileName, vector<Basket> *baskets, vector<string> *item_ids, map<string, int> *item_nums, set<int> *new_items)
{
	map<string, vector<int> > baskets_map;
	ifstream ifs(fileName.c_str());

	while(ifs.good()) { 

		string s;
		std::getline(ifs, s);

		if(s.length() == 0)
			continue;

		vector<string> parts = Split(s, '\t');
		string sid = parts[0];
		string item_id = parts[1];

		if (item_nums->find(item_id) == item_nums->end()) {
			int idx = item_nums->size();
			item_nums->insert(make_pair(item_id, idx));
			if (new_items)
				 new_items->insert(idx);
		}
 
		if (baskets_map.find(sid) == baskets_map.end()) {
			baskets_map[sid] = vector<int>();
		}

		int item_num = item_nums->at(item_id);
		baskets_map[sid].push_back(item_num);
	}
	ifs.close();

	item_ids->resize(item_nums->size());

	for (auto item_iter : *item_nums)
		item_ids->at(item_iter.second) = item_iter.first;

	for (auto basket : baskets_map) {
		baskets->push_back(Basket());

		for (auto item_num : basket.second) {
			baskets->back().push_back(item_num);
			//items_with_counts.push_back(item_num);
		}
	}
}

/****************************************************************
*
* Safe function for sigmoid calculation.
*
****************************************************************/
double sigmoid(double x) 
{
	if (x > -15.0) {
		return 1.0 / (1.0 + exp(-x));
	}
	else {
		return 0.0;
	}
}

/****************************************************************
*
* Calculates inner product of 2 vectors.
*
****************************************************************/
double inner_prod(double *a, double *b, int n)
{
	double prod = 0.0;
	
	for (int i = 0; i < n; ++i) {
		prod += a[i] * b[i];
	}

	return prod * SQUARE(g_c);
}

/****************************************************************
*
* Samples random negative items.
*
****************************************************************/
vector<int> GetNegativeItems(int count) {

	// FIXME! maybe not safe in multithreading
	
	vector<int> neg_items;

	for (int i = 0; i < count; ++i) {
		if (g_negative_sampling == 1) {
			int idx = std::rand() % g_items_with_counts.size();
			neg_items.push_back(g_items_with_counts[idx]);
		}
		else if (g_negative_sampling == 0) {
			//int item_num = std::rand() % g_item_count;
			//neg_items.push_back(item_num);
			int idx = std::rand() % g_train_items.size();  // sampling of negative items from train only!
			neg_items.push_back(g_train_items[idx]);
		}
	}

	return neg_items;
}

/****************************************************************
*
* Initializes matrix of (dim1 * dim2) sizes.
*
****************************************************************/
double** init_matrix(int dim1, int dim2) 
{
	double **v = (double**)calloc(dim1, sizeof(double*));

	for (int i = 0; i < dim1; ++i)
		v[i] = (double*)calloc(dim2, sizeof(double));

	return v;
}

/****************************************************************
*
* Releases memory allocated for matrix.
*
****************************************************************/
void free_matrix(double **v, int dim1) 
{
	for (int i = 0; i < dim1; ++i)
		free(v[i]);

	free(v);
}

/****************************************************************
*
* Calculates negative log-likelihood (NLL) for one object.
*
****************************************************************/
double CalcNLLObject(int item_k, int item_m, const vector<int>& neg_items, double **v_in, double **v_out)
{
	/***************/
	/* Update v_in */
	/***************/
	double nll = 0.0;

	// positive	
	double p1 = sigmoid(-inner_prod(v_out[item_m], v_in[item_k], g_dim));
	p1 = LIMIT(p1);
	nll += -log(1 - p1);

	// negative
	for (int j = 0; j < (int)neg_items.size(); ++j) {
		int item_r = neg_items[j];
		double p2 = sigmoid(inner_prod(v_out[item_r], v_in[item_k], g_dim));
		p2 = LIMIT(p2);
		nll += -log(1 - p2);
	}

	return nll;
}

/****************************************************************
*
* Calculates part of NLL objective, required for multithreading.
*
****************************************************************/
void CalcNLLPart(const vector<pair<int, int> >& val_pairs, int from_idx, int to_idx, const vector<vector<int> >& neg_items, double **v_in, double **v_out, double *nll)
{
	*nll = 0.0;

	for (int i = from_idx; i < to_idx; ++i) {
		*nll += CalcNLLObject(val_pairs[i].first, val_pairs[i].second, neg_items[i], v_in, v_out);
	}
}

/****************************************************************
*
* Calculates NLL objective.
*
****************************************************************/
double CalcNLL(const vector<pair<int, int> >& val_pairs, const vector<vector<int> >& neg_items, double **v_in, double **v_out)
{
	vector<thread> threads;
	double nll = 0.0;
	double nlls[g_threads];

	for (int i = 0; i < g_threads; ++i) {
		int from_idx = (val_pairs.size() * i)/ g_threads;
		int to_idx =  (val_pairs.size() * (i + 1)) / g_threads;
		nlls[i] = 0;

		threads.push_back(thread(CalcNLLPart, val_pairs, from_idx, to_idx, neg_items, v_in, v_out, &nlls[i]));
	}

	for (int i = 0; i < g_threads; ++i) {
		threads[i].join();
		nll += nlls[i];
	}
	
	return nll / val_pairs.size();
}

/****************************************************************
*
* Calculates inner product of 2 items vectors, including features of items.
*
****************************************************************/
inline double calc_inner_prod(int item_m, int item_k, double **v_out, double **v_in)
{
	double prod = inner_prod(v_out[item_m], v_in[item_k], g_dim);

	if (g_use_features) {
		for (auto t : item_features[item_m])
			prod += inner_prod(w_out[t], v_in[item_k], g_dim);

		for (auto t : item_features[item_k])
			prod += inner_prod(v_out[item_m], w_in[t], g_dim);

		for (auto tm : item_features[item_m])
			for (auto tk : item_features[item_k])
				prod += inner_prod(w_out[tm], w_in[tk], g_dim);
	}

	return prod;
}

/**************************************************************************
*
* Calculates part of all inner product pairs, required for multithreading.
*
***************************************************************************/
void CalcInnerProdsPart(double **v_in, double **v_out, int from_idx, int to_idx, double **inner_prods)
{
	for (int i = from_idx; i < to_idx; ++i) {
		for (int j = 0; j < (int)g_item_count; ++j) {

			inner_prods[i][j] = calc_inner_prod(j, i, v_out, v_in);
		}
	}
}

/**************************************************************************
*
* Calculates all pair-wide inner products.
*
***************************************************************************/
void CalcInnerProds(double **v_in, double **v_out, double **inner_prods)
{
	vector<thread> threads;
	int recall_threads = 8; //g_threads

	for (int i = 0; i < recall_threads; ++i) {
		int from_idx = (g_item_count * i) / recall_threads;
		int to_idx =  (g_item_count * (i + 1)) / recall_threads;

		threads.push_back(thread(CalcInnerProdsPart, v_in, v_out, from_idx, to_idx, inner_prods));
	}

	for (int i = 0; i < recall_threads; ++i)
		threads[i].join();
}

/**************************************************************************
*
* Calculates recall@length at validation set.
*
***************************************************************************/
void CalcRecall(const vector<pair<int, int> >& val_pairs, int length, double **v_in, double **v_out, double *score, double *wscore)
{
	*score = 0;
	*wscore = 0.0;

	if (val_pairs.size() == 0)
		return;

	double **inner_prods = init_matrix(g_item_count, g_item_count);
	CalcInnerProds(v_in, v_out, inner_prods);	

	double max_score = 0.0, max_wscore = 0.0;
	map<int, map<int, int> > pred_cache;

	for (auto r : val_pairs) {

		int query_item = r.first;
		
		if (pred_cache.find(query_item) == pred_cache.end()) {

			vector<pair<int, double> > candidates;

			for (int item = 0; item < g_item_count; ++item) {
				if (item != query_item)				
					if (!g_new_items.count(item) || g_recommend_new)
						candidates.push_back(make_pair(item, inner_prods[query_item][item]));
			}

			sort(candidates.begin(), candidates.end(), [](pair<int, double> x, pair<int, double> y) { return x.second > y.second; });

			pred_cache[query_item] = map<int, int>();

			for (int i = 0; i < length; ++i) {
				pred_cache[query_item].insert(make_pair(candidates[i].first, i));
			}
		}

		if (pred_cache[r.first].count(r.second)) {
			*score += 1.0 / g_item_weights[r.second];

			int pos = pred_cache[r.first][r.second];
			*wscore += 1.0 / log(2 + pos) / g_item_weights[r.second];
		}

		max_score += 1.0 / g_item_weights[r.second];
		max_wscore += 1.0 / log(2) / g_item_weights[r.second];
	}

	free_matrix(inner_prods, g_item_count);
		
	*score /= max_score;
	*wscore /= max_wscore;
}

/**************************************************************************
*
* Calculates part of predictions, required for multi-threading.
*
***************************************************************************/
void GetPredictionsPart(double **v_in, double **v_out, int from_idx, int to_idx, cache_type *res)
{
	for (int query_item = from_idx; query_item < to_idx; ++query_item) {
		vector<pair<int, double> > candidates;

		for (int item = 0; item < g_item_count; ++item) {
			if (item != query_item) {				
				double inner_prod = calc_inner_prod(item, query_item, v_out, v_in);
				candidates.push_back(make_pair(item, inner_prod));
			}
		}

		sort(candidates.begin(), candidates.end(), [](pair<int, double> x, pair<int, double> y) { return x.second > y.second; });
		res->insert(make_pair(query_item, vector<pair<int, double> >(candidates.begin(), candidates.begin() + 1000)));
	}
}

/**************************************************************************
*
* Calculates predictions for all items.
*
***************************************************************************/
void GetPredictions(const string filename, vector<string>& item_ids, double **v_in, double **v_out)
{
	cout << "calculating predictions..." << endl;

	int threads_cnt = 8; //g_threads
	vector<thread> threads;

	vector<cache_type> caches(threads_cnt);

	for (int i = 0; i < threads_cnt; ++i) {
		int from_idx = (g_item_count * i) / threads_cnt;
		int to_idx =  (g_item_count * (i + 1)) / threads_cnt;

		threads.push_back(thread(GetPredictionsPart, v_in, v_out, from_idx, to_idx, &caches[i]));
	}

	for (int i = 0; i < threads_cnt; ++i)
		threads[i].join();

	ofstream ofs(filename.c_str());
	
	for (int k = 0; k < threads_cnt; ++k) {
		int from_idx = (g_item_count * k) / threads_cnt;
		int to_idx =  (g_item_count * (k + 1)) / threads_cnt;

		for (int i = from_idx; i < to_idx; ++i) {
			ofs << item_ids[i] << "\t";
	
			for (int j = 0; j < (int)caches[k][i].size(); ++j) {
				if (j != 0) ofs << ",";
				ofs << item_ids[caches[k][i][j].first];
			}

			ofs << endl; 
		}
	}

	ofs.close();
}

/**************************************************************************
*
* Update component of a vector via AdaGrad rule.
*
***************************************************************************/
void inline update_adagrad(double grad, int item, int k, double **v, double **grad_sq_sum)
{
	grad *= g_c;

	grad_sq_sum[item][k] += SQUARE(grad);
	v[item][k] -= (g_eta / g_c) * grad / sqrt(grad_sq_sum[item][k] + 1e-12);
}

/**************************************************************************
*
* Update embeddings from views objective.
*
***************************************************************************/
void UpdateViews(const ViewObject& view, int no_update_idx1, int no_update_idx2,
		double **v1, double **v2, double **sq_sum1, double **sq_sum2,
		double **w1, double **w2, double **w_sq_sum1, double **w_sq_sum2,
		double *mse)
{
	int item_k = 0, item_m = 0;
	double prod = 0.0;
	double emb1[g_dim], emb2[g_dim];

	double v_item_m_copy[g_dim];

	if (not g_views_features) {
		item_k = view.Item1;
		item_m = view.Item2;

		prod = inner_prod(v2[item_k], v1[item_m], g_dim);
		memcpy(v_item_m_copy, v1[item_m], g_dim * sizeof(double));
	}
	else {
		memset(emb1, 0, g_dim * sizeof(double));
		memset(emb2, 0, g_dim * sizeof(double));

		for (auto t : view.features1)
			for (int k = 0; k < g_dim; ++k) 
				emb1[k] += w1[t][k];

		for (auto t : view.features2)
			for (int k = 0; k < g_dim; ++k) 
				emb2[k] += w2[t][k];

		for (int k = 0; k < g_dim; ++k)
			prod += emb1[k] * emb2[k];
	}

	*mse = SQUARE(view.PMI - prod);


	/* 
	*
	*
	*/

	double grad = 0;

	for (int k = 0; k < g_dim; ++k) {		

		if (not g_views_features) {

			if (k != no_update_idx1) {
				grad = g_mult_mf * (prod - view.PMI) * v2[item_k][k] + g_lambda_2 * v1[item_m][k];
				update_adagrad(grad, item_m, k, v1, sq_sum1);
			}

			if (k != no_update_idx2) {
				grad = g_mult_mf * (prod - view.PMI) * v_item_m_copy[k] + g_lambda_2 * v2[item_m][k];
				update_adagrad(grad, item_k, k, v2, sq_sum2);
			}
		}
		else {
			for (auto t : view.features1) {
				grad = g_mult_mf * (prod - view.PMI) * emb2[k] + g_lambda_2 * w1[t][k];
				update_adagrad(grad, t, k, w1, w_sq_sum1);
			}

			for (auto t : view.features2) {
				grad = g_mult_mf * (prod - view.PMI) * emb1[k] + g_lambda_2 * w2[t][k];
				update_adagrad(grad, t, k, w2, w_sq_sum2);
			}
		}
	}

	if (g_norm_v_out) {
		norm(v1, item_m);
		norm(v2, item_k);
	}
}

/**************************************************************************
*
* Update embeddings for regularization part.
*
***************************************************************************/
void UpdateReg(double **v, double **grad_sq_sum, int item, vector<int>* v_ts)
{
	int age = g_time - v_ts->at(item);

	if (age > 1) {
		for (int k = 0; k < g_dim; ++k) {
			double grad = g_lambda_2 * v[item][k];
			double a;

			if (grad_sq_sum[item][k])
				a = g_eta * g_lambda_2 / sqrt(grad_sq_sum[item][k]);
			else
				a = g_eta * g_lambda_2;
	
			double factor = pow(1 - a, age - 1);

			v[item][k] *= factor;
			grad_sq_sum[item][k] += age * SQUARE(grad);
		}
	}

	v_ts->at(item) = g_time;
}

/**************************************************************************
*
* Update for multi-modal embeddings, ranking objective.
*
***************************************************************************/
void UpdateRank(int item_k, int item_m, const vector<Basket>& baskets, double **v_in, double **v_out, double **grad_in_sq_sum, double **grad_out_sq_sum, double *nll)
{
	/************************/
	/* Calc update for v_in */
	/************************/
	*nll = 0.0;
	vector<double> v_in_grad(g_dim, 0);
	map<int, double> inner_prods_cache;

	double prod_m = calc_inner_prod(item_m, item_k, v_out, v_in);
	inner_prods_cache[item_m] = prod_m; 

	vector<int> neg_items = GetNegativeItems(g_neg_count);

	if (g_lambda_2) {
		UpdateReg(v_in, grad_in_sq_sum, item_k, &v_in_ts);
		UpdateReg(v_out, grad_out_sq_sum, item_m, &v_out_ts);
		
		for (auto item_r : neg_items)
			UpdateReg(v_out, grad_out_sq_sum, item_r, &v_out_ts);
	}

	for (auto item_r : neg_items) {
		double prod_r = calc_inner_prod(item_r, item_k, v_out, v_in);
		inner_prods_cache[item_r] = prod_r;

		double p = sigmoid(prod_r - prod_m);
		p = LIMIT(p);
		*nll += -log(1 - p);
	
		for (int k = 0; k < g_dim; ++k)
			v_in_grad[k] += p * (v_out[item_r][k] - v_out[item_m][k]);  
	
		if (g_use_features) {	
			for (auto t : item_features[item_r])
				for (int k = 0; k < g_dim; ++k)
					v_in_grad[k] += p * w_out[t][k];  

			for (auto t : item_features[item_m])
				for (int k = 0; k < g_dim; ++k)
					v_in_grad[k] += -p * w_out[t][k];  
		}
	}

	/****************/
	/* Update v_in  */
	/****************/
	double v_in_copy[g_dim];
	memcpy(v_in_copy, v_in[item_k], g_dim * sizeof(double));

	if (g_use_features) {
		for (auto t : item_features[item_k])
			for (int k = 0; k < g_dim; ++k)
				v_in_copy[k] += w_in[t][k];  
	}

	for (int k = 0; k < g_dim; ++k) {	
		if (k == V_IN_CONST_IDX) 
			continue;

		double grad = g_mult_st * v_in_grad[k] + g_lambda_2 * v_in[item_k][k];
		update_adagrad(grad, item_k, k, v_in, grad_in_sq_sum);
	}

	// Update features
	if (g_use_features) {
		for (auto t : item_features[item_k]) {
			for (int k = 0; k < g_dim; ++k) {	
				double grad = g_mult_st * v_in_grad[k] + g_lambda_2 * w_in[t][k];
				update_adagrad(grad, t, k, w_in, grad_w_in_sq_sum);
			}
		}
	}

	/*****************/
	/* Update v_out  */
	/*****************/
	for (auto item_r : neg_items) {
		double p = sigmoid(inner_prods_cache[item_r] - inner_prods_cache[item_m]);

		for (int k = 0; k < g_dim; ++k) {
			if (k == V_OUT_CONST_IDX) 
				continue;

			// update v_out for item_m
			double grad = -g_mult_st * p * v_in_copy[k] + g_lambda_2 * v_out[item_m][k];
			update_adagrad(grad, item_m, k, v_out, grad_out_sq_sum);

			if (g_use_features) {
				for (auto t : item_features[item_m])
					update_adagrad(grad, t, k, w_out, grad_w_out_sq_sum);
			}

			// update v_out for negative samples
			grad = g_mult_st * p * v_in_copy[k] + g_lambda_2 * v_out[item_r][k];
			update_adagrad(grad, item_r, k, v_out, grad_out_sq_sum);

			if (g_use_features)  {
				for (auto t : item_features[item_r])
					update_adagrad(grad, t, k, w_out, grad_w_out_sq_sum);
			}
		}

		// FIXME !!!
		//if (g_norm_v_out) {
			norm(v_out, item_r);
			norm(v_out, item_m);
		//}
	}
}

/**************************************************************************
*
* Update for multi-modal embeddings,  OBSOLETE
*
***************************************************************************/
void UpdateClass(int item_k, int item_m, const vector<Basket>& baskets, double **v_in, double **v_out, double **grad_in_sq_sum, double **grad_out_sq_sum, double *nll)
{
	/***************/
	/* Update v_in */
	/***************/
	*nll = 0.0;
	vector<double> v_in_grad(g_dim, 0);
	map<int, double> inner_prods_cache;

	// positive examples
	double prod = inner_prod(v_out[item_m], v_in[item_k], g_dim);
	inner_prods_cache[item_m] = prod;

	double p1 = sigmoid(-prod);
	p1 = LIMIT(p1);
	*nll += -log(1 - p1);

	for (int k = 0; k < g_dim; ++k) {	
		v_in_grad[k] = -p1 * v_out[item_m][k];  
	}

	// negative examples
	vector<int> neg_items = GetNegativeItems(g_neg_count);

	if (g_lambda_2) {
		UpdateReg(v_in, grad_in_sq_sum, item_k, &v_in_ts);
		UpdateReg(v_out, grad_out_sq_sum, item_m, &v_out_ts);
		
		for (auto item_r : neg_items)
			UpdateReg(v_out, grad_out_sq_sum, item_r, &v_out_ts);
	}

	for (int j = 0; j < (int)neg_items.size(); ++j) {
		int item_r = neg_items[j];

		prod = inner_prod(v_out[item_r], v_in[item_k], g_dim);
		inner_prods_cache[item_r] = prod;

		double p2 = sigmoid(prod);
		p2 = LIMIT(p2);
		*nll += -log(1 - p2);
	
		for (int k = 0; k < g_dim; ++k) {	
			v_in_grad[k] += g_dw * p2 * v_out[item_r][k];  

		}
	}

	/****************/
	/* Update v_out */
	/****************/
	map<int, int> pos_updates;
	map<int, int> neg_updates;
	map<int, int> all_updates;
	
	// positive examples
	INCREMENT(pos_updates, item_m);
	all_updates[item_m] = 1;

	// negative examples
	for (int j = 0; j < (int)neg_items.size(); ++j) {

		int item_r = neg_items[j];
		INCREMENT(neg_updates, item_r);
		all_updates[item_r] = 1;
	}

	// update v_in
	double v_in_copy[g_dim];
	memcpy(v_in_copy, v_in[item_k], g_dim * sizeof(double));

	for (int k = 0; k < g_dim; ++k) {	
		if (k == V_IN_CONST_IDX) 
			continue;

		double grad = v_in_grad[k] + g_lambda_2 * v_in[item_k][k];
		grad *= g_mult_st;

		//grad_in_sq_sum[item_k][k] += SQUARE(grad);
		//v_in[item_k][k] -= (g_eta / g_c) * grad / sqrt(grad_in_sq_sum[item_k][k] + 1e-12);

		update_adagrad(grad, item_k, k, v_in, grad_in_sq_sum);
	}

	// update v_out
	for (auto r : all_updates) {

		int item_p = r.first; // item to predict
		int pos_count = GET(pos_updates, item_p, 0);
		int neg_count = GET(neg_updates, item_p, 0);
		double p1 = 0.0, p2 = 0.0;

		if (pos_count) {
			p1 = sigmoid(-inner_prods_cache[item_p]);
			p1 = LIMIT(p1);
		} 

		if (neg_count) {
			p2 = sigmoid(inner_prods_cache[item_p]);
			p2 = LIMIT(p2);
		}

		for (int k = 0; k < g_dim; ++k) {

			if (k == V_OUT_CONST_IDX) 
				continue;

			double grad = (-p1 * pos_count + g_dw * p2 * neg_count) * v_in_copy[k] + g_lambda_2 * v_out[item_p][k];
			grad *= g_mult_st;
			
			//grad_out_sq_sum[item_p][k] += SQUARE(grad);
			//v_out[item_p][k] -= (g_eta / g_c) * grad / sqrt(grad_out_sq_sum[item_p][k] + 1e-12);

			update_adagrad(grad, item_p, k, v_out, grad_out_sq_sum);
		}
	}
}

/**************************************************************************
*
* Update for multi-modal embeddings,  OBSOLETE
*
***************************************************************************/
void Update(int basket_idx, const vector<Basket>& baskets, double **v_in, double **v_out, double **grad_in_sq_sum, double **grad_out_sq_sum, double *nll)
{
	/***************/
	/* Update v_in */
	/***************/
	int basket_size = (int)baskets[basket_idx].size();
	int item_idx = std::rand() % basket_size;
	int item_k = baskets[basket_idx][item_idx]; // center item

	*nll = 0.0;
	vector<double> v_in_grad(g_dim, 0);
	
	for (int i = 0; i < basket_size; ++i) {
		
		if (i != item_idx) {				

			int item_m = baskets[basket_idx][i]; // item to predict
			double p1 = sigmoid(-inner_prod(v_out[item_m], v_in[item_k], g_dim));
			p1 = LIMIT(p1);
			*nll += -log(1 - p1);

			for (int k = 0; k < g_dim; ++k) {	
				v_in_grad[k] = -p1 * v_out[item_m][k];  
				//v_in[item_k][k] -= -g_eta * p_1 * v_out[item_m][k]; 
			}
		}
	}

	vector<int> neg_items = GetNegativeItems(g_neg_count * (basket_size - 1));

	for (int j = 0; j < (int)neg_items.size(); ++j) {
		int item_r = neg_items[j];
		double p2 = sigmoid(inner_prod(v_out[item_r], v_in[item_k], g_dim));
		p2 = LIMIT(p2);
		*nll += -log(1 - p2);
	
		for (int k = 0; k < g_dim; ++k) {	
			v_in_grad[k] += p2 * v_out[item_r][k];  
			//v_in[item_k][k] -= g_eta * p_2 * v_out[item_r][k]; 
		}
	}

	for (int k = 0; k < g_dim; ++k) {	
		if (k == V_IN_CONST_IDX) 
			continue;

		//grad_in_sq_sum[item_k][k] = GAMMA * grad_in_sq_sum[item_k][k] + (1 - GAMMA) * SQUARE(v_in_grad[k]);
		grad_in_sq_sum[item_k][k] += SQUARE(v_in_grad[k]);
		v_in[item_k][k] -= g_eta * v_in_grad[k] / sqrt(grad_in_sq_sum[item_k][k] + 1e-12);

		//v_in[item_k][k] -= g_eta * v_in_grad[k];
	}

	/***************/
	/* Update v_in */
	/***************/
	map<int, int> pos_updates;
	map<int, int> neg_updates;
	map<int, int> all_updates;
	
	for (int i = 0; i < (int)baskets[basket_idx].size(); ++i) {

		int item_m = baskets[basket_idx][i]; // item to predict
		//double p_1 = sigmoid(-inner_prod(v_out[item_m], v_in[item_k], g_dim));
		//p_1 = LIMIT(p_1);

		if (i != item_idx) {

			INCREMENT(pos_updates, item_m);
			all_updates[item_m] = 1;

			//for (int k = 0; k < g_dim; ++k) {	
			//	v_out[item_m][k] -= -g_eta * p_1 * v_in[item_k][k]; 
			//}
		}
	}

	for (int j = 0; j < (int)neg_items.size(); ++j) {
		int item_r = neg_items[j];
		//double p_2 = sigmoid(inner_prod(v_out[item_r], v_in[item_k], g_dim));
		//p_2 = LIMIT(p_2);
			
		INCREMENT(neg_updates, item_r);
		all_updates[item_r] = 1;
	
		//for (int k = 0; k < g_dim; ++k) {	
		//	v_out[item_r][k] -= g_eta * p_2 * v_in[item_k][k]; 
		//}
	}

	for (auto r : all_updates) {

		int item_p = r.first; // item to predict
		int pos_count = GET(pos_updates, item_p, 0);
		int neg_count = GET(neg_updates, item_p, 0);
		double p1 = 0.0, p2 = 0.0;

		if (pos_count) {			
			p1 = sigmoid(-inner_prod(v_out[item_p], v_in[item_k], g_dim));
			p1 = LIMIT(p1);
		} 

		if (neg_count) {
			p2 = sigmoid(inner_prod(v_out[item_p], v_in[item_k], g_dim));
			p2 = LIMIT(p2);
		}

		for (int k = 0; k < g_dim; ++k) { // NB!!! start from 1	

			if (k == V_OUT_CONST_IDX) 
				continue;

			double grad = (-p1 * pos_count + p2 * neg_count) * v_in[item_k][k]; 
			
			//grad_out_sq_sum[item_p][k] = GAMMA * grad_out_sq_sum[item_p][k] + (1 - GAMMA) * SQUARE(grad);
			grad_out_sq_sum[item_p][k] += SQUARE(grad);
			v_out[item_p][k] -= g_eta * grad / sqrt(grad_out_sq_sum[item_p][k] + 1e-12) ;

			//v_out[item_p][k] -= g_eta * grad;
		}
	}
}

/**************************************************************************
*
* Generate train pairs from baskets.
*
***************************************************************************/
void GetTrainPairs(const vector<Basket>& baskets, vector<pair<int, int> > *train_pairs) 
{
	for (auto basket : baskets) {
		if (basket.size() >= 2) {
			for (auto item_num1 : basket) {
				for (auto item_num2: basket) {
					if (item_num1 != item_num2) {
						train_pairs->push_back(make_pair(item_num1, item_num2));
					}
				}
			}
		}
	}
}

/**************************************************************************
*
* Calculate improvement for quality score.
*
***************************************************************************/
double GetImprovement()
{
	if ((int)g_scores.size() < g_stop_count + 1) 
		return 1.0;

	double prev_max = *std::max_element(g_scores.begin(), g_scores.end() - g_stop_count);
	double max = *std::max_element(g_scores.begin(), g_scores.end());

	if (prev_max) 
		return max / prev_max - 1;
	else
		return 0.0;
}

/**************************************************************************
*
* Function to be run in a separate thread for updateing weights.
*
***************************************************************************/
void async_updater(int thread_num, const vector<Basket>& baskets, const vector<pair<int, int> >& train_pairs, const vector<ViewObject>& views, int samples, 
          double **v_in, double **v_out, double **grad_in_sq_sum, double **grad_out_sq_sum,
          double **v_in2, double **v_out2, double **grad_in_sq_sum2, double **grad_out_sq_sum2,	
          double *nll, int *nll_count, double *mse, int *mse_count)
{
	*nll = 0;
	*mse = 0;
	*mse_count = 0;
	*nll_count = 0;
	std::uniform_int_distribution<int> distr1(0, views.size() - 1);
	std::uniform_int_distribution<int> distr2(0, train_pairs.size() - 1);
	std::uniform_int_distribution<int> distr3(0, g_item_count - 1);

	std::bernoulli_distribution bad_coin(g_views_importance / (1 + g_views_importance));
	std::bernoulli_distribution bad_coin_ni(g_ni);

	std::bernoulli_distribution bad_coin2(g_fake_ratio);

	// multithreading ???? max samples ???
	if (g_views_sample)
		samples *= (1 + g_views_importance);

	int i = 0;
	
	while (i < samples) {

		if (g_stop_threads)
			break;

		int neg_updates = 1, views_updates = 1;

		if (g_views_sample == 0) { // no views
			neg_updates = 1;
			views_updates = 0;
		}
		else {
			if (bad_coin(g_generators[thread_num])) {
				views_updates = 1;
				neg_updates = 0;
			}
			else {
				views_updates = 0;
				neg_updates = 1;
			}
		}

		double part_nll = 0, part_mse = 0;

		for (int j = 0; j < views_updates; ++j) {
	
			int ni = bad_coin_ni(g_generators[thread_num]);

			if (ni) {
				int view_idx = distr1(g_generators[thread_num]);
				UpdateViews(views[view_idx], V_IN_CONST_IDX, -1, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, w_in, w_out, grad_w_in_sq_sum, grad_w_out_sq_sum, &part_mse);

				*mse += part_mse;
				(*mse_count)++;
			}
			else {

				int view_idx = distr1(g_generators[thread_num]);

				if (!g_simmetric_views) {
					if (g_lambda_2) {
						UpdateReg(v_in, grad_in_sq_sum, views[view_idx].Item1, &v_in_ts);
						UpdateReg(v_in2, grad_in_sq_sum2, views[view_idx].Item2, &v_in2_ts);
					}
				
					UpdateViews(views[view_idx], V_IN_CONST_IDX, -1, v_in, v_in2, grad_in_sq_sum, grad_in_sq_sum2, w_in, w_in2, grad_w_in_sq_sum, grad_w_in_sq_sum2, &part_mse);
				}
				else {
					UpdateViews(views[view_idx], V_IN_CONST_IDX, V_IN_CONST_IDX, v_in, v_in, grad_in_sq_sum, grad_in_sq_sum, w_in, w_in, grad_w_in_sq_sum, grad_w_in_sq_sum, &part_mse);
				}

				*mse += part_mse;
				(*mse_count)++;
				
				view_idx = distr1(g_generators[thread_num]);

				if (!g_simmetric_views) {
					if (g_lambda_2) {
						UpdateReg(v_out, grad_out_sq_sum, views[view_idx].Item1, &v_out_ts);
						UpdateReg(v_out2, grad_out_sq_sum2, views[view_idx].Item2, &v_out2_ts);
					}

					UpdateViews(views[view_idx], V_OUT_CONST_IDX, -1, v_out, v_out2, grad_out_sq_sum, grad_out_sq_sum2, w_out, w_out2, grad_w_out_sq_sum, grad_w_out_sq_sum2, &part_mse);
				}
				else
					UpdateViews(views[view_idx], V_OUT_CONST_IDX, V_OUT_CONST_IDX, v_out, v_out, grad_out_sq_sum, grad_out_sq_sum, w_out, w_out, grad_w_out_sq_sum, grad_w_out_sq_sum, &part_mse);

				*mse += part_mse;
				(*mse_count)++;
			}
		}

		for (int j = 0; j < neg_updates; ++j) {
			int pair_idx = distr2(g_generators[thread_num]);

			if (!g_rank) 
				UpdateClass(train_pairs[pair_idx].first, train_pairs[pair_idx].second, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);
			else 
				UpdateRank(train_pairs[pair_idx].first, train_pairs[pair_idx].second, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);

			//if (g_l2)
			//	g_c *= (1 - g_l2 / train_pairs.size());
		}


		int cnt;

		if (g_fake_ratio) {
			if (g_fake_ratio < 1) 
				cnt = (int)bad_coin2(g_generators[thread_num]);
			else
				cnt = (int)g_fake_ratio;

			for (int m = 0; m < cnt; ++m) {
				int fake_item1 = distr3(g_generators[thread_num]);
				int fake_item2 = distr3(g_generators[thread_num]);

				if (!g_rank) 
					UpdateClass(fake_item1, fake_item2, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);
				else 
					UpdateRank(fake_item1, fake_item2, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);
			}
		}

		/*if (g_l2) {
			for (int i = 0; i < g_item_count; ++i) {
				for (int k = 0; k < g_dim; ++k) {
					double grad;

					grad = g_l2 * v_in[i][k];
					update_adagrad(grad, i, k, v_in, grad_in_sq_sum);

					grad = g_l2 * v_out[i][k];
					update_adagrad(grad, i, k, v_out, grad_out_sq_sum);

					grad = g_l2 * v_in2[i][k];
					update_adagrad(grad, i, k, v_in2, grad_in_sq_sum2);

					grad = g_l2 * v_out2[i][k];
					update_adagrad(grad, i, k, v_out2, grad_out_sq_sum2);
				}
			}
		}*/

		*nll += part_nll;
		(*nll_count)++;
		g_time++;
		i++;
	}

	for (int i = 0; i < g_item_count; ++i) {
		for (int k = 0; k < g_dim; ++k) {
			v_in[i][k] *= g_c;
			v_out[i][k] *= g_c;
			v_in2[i][k] *= g_c;
			v_out2[i][k] *= g_c;
		}
	}

	g_c = 1;
}

/**************************************************************************
*
* Train multi-modal embeddings.
*
**************************************************************************/
void FitMME(const vector<Basket>& baskets, const vector<Basket>& valBaskets, const vector<ViewObject>& views, double ***p_v_in, double ***p_v_out)
{
	double **v_in = init_matrix(g_item_count, g_dim);
	double **v_out = init_matrix(g_item_count, g_dim);
	double **grad_in_sq_sum = init_matrix(g_item_count, g_dim);
	double **grad_out_sq_sum = init_matrix(g_item_count, g_dim);

	double **v_in2 = init_matrix(g_item_count, g_dim);
	double **v_out2 = init_matrix(g_item_count, g_dim);
	double **grad_in_sq_sum2 = init_matrix(g_item_count, g_dim);
	double **grad_out_sq_sum2 = init_matrix(g_item_count, g_dim);

	v_in_ts = vector<int>(g_item_count, 0);
	v_out_ts = vector<int>(g_item_count, 0);
	v_in2_ts = vector<int>(g_item_count, 0);
	v_out2_ts = vector<int>(g_item_count, 0);

	if (g_use_features || g_views_features) {
		int size = token_nums.size();

		w_in = init_matrix(size, g_dim);
		w_out = init_matrix(size, g_dim);
		grad_w_in_sq_sum = init_matrix(size, g_dim);
		grad_w_out_sq_sum = init_matrix(size, g_dim);

		w_in2 = init_matrix(size, g_dim);
		w_out2 = init_matrix(size, g_dim);
		grad_w_in_sq_sum2 = init_matrix(size, g_dim);
		grad_w_out_sq_sum2 = init_matrix(size, g_dim);
	}

	*p_v_in = v_in;
	*p_v_out = v_out;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 0.1);

	for (int i = 0; i < g_item_count; ++i) {
		for (int j = 0; j < g_dim; ++j) {
			v_in[i][j] = distribution(generator);
			v_out[i][j] = distribution(generator);
		}

		if (V_IN_CONST_IDX > 0) 
			v_in[i][V_IN_CONST_IDX] = 1;

		if (V_OUT_CONST_IDX > 0)
			v_out[i][V_OUT_CONST_IDX] = 1;
		
		//v_in2[i][1 - V_IN_CONST_IDX] = 1;
		//v_out2[i][1 - V_OUT_CONST_IDX] = 1;
	}

	if (g_use_features || g_views_features) {
		int size = token_nums.size();

		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < g_dim; ++j) {
				w_in[i][j] = distribution(generator);
				w_out[i][j] = distribution(generator);				
			}
		}
	}


/*	vector<pair<int, int> > all_items_sessions;

	for (int i = 0; i < (int)baskets.size(); ++i) {
		for (int j = 0; j < (int)baskets[i].size(); ++j) {
			all_items_sessions.push_back(pair<int, int>(i, j));	
		}
	}

	cout << all_items_sessions.size() << endl; */

	vector<pair<int, int> > train_pairs;
	GetTrainPairs(baskets, &train_pairs);

	assert(train_pairs.size() != 0);

	vector<pair<int, int> > val_pairs;
	vector<vector<int> > val_neg_items;

	for (int repl = 0; repl < 3; ++repl) {
		GetTrainPairs(valBaskets, &val_pairs);

		for (int i = 0; i < (int)val_pairs.size(); ++i) {
			val_neg_items.push_back(GetNegativeItems(g_neg_count));
		}
	}

	cout << "train pairs " << train_pairs.size() << endl;

	for (int i = 0; i < g_threads; ++i) 
		g_generators.push_back(std::default_random_engine(i));

	for (int epoch = 0; epoch < g_iterations_max; ++epoch) {
		vector<thread> threads;
		int nlls_count[g_threads], mses_count[g_threads];
		double nlls[g_threads], mses[g_threads];

		g_stop_threads = false;
	
		for (int i = 0; i < g_threads; ++i) {
			nlls[i] = 0;

			int samples = (i == 0 ? train_pairs.size() / g_threads : INT_MAX);
			samples = max(samples, 1);

			threads.push_back(thread(async_updater, i, baskets, train_pairs, views, samples,
							 v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, v_in2, v_out2, grad_in_sq_sum2, grad_out_sq_sum2,
							 &nlls[i], &nlls_count[i], &mses[i], &mses_count[i]));
		}

		double nll = 0.0, mse = 0.0;
		int nll_count = 0, mse_count = 0;

		for (int i = 0; i < g_threads; ++i) {

			threads[i].join();
			
			if (i == 0)
				g_stop_threads = true;
				
			nll += nlls[i];
			mse += mses[i];
			mse_count += mses_count[i];
			nll_count += nlls_count[i];
		}

		nll /= nll_count;
		mse = (mse_count ? mse / mse_count : 0.0);

		double test_nll = 0; //CalcNLL(val_pairs, val_neg_items, v_in, v_out);
		double recall = 0, ndcg = 0;
		
		CalcRecall(val_pairs, 20, v_in, v_out, &recall, &ndcg);

		//g_scores.push_back(1.0 / test_nll);
		g_scores.push_back(ndcg);
		double imp = GetImprovement();

		cout << "epoch " << epoch <<  " " << nll << " " << test_nll << " " << sqrt(mse) << " " << recall << " " << ndcg <<  " " << imp <<  " " << (time(NULL) - g_start_time) << endl;
		//cout << g_c << endl;
		
		if (val_pairs.size() > 0 && imp < g_termination)
			break;
	}

/*	for (int epoch = 0; epoch < g_iterations_max; ++epoch) {
		
		double test_nll = CalcNLL(val_pairs, val_neg_items, v_in, v_out);
		double recall, wrecall;
		CalcRecall(val_pairs, 20, v_in, v_out, &recall, &wrecall);

		//g_scores.push_back(recall);
		g_scores.push_back(1.0 / test_nll);
		double imp = GetImprovement();

		nll /= train_pairs.size();
		mse = (mse_count ? mse / mse_count : 0.0);

		cout << "epoch " << epoch <<  " " << nll << " " << test_nll << " " << sqrt(mse) << " " << recall << " " << wrecall <<  " " << imp << endl;

		if (imp < 1e-4)
			break;
	}
*/
/*	for (int epoch = 0; epoch < 200; ++epoch) {

		double nll = 0.0;

		for (int i = 0; i < (int)baskets.size(); ++i) {

			int basket_idx = std::rand() % baskets.size();

			if (baskets[basket_idx].size() >= 2) {
				double part_nll = 0.0;
				Update(basket_idx, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);

				nll += part_nll;
			}
		}

		cout << "epoch " << epoch <<  " " << nll << endl;
	}
*/

	free_matrix(grad_in_sq_sum, g_item_count);
	free_matrix(grad_out_sq_sum, g_item_count);
	free_matrix(v_in2, g_item_count);
	free_matrix(v_out2, g_item_count);
	free_matrix(grad_in_sq_sum2, g_item_count);
	free_matrix(grad_out_sq_sum2, g_item_count);
}

/**************************************************************************
*
* Write embeddings of items to a file.
*
**************************************************************************/
void SaveToFile(string filename, double **v, const vector<string>& item_ids) 
{
	ofstream ofs(filename.c_str());

	for (int i = 0; i < (int)item_ids.size(); ++i) {
		ofs << item_ids[i];

		for (int k = 0; k < g_dim; ++k)
			ofs << " " << v[i][k];

		ofs << endl; 
	}

	ofs.close();
}

/**************************************************************************
*
* Parse command line options.
*
**************************************************************************/
void ParseOptions(const po::variables_map& vm)
{
	g_lambda_2 = vm["lambda-2"].as<float>();
	g_l2 = vm["l2"].as<float>();
	g_iterations_max = vm["iterations"].as<int>();
	g_eta = vm["learning-rate"].as<float>();
	g_neg_count = vm["neg-sample"].as<int>();
	g_dim = vm["dim"].as<int>();
	g_views_sample = vm["views-sample"].as<int>();
	g_views_importance = vm["vi"].as<float>();
	g_termination = vm["termination"].as<float>();
	g_threads = vm["threads"].as<int>();
	g_simmetric_views = vm.count("sim");
	g_recommend_new = vm.count("new");
	g_rank = vm.count("rank");
	g_norm_v_out = vm.count("norm");
	g_stop_count = vm["stop"].as<int>();
	g_negative_sampling = vm["neg-sampling"].as<int>();
	g_mult_st = vm["mult-stochastic"].as<float>();
	g_views_features = vm.count("views-f");	
	g_dw = vm["dw"].as<float>();	
	g_fake_ratio = vm["fake"].as<float>();	
	g_naive = vm.count("naive");	
	g_ni = vm["ni"].as<float>();

	if (g_mult_st > 1) {
		g_mult_mf = 1 / g_mult_st;
		g_mult_st = 1;
	}
}

int run_main(int argc, char **argv)
{
	std::srand(11);
	g_start_time = time(NULL);

	po::options_description general_desc("General options");
	general_desc.add_options()
      	("help,h", "produce help message")
       	("baskets,b", po::value<string>(), "baskets: train set")
       	("views,v", po::value<string>()->default_value(""), "baskets: test set")
       	("val", po::value<string>()->default_value(""), "baskets validation dataset")
       	("item-count", po::value<string>(), "a file for item counts for Recall calculation")
       	("item-features", po::value<string>(), "list of item features")
		("iterations,i", po::value<int>()->default_value(500), "maximum number of iterations")
		("threads,t", po::value<int>()->default_value(1), "number of threads")
		("dim", po::value<int>()->default_value(100), "embedding dimensionality")
		("learning-rate,l", po::value<float>()->default_value(0.05), "learning rate")
		("neg-sample,n", po::value<int>()->default_value(20), "number of negative samples")
       	("views-sample", po::value<int>()->default_value(1), "number of views samples")
		("termination", po::value<float>()->default_value(1e-3), "termination threshold")
		("stop", po::value<int>()->default_value(10), "termination iter count")
		("neg-sampling", po::value<int>()->default_value(1), "sample negative items with this power")
		("mult-stochastic,m", po::value<float>()->default_value(1), "")
		("rank", "use ranking")
		("new", "include new items in rec. lists");

	po::options_description exp_desc("Experimental options");
	exp_desc.add_options()
		("lambda-2", po::value<float>()->default_value(0.0), "L2 regularization")
		("l2", po::value<float>()->default_value(0.0), "L2 regularization, v2")
		("fake", po::value<float>()->default_value(0.0), "")
		("vi", po::value<float>()->default_value(1.0), "L2 regularization")
		("neg-reg", po::value<int>()->default_value(0), "")
		("output,o", po::value<string>()->default_value(""), "output files prefix")
		("norm", "norm v_out")
		("views-f", "")
		("all-items", "")
		("dw", po::value<float>()->default_value(1.0), "")
		("sim", "use simmetric views vectors")
		("naive", "")
		("ni", po::value<float>()->default_value(0.0));

	general_desc.add(exp_desc);

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, general_desc), vm);
	po::notify(vm);  

	if (vm.count("help")) {
		cout << general_desc << endl;
		return 0;
	}

	ParseOptions(vm);

	vector<Basket> baskets;
	vector<string> item_ids;
	map<string, int> item_nums;

	const string basketsFilename = vm["baskets"].as<string>();
	ReadBaskets(basketsFilename, &baskets, &item_ids, &item_nums, nullptr);

	vector<ViewObject> views;

	if (vm["views"].as<string>().size()) 
		ReadViews(vm["views"].as<string>(), &item_ids, &item_nums, &token_nums, &views);
	else 
		g_views_sample = 0;		

	for (auto basket : baskets)
		for (auto item_num : basket)
			g_items_with_counts.push_back(item_num);

	int neg_reg = vm["neg-reg"].as<int>();

	if (!vm.count("all-items")) {
		for (int item_num = 0; item_num < (int)item_ids.size(); ++item_num)  {
			g_train_items.push_back(item_num);

			for (int i = 0; i < neg_reg; ++i) 
				g_items_with_counts.push_back(item_num);
		}
	}

	const string valBasketsFilename = vm["val"].as<string>();
	vector<Basket> valBaskets;

	if (valBasketsFilename.size()) 
		ReadBaskets(valBasketsFilename, &valBaskets, &item_ids, &item_nums, &g_new_items);

	if (vm.count("all-items")) {
		for (int item_num = 0; item_num < (int)item_ids.size(); ++item_num)  {
			g_train_items.push_back(item_num);

			for (int i = 0; i < neg_reg; ++i) 
				g_items_with_counts.push_back(item_num);
		}
	}

	g_item_count = item_ids.size();

	double norm = 0;

	for (auto r : views)
		norm += SQUARE(r.PMI);

	cout << "views PMI norm " << sqrt(norm / views.size()) << endl;
	//RemoveBias(&views);
	
	g_item_weights.resize(g_item_count);
	if (vm.count("item-count"))
		ReadUnaryCount(vm["item-count"].as<string>(), item_nums, &g_item_weights);
	else 
		std::fill(g_item_weights.begin(), g_item_weights.end(), 1);

	g_use_features = vm.count("item-features");

	if (vm.count("item-features"))
		ReadItemFeatures(vm["item-features"].as<string>(), item_nums, &token_nums, &item_features);

	cout << "train baskets " << baskets.size() << endl;
	cout << "val baskets " << valBaskets.size() << endl;
	cout << "train items " << g_train_items.size() << endl;
	cout << "items " << item_ids.size() << endl;
	cout << "new items " << g_new_items.size() << endl;
	cout << "items with counts " << g_items_with_counts.size() << endl;
	cout << "output prefix " << vm["output"].as<string>() << endl;
	cout << "item features " << item_features.size() << endl;
	cout << "token nums " << token_nums.size() << endl;

	double **v_in, **v_out;
	FitMME(baskets, valBaskets, views, &v_in, &v_out);

	if (g_use_features || g_views_features) {
		for (int i = 0; i < g_item_count; ++i) {
			for (auto t : item_features[i]) {
				for (int k = 0; k < g_dim; ++k) {
					v_in[i][k] += w_in[t][k];
					v_out[i][k] += w_out[t][k];
				}
			}
		}
	}

	string output = vm["output"].as<string>();

	SaveToFile(string("v_in") + output + string(".txt"), v_in, item_ids);
	SaveToFile(string("v_out") + output + string(".txt"), v_out, item_ids);

	if (g_use_features || g_views_features) {
		SaveToFile(string("w_in") + output + string(".txt"), w_in, token_ids);
		SaveToFile(string("w_out") + output + string(".txt"), w_out, token_ids);
	}

	GetPredictions(string("predictions") + output, item_ids, v_in, v_out);

	free_matrix(v_in, g_item_count);
	free_matrix(v_out, g_item_count);

	return 0;
}
