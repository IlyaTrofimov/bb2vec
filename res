diff --git a/acc-rec.cpp b/acc-rec.cpp
index 6d16664..8361dd2 100644
--- a/acc-rec.cpp
+++ b/acc-rec.cpp
@@ -323,7 +323,7 @@ double inner_prod(double *a, double *b, int n)
 		prod += a[i] * b[i];
 	}
 
-	return prod;
+	return prod * SQUARE(g_c);
 }
 
 /****************************************************************
@@ -526,7 +526,7 @@ void CalcRecall(const vector<pair<int, int> >& val_pairs, int length, double **v
 	double **inner_prods = init_matrix(g_item_count, g_item_count);
 	CalcInnerProds(v_in, v_out, inner_prods);	
 
-	double max_score = 0.0;
+	double max_score = 0.0, max_wscore = 0.0;
 	map<int, map<int, int> > pred_cache;
 
 	for (auto r : val_pairs) {
@@ -556,16 +556,17 @@ void CalcRecall(const vector<pair<int, int> >& val_pairs, int length, double **v
 			*score += 1.0 / g_item_weights[r.second];
 
 			int pos = pred_cache[r.first][r.second];
-			*wscore += pow(0.95, pos) / g_item_weights[r.second];
+			*wscore += 1.0 / log(2 + pos) / g_item_weights[r.second];
 		}
 
 		max_score += 1.0 / g_item_weights[r.second];
+		max_wscore += 1.0 / log(2) / g_item_weights[r.second];
 	}
 
 	free_matrix(inner_prods, g_item_count);
 		
 	*score /= max_score;
-	*wscore /= max_score;
+	*wscore /= max_wscore;
 }
 
 /**************************************************************************
@@ -595,7 +596,7 @@ void GetPredictionsPart(double **v_in, double **v_out, int from_idx, int to_idx,
 * Calculates predictions for all items.
 *
 ***************************************************************************/
-void GetPredictions(const vector<string>& item_ids, double **v_in, double **v_out)
+void GetPredictions(const string filename, vector<string>& item_ids, double **v_in, double **v_out)
 {
 	cout << "calculating predictions..." << endl;
 
@@ -614,7 +615,7 @@ void GetPredictions(const vector<string>& item_ids, double **v_in, double **v_ou
 	for (int i = 0; i < threads_cnt; ++i)
 		threads[i].join();
 
-	ofstream ofs("predictions");
+	ofstream ofs(filename.c_str());
 	
 	for (int k = 0; k < threads_cnt; ++k) {
 		int from_idx = (g_item_count * k) / threads_cnt;
@@ -642,8 +643,10 @@ void GetPredictions(const vector<string>& item_ids, double **v_in, double **v_ou
 ***************************************************************************/
 void inline update_adagrad(double grad, int item, int k, double **v, double **grad_sq_sum)
 {
+	grad *= g_c;
+
 	grad_sq_sum[item][k] += SQUARE(grad);
-	v[item][k] -= g_eta * grad / sqrt(grad_sq_sum[item][k] + 1e-12);
+	v[item][k] -= (g_eta / g_c) * grad / sqrt(grad_sq_sum[item][k] + 1e-12);
 }
 
 /**************************************************************************
@@ -738,12 +741,19 @@ void UpdateReg(double **v, double **grad_sq_sum, int item, vector<int>* v_ts)
 	int age = g_time - v_ts->at(item);
 
 	if (age > 1) {
-
 		for (int k = 0; k < g_dim; ++k) {
-			double a = g_eta * g_lambda_2 / sqrt(grad_sq_sum[item][k] + 1);
+			double grad = g_lambda_2 * v[item][k];
+			double a;
+
+			if (grad_sq_sum[item][k])
+				a = g_eta * g_lambda_2 / sqrt(grad_sq_sum[item][k]);
+			else
+				a = g_eta * g_lambda_2;
+	
 			double factor = pow(1 - a, age - 1);
 
 			v[item][k] *= factor;
+			grad_sq_sum[item][k] += age * SQUARE(grad);
 		}
 	}
 
@@ -858,8 +868,11 @@ void UpdateRank(int item_k, int item_m, const vector<Basket>& baskets, double **
 			}
 		}
 
-		norm(v_out, item_r);
-		norm(v_out, item_m);
+		// FIXME !!!
+		if (g_norm_v_out) {
+			norm(v_out, item_r);
+			norm(v_out, item_m);
+		}
 	}
 }
 
@@ -868,7 +881,7 @@ void UpdateRank(int item_k, int item_m, const vector<Basket>& baskets, double **
 * Update for multi-modal embeddings,  OBSOLETE
 *
 ***************************************************************************/
-void Update2(int item_k, int item_m, const vector<Basket>& baskets, double **v_in, double **v_out, double **grad_in_sq_sum, double **grad_out_sq_sum, double *nll)
+void UpdateClass(int item_k, int item_m, const vector<Basket>& baskets, double **v_in, double **v_out, double **grad_in_sq_sum, double **grad_out_sq_sum, double *nll)
 {
 	/***************/
 	/* Update v_in */
@@ -892,6 +905,14 @@ void Update2(int item_k, int item_m, const vector<Basket>& baskets, double **v_i
 	// negative examples
 	vector<int> neg_items = GetNegativeItems(g_neg_count);
 
+	if (g_lambda_2) {
+		UpdateReg(v_in, grad_in_sq_sum, item_k, &v_in_ts);
+		UpdateReg(v_out, grad_out_sq_sum, item_m, &v_out_ts);
+		
+		for (auto item_r : neg_items)
+			UpdateReg(v_out, grad_out_sq_sum, item_r, &v_out_ts);
+	}
+
 	for (int j = 0; j < (int)neg_items.size(); ++j) {
 		int item_r = neg_items[j];
 
@@ -903,7 +924,8 @@ void Update2(int item_k, int item_m, const vector<Basket>& baskets, double **v_i
 		*nll += -log(1 - p2);
 	
 		for (int k = 0; k < g_dim; ++k) {	
-			v_in_grad[k] += p2 * v_out[item_r][k];  
+			v_in_grad[k] += g_dw * p2 * v_out[item_r][k];  
+
 		}
 	}
 
@@ -937,8 +959,10 @@ void Update2(int item_k, int item_m, const vector<Basket>& baskets, double **v_i
 		double grad = v_in_grad[k] + g_lambda_2 * v_in[item_k][k];
 		grad *= g_mult_st;
 
-		grad_in_sq_sum[item_k][k] += SQUARE(grad);
-		v_in[item_k][k] -= g_eta * grad / sqrt(grad_in_sq_sum[item_k][k] + 1e-12);
+		//grad_in_sq_sum[item_k][k] += SQUARE(grad);
+		//v_in[item_k][k] -= (g_eta / g_c) * grad / sqrt(grad_in_sq_sum[item_k][k] + 1e-12);
+
+		update_adagrad(grad, item_k, k, v_in, grad_in_sq_sum);
 	}
 
 	// update v_out
@@ -964,11 +988,13 @@ void Update2(int item_k, int item_m, const vector<Basket>& baskets, double **v_i
 			if (k == V_OUT_CONST_IDX) 
 				continue;
 
-			double grad = (-p1 * pos_count + p2 * neg_count) * v_in_copy[k] + g_lambda_2 * v_out[item_p][k];
+			double grad = (-p1 * pos_count + g_dw * p2 * neg_count) * v_in_copy[k] + g_lambda_2 * v_out[item_p][k];
 			grad *= g_mult_st;
 			
-			grad_out_sq_sum[item_p][k] += SQUARE(grad);
-			v_out[item_p][k] -= g_eta * grad / sqrt(grad_out_sq_sum[item_p][k] + 1e-12);
+			//grad_out_sq_sum[item_p][k] += SQUARE(grad);
+			//v_out[item_p][k] -= (g_eta / g_c) * grad / sqrt(grad_out_sq_sum[item_p][k] + 1e-12);
+
+			update_adagrad(grad, item_p, k, v_out, grad_out_sq_sum);
 		}
 	}
 }
@@ -1156,15 +1182,42 @@ void async_updater(int thread_num, const vector<Basket>& baskets, const vector<p
 	*nll_count = 0;
 	std::uniform_int_distribution<int> distr1(0, views.size() - 1);
 	std::uniform_int_distribution<int> distr2(0, train_pairs.size() - 1);
+	std::uniform_int_distribution<int> distr3(0, g_item_count - 1);
+
+	std::bernoulli_distribution bad_coin(g_views_importance / (1 + g_views_importance));
+	std::bernoulli_distribution bad_coin2(g_fake_ratio);
 
-	for (int i = 0; i < samples; ++i) {
+	// multithreading ???? max samples ???
+	if (g_views_sample)
+		samples *= (1 + g_views_importance);
+
+	int i = 0;
+	
+	while (i < samples) {
 
 		if (g_stop_threads)
 			break;
 
+		int neg_updates = 1, views_updates = 1;
+
+		if (g_views_sample == 0) { // no views
+			neg_updates = 1;
+			views_updates = 0;
+		}
+		else {
+			if (bad_coin(g_generators[thread_num])) {
+				views_updates = 1;
+				neg_updates = 0;
+			}
+			else {
+				views_updates = 0;
+				neg_updates = 1;
+			}
+		}
+
 		double part_nll = 0, part_mse = 0;
 
-		for (int j = 0; j < g_views_sample; ++j) {
+		for (int j = 0; j < views_updates; ++j) {
 			int view_idx = distr1(g_generators[thread_num]);
 
 			if (!g_simmetric_views) {
@@ -1197,20 +1250,76 @@ void async_updater(int thread_num, const vector<Basket>& baskets, const vector<p
 
 			*mse += part_mse;
 			(*mse_count)++;
-			g_time++;
 		}
 
-		int pair_idx = distr2(g_generators[thread_num]);
+		for (int j = 0; j < neg_updates; ++j) {
+			int pair_idx = distr2(g_generators[thread_num]);
 
-		if (!g_rank) 
-			Update2(train_pairs[pair_idx].first, train_pairs[pair_idx].second, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);
-		else 
-			UpdateRank(train_pairs[pair_idx].first, train_pairs[pair_idx].second, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);
+			if (!g_rank) 
+				UpdateClass(train_pairs[pair_idx].first, train_pairs[pair_idx].second, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);
+			else 
+				UpdateRank(train_pairs[pair_idx].first, train_pairs[pair_idx].second, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);
+
+			//if (g_l2)
+			//	g_c *= (1 - g_l2 / train_pairs.size());
+		}
+
+
+		int cnt;
+
+		if (g_fake_ratio) {
+			if (g_fake_ratio < 1) 
+				cnt = (int)bad_coin2(g_generators[thread_num]);
+			else
+				cnt = (int)g_fake_ratio;
+
+			for (int m = 0; m < cnt; ++m) {
+				int fake_item1 = distr3(g_generators[thread_num]);
+				int fake_item2 = distr3(g_generators[thread_num]);
+
+				if (!g_rank) 
+					UpdateClass(fake_item1, fake_item2, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);
+				else 
+					UpdateRank(fake_item1, fake_item2, baskets, v_in, v_out, grad_in_sq_sum, grad_out_sq_sum, &part_nll);
+			}
+		}
+
+		/*if (g_l2) {
+			for (int i = 0; i < g_item_count; ++i) {
+				for (int k = 0; k < g_dim; ++k) {
+					double grad;
+
+					grad = g_l2 * v_in[i][k];
+					update_adagrad(grad, i, k, v_in, grad_in_sq_sum);
+
+					grad = g_l2 * v_out[i][k];
+					update_adagrad(grad, i, k, v_out, grad_out_sq_sum);
+
+					grad = g_l2 * v_in2[i][k];
+					update_adagrad(grad, i, k, v_in2, grad_in_sq_sum2);
+
+					grad = g_l2 * v_out2[i][k];
+					update_adagrad(grad, i, k, v_out2, grad_out_sq_sum2);
+				}
+			}
+		}*/
 
 		*nll += part_nll;
 		(*nll_count)++;
 		g_time++;
+		i++;
+	}
+
+	for (int i = 0; i < g_item_count; ++i) {
+		for (int k = 0; k < g_dim; ++k) {
+			v_in[i][k] *= g_c;
+			v_out[i][k] *= g_c;
+			v_in2[i][k] *= g_c;
+			v_out2[i][k] *= g_c;
+		}
 	}
+
+	g_c = 1;
 }
 
 /**************************************************************************
@@ -1352,15 +1461,16 @@ void FitMME(const vector<Basket>& baskets, const vector<Basket>& valBaskets, con
 		mse = (mse_count ? mse / mse_count : 0.0);
 
 		double test_nll = 0; //CalcNLL(val_pairs, val_neg_items, v_in, v_out);
-		double recall = 0, wrecall = 0;
+		double recall = 0, ndcg = 0;
 		
-		CalcRecall(val_pairs, 20, v_in, v_out, &recall, &wrecall);
+		CalcRecall(val_pairs, 20, v_in, v_out, &recall, &ndcg);
 
 		//g_scores.push_back(1.0 / test_nll);
-		g_scores.push_back(recall);
+		g_scores.push_back(ndcg);
 		double imp = GetImprovement();
 
-		cout << "epoch " << epoch <<  " " << nll << " " << test_nll << " " << sqrt(mse) << " " << recall << " " << wrecall <<  " " << imp <<  " " << (time(NULL) - g_start_time) << endl;
+		cout << "epoch " << epoch <<  " " << nll << " " << test_nll << " " << sqrt(mse) << " " << recall << " " << ndcg <<  " " << imp <<  " " << (time(NULL) - g_start_time) << endl;
+		//cout << g_c << endl;
 		
 		if (val_pairs.size() > 0 && imp < g_termination)
 			break;
@@ -1418,9 +1528,9 @@ void FitMME(const vector<Basket>& baskets, const vector<Basket>& valBaskets, con
 * Write embeddings of items to a file.
 *
 **************************************************************************/
-void SaveToFile(const char* filename, double **v, const vector<string>& item_ids) 
+void SaveToFile(string filename, double **v, const vector<string>& item_ids) 
 {
-	ofstream ofs(filename);
+	ofstream ofs(filename.c_str());
 
 	for (int i = 0; i < (int)item_ids.size(); ++i) {
 		ofs << item_ids[i];
@@ -1442,11 +1552,13 @@ void SaveToFile(const char* filename, double **v, const vector<string>& item_ids
 void ParseOptions(const po::variables_map& vm)
 {
 	g_lambda_2 = vm["lambda-2"].as<float>();
+	g_l2 = vm["l2"].as<float>();
 	g_iterations_max = vm["iterations"].as<int>();
 	g_eta = vm["learning-rate"].as<float>();
 	g_neg_count = vm["neg-sample"].as<int>();
 	g_dim = vm["dim"].as<int>();
 	g_views_sample = vm["views-sample"].as<int>();
+	g_views_importance = vm["vi"].as<float>();
 	g_termination = vm["termination"].as<float>();
 	g_threads = vm["threads"].as<int>();
 	g_simmetric_views = vm.count("sim");
@@ -1457,6 +1569,8 @@ void ParseOptions(const po::variables_map& vm)
 	g_negative_sampling = vm["neg-sampling"].as<int>();
 	g_mult_st = vm["mult-stochastic"].as<float>();
 	g_views_features = vm.count("views-f");	
+	g_dw = vm["dw"].as<float>();	
+	g_fake_ratio = vm["fake"].as<float>();	
 
 	if (g_mult_st > 1) {
 		g_mult_mf = 1 / g_mult_st;
@@ -1493,10 +1607,15 @@ int run_main(int argc, char **argv)
 	po::options_description exp_desc("Experimental options");
 	exp_desc.add_options()
 		("lambda-2", po::value<float>()->default_value(0.0), "L2 regularization")
+		("l2", po::value<float>()->default_value(0.0), "L2 regularization, v2")
+		("fake", po::value<float>()->default_value(0.0), "")
+		("vi", po::value<float>()->default_value(1.0), "L2 regularization")
 		("neg-reg", po::value<int>()->default_value(0), "")
 		("output,o", po::value<string>()->default_value(""), "output files prefix")
 		("norm", "norm v_out")
 		("views-f", "")
+		("all-items", "")
+		("dw", po::value<float>()->default_value(1.0), "")
 		("sim", "use simmetric views vectors");
 
 	general_desc.add(exp_desc);
@@ -1532,11 +1651,13 @@ int run_main(int argc, char **argv)
 
 	int neg_reg = vm["neg-reg"].as<int>();
 
-	for (int item_num = 0; item_num < (int)item_ids.size(); ++item_num)  {
-		g_train_items.push_back(item_num);
+	if (!vm.count("all-items")) {
+		for (int item_num = 0; item_num < (int)item_ids.size(); ++item_num)  {
+			g_train_items.push_back(item_num);
 
-		for (int i = 0; i < neg_reg; ++i) 
-			g_items_with_counts.push_back(item_num);
+			for (int i = 0; i < neg_reg; ++i) 
+				g_items_with_counts.push_back(item_num);
+		}
 	}
 
 	const string valBasketsFilename = vm["val"].as<string>();
@@ -1545,6 +1666,15 @@ int run_main(int argc, char **argv)
 	if (valBasketsFilename.size()) 
 		ReadBaskets(valBasketsFilename, &valBaskets, &item_ids, &item_nums, &g_new_items);
 
+	if (vm.count("all-items")) {
+		for (int item_num = 0; item_num < (int)item_ids.size(); ++item_num)  {
+			g_train_items.push_back(item_num);
+
+			for (int i = 0; i < neg_reg; ++i) 
+				g_items_with_counts.push_back(item_num);
+		}
+	}
+
 	g_item_count = item_ids.size();
 
 	double norm = 0;
@@ -1590,15 +1720,17 @@ int run_main(int argc, char **argv)
 		}
 	}
 
-	SaveToFile("v_in.txt", v_in, item_ids);
-	SaveToFile("v_out.txt", v_out, item_ids);
+	string output = vm["output"].as<string>();
+
+	SaveToFile(string("v_in") + output + string(".txt"), v_in, item_ids);
+	SaveToFile(string("v_out") + output + string(".txt"), v_out, item_ids);
 
 	if (g_use_features || g_views_features) {
-		SaveToFile("w_in.txt", w_in, token_ids);
-		SaveToFile("w_out.txt", w_out, token_ids);
+		SaveToFile(string("w_in") + output + string(".txt"), w_in, token_ids);
+		SaveToFile(string("w_out") + output + string(".txt"), w_out, token_ids);
 	}
 
-	GetPredictions(item_ids, v_in, v_out);
+	GetPredictions(string("predictions") + output, item_ids, v_in, v_out);
 
 	free_matrix(v_in, g_item_count);
 	free_matrix(v_out, g_item_count);
diff --git a/global.h b/global.h
index ee5c2ee..ae50357 100644
--- a/global.h
+++ b/global.h
@@ -8,7 +8,9 @@ int g_dim = 100;
 int V_IN_CONST_IDX = -1;
 int V_OUT_CONST_IDX = -1;
 double g_lambda_2 = 0;
+double g_l2 = 0;
 int g_views_sample = 20;
+double g_views_importance = 1;
 double g_termination = 1e-3;
 int g_iterations_max = -1;
 //int GAMMA = 0.95;
@@ -34,6 +36,9 @@ long int g_start_time;
 bool g_use_features;
 int g_negative_sampling;
 bool g_norm_v_out;
+double g_dw = 1.0;
+double g_c = 1.0;
+double g_fake_ratio = 0.0;
 
 double **w_in, **w_out, **grad_w_in_sq_sum, **grad_w_out_sq_sum;
 double **w_in2, **w_out2, **grad_w_in_sq_sum2, **grad_w_out_sq_sum2;
