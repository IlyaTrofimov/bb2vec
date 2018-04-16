vector<int> g_items_with_counts;
vector<int> g_train_items;

int g_item_count = -1;
int g_neg_count = -1;
double g_eta = 0.05;
int g_dim = 100;
int V_IN_CONST_IDX = -1;
int V_OUT_CONST_IDX = -1;
double g_lambda_2 = 0;
double g_l2 = 0;
int g_views_sample = 20;
double g_views_importance = 1;
double g_termination = 1e-3;
int g_iterations_max = -1;
//int GAMMA = 0.95;
bool g_simmetric_views = false;
vector<double> g_item_weights;
vector<double> g_scores;
bool g_rank;
int g_stop_count;
vector<vector<int> > item_features;
vector<string> token_ids;
map<string, int> token_nums;
int g_neg_reg = -1;
double g_mult_st = 1.0;
double g_mult_mf = 1.0;
bool g_views_features = false;

set<int> g_new_items;
bool g_recommend_new = false;
int g_threads = 0;
bool g_stop_threads = false;
vector<std::default_random_engine> g_generators;
long int g_start_time;
bool g_use_features;
int g_negative_sampling;
bool g_norm_v_out;
double g_dw = 1.0;
double g_c = 1.0;
double g_fake_ratio = 0.0;
bool g_naive;
float g_ni = 0;

double **w_in, **w_out, **grad_w_in_sq_sum, **grad_w_out_sq_sum;
double **w_in2, **w_out2, **grad_w_in_sq_sum2, **grad_w_out_sq_sum2;

int g_time = 0;
vector<int> v_in_ts, v_out_ts, v_in2_ts, v_out2_ts;
