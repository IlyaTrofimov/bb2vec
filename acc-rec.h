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

namespace po = boost::program_options;

typedef vector<int> Basket;
typedef map<int, vector<pair<int, double> > > cache_type;

#define LIMIT(p) (p > (1.0 - 1e-6) ? (1.0 - 1e-6) : (p < 1e-6 ? 1e-6 : p))
#define GET(map, key, default) (map.find(key) == map.end() ? default : map[key])
#define INCREMENT(map, key) (map.find(key) == map.end() ? map[key] = 1 : map[key] += 1)
#define SQUARE(x) ((x) * (x))

/********************************************************************************************************
*
* An object from browsing sessions (views).
*
********************************************************************************************************/
struct ViewObject {
	int Item1;
	int Item2;
	vector<int> features1;
	vector<int> features2;
	double PMI;
};

void ReadBaskets(const string& fileName, vector<Basket> *baskets, vector<string> *item_ids, map<string, int> *item_nums, set<int> *new_items);
void ReadViews(const string& fileName, vector<string> *item_ids, map<string, int> *item_nums, map<string, int> *token_nums, vector<ViewObject> *views);
void FitMME(const vector<Basket>& baskets, const vector<Basket>& valBaskets, const vector<ViewObject>& views, double ***p_v_in, double ***p_v_out);
void SaveToFile(const char* filename, double **v, const vector<string>& item_ids);
void GetPredictions(const vector<string>& item_ids, double **v_in, double **v_out);
void ReadUnaryCount(const string& fileName, const map<string, int>& item_nums, vector<double> *item_weights);
void free_matrix(double **v, int dim1);
void ReadItemFeatures(const string& fileName, const map<string, int>& item_nums, map<string, int>* token_nums, vector<vector<int> >* item_features);
void ParseOptions(const po::variables_map& vm);
vector<string> Split(const string& line, char separator);
int run_main(int argc, char **argv);
