
#Constants
RESULTS = 'results4'
LOG = '.log'


#Configuration
samplers = ['random']
models = ['log_reg']
batch_size = [20]
warmstart_size = [20]
seeds = list(range(2))
checkpoint = './checkpoint.csv'
grid_search_step = -1
workers = 6
cpus = list(range(56, 62))
#datapath = constants.DATA + 'ts_10000.pkl'
#dataset_name = 'toxic_speech'
verbose = False
STORAGE_PATH = constants.RESULTS

