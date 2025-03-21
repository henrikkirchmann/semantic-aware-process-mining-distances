import os
import pickle
from definitions import ROOT_DIR

EVENT_LOGS_DIR = os.path.join(ROOT_DIR , "evaluation", "evaluation_of_activity_distances", "event_logs", "intrinsic_evaluation", "results")

all_logs = ['BPIC12',
            'BPIC12_A',
            'BPIC12_Complete',
            'BPIC12_O',
            'BPIC12_W',
            'BPIC12_W_Complete',
            'BPIC13_closed_problems',
            'BPIC13_incidents',
            'BPIC13_open_problems',
            'BPIC15_1',
            'BPIC15_2',
            'BPIC15_3',
            'BPIC15_4',
            'BPIC15_5',
            'BPIC17',
            'BPIC18',
            'BPIC19',
            'BPIC20_DomesticDeclarations',
            'BPIC20_InternationalDeclarations',
            'BPIC20_PermitLog',
            'BPIC20_PrepaidTravelCost',
            'BPIC20_RequestForPayment',
            'CCC19',
            'Env Permit',
            'Helpdesk',
            'Hospital Billing',
            'RTFM',
            'Sepsis']

log_statistics = {}

window_size_list = [3,5,9]
r = 3
w = 3
sampling_size = 3
df_avg_dir = os.path.join(ROOT_DIR, "results", "activity_distances", "intrinsic_df_avg", "Sepsis")
os.makedirs(df_avg_dir, exist_ok=True)
file_name = f"dfavg_r{r}_w{w}_samplesize_{sampling_size}.pkl"
file_path = os.path.join(df_avg_dir, file_name)

if os.path.isfile(file_path):
    logs_with_replaced_activities_dict = pickle.load(open(file_path, "rb"))
    print("a")
