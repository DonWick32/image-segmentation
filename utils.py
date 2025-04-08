import os
import wandb
from evaluate import run_eval
import traceback

def logger(config, dict, wandb_log=True):
    if wandb_log:
        try:
            wandb.log(dict)
        except Exception as e:
            print(f"Error logging to wandb: {e}")
            print("Logging to wandb failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Traceback:", traceback.format_exc())
        
    log_path = os.path.join(config.output_dir, wandb.run.id, "log.txt")
    with open(log_path, "a") as f:
        for key, value in dict.items():
            if key != "epoch":
                f.write(f"{key}: {value}\n")
        f.write("\n")


def rm_output_keys(output):
    for o_idx in range(len(output)):
        for k in list(output[o_idx].keys()):
            if k not in ["multistep_pred_multimasks_high_res", "multistep_pred_ious", "multistep_object_score_logits"]:
                del output[o_idx][k]


DOMAINS = ['regular', 'blood', 'bg_change', 'smoke', 'low_brightness']

def train_val_eval(val_performance, train_performance, model, domain_idx, config, monitor_vids, domain):
    for perf_list, type_ in zip([val_performance, train_performance], ['val', 'train']):
        print(f"Evaluating {type_} performance from current domain {domain}")
        perf_total = {}
        for domain_prev in DOMAINS[:domain_idx+1]:
            print(f"Evaluating prev domain: {domain_prev} performance")
            annot_file = "val" if type_ == "train" else "test"
            perf = run_eval(model.module, monitor_vids[type_], domain, os.path.join(config.dataset.annotation_path, f"{annot_file}.json"))
            perf_total[domain_prev] = perf
            for k, v in perf.items():
                logger(config, {f"{type_}_perf/{domain_prev}/{k}": v})
            print(f"Performance of {domain_prev} domain: {perf}")
        insert_perf(perf_list, perf_total)
        calculate_forgetting(perf_list, domain_idx, config, tag=type_)

def insert_perf(perf_dict, new_perf):
    for key in new_perf.keys():
        perf_dict[key].append(new_perf[key])

def calculate_forgetting(perf_dict, domain_idx, config, tag="train"):
    for i, domain in enumerate(DOMAINS[:domain_idx]):
        print("Domain performance history of domain", domain, perf_dict[domain])
        if type(perf_dict[domain][-1]) == list:  
            for metric in perf_dict[domain][-1][-1].keys():
                avg_forgetting = 0
                avg_forgetting_prev = 0
                for idx in range(2):
                    f = perf_dict[domain][-1][idx][metric] - perf_dict[domain][0][idx][metric]
                    f_prev = perf_dict[domain][-1][idx][metric] - perf_dict[domain][-2][idx][metric]
                    avg_forgetting += f
                    avg_forgetting_prev += f_prev
                    logger(config, {f"metric/{tag}_forgetting_{domain}_video_{idx}_{metric}": f})
                    print("Video", idx, "Metric", metric, "of domain", domain, ":", perf_dict[domain][-1][idx][metric])
                    print("Metric", metric, "of domain", domain, ":", perf_dict[domain][-1][idx][metric])
                    print("Forgetting of domain", domain, ":", f)
                avg_forgetting /= 3
                avg_forgetting_prev /= 3
                print("Average forgetting:", avg_forgetting)
                print("Average forgetting previous:", avg_forgetting_prev)
                logger(config, {f"metric/test_avg_forgetting_{domain}_{metric}": avg_forgetting})
                logger(config, {f"metric/test_avg_forgetting_prev_{domain}_{metric}": avg_forgetting_prev})
                
        else:
            for metric in perf_dict[domain][-1].keys():
                f = perf_dict[domain][-1][metric] - perf_dict[domain][0][metric]
                f_prev = perf_dict[domain][-1][metric] - perf_dict[domain][-2][metric]
                logger(config, {f"metric/{tag}_forgetting_{domain}_{metric}": f})
                print("Metric", metric, "of domain", domain, ":", perf_dict[domain][-1][metric])
                print("Forgetting of domain", domain, ":", f)