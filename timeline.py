import pandas as pd
from datetime import datetime
import dateutil 
import argparse
import difflib
import os
import sys
import re

from collections import defaultdict

def convert_datetime(pred_data):
    dates_only_pred_series = pred_data[pred_data.columns[-1]]
    dates_only_pred_df = dates_only_pred_series.to_frame()
    dates_only = list(dates_only_pred_series)
    for date in range(len(dates_only)):
        if "/" in dates_only[date]:
            datetime_obj = datetime.strptime(dates_only[date], '%m/%d/%y')
            dates_only[date] = datetime_obj.strftime('%Y-%m-%d')
    dates_only_df = pd.DataFrame(dates_only, columns = ['timex'])
    pred_data[pred_data.columns[-1]] = dates_only_df
    return pred_data

def gold_file_parse(goldpath):
    gold_df = pd.read_csv(goldpath)
    gold_df['chemo'] = gold_df['chemo'].str.lower()
    select_header = ['chemo', 'relation', 'timex']
    new = gold_df.loc[:, select_header]
    gold = [tuple(x) for x in new.values.tolist()]
    return goldpath, gold

def preds_file_parse(predpath):
    preds_df = pd.read_csv(predpath,
                       names=['chemo', 'relation', 'timex'])
    preds_new = convert_datetime(preds_df)
    select_header = ['chemo', 'relation', 'timex']
    preds_df = preds_new.loc[:, select_header]
    preds = [tuple(x) for x in preds_df.values.tolist()]
    return predpath, preds
    
def fix_neg_strict(true_pos_list, false_list, date_flag, spell_flag):
    """
    :param date_flag: for strict, date_flag is automatically 'year-month-day'
    :param spell_flag: flag to fix misspellings (if they exist)
    @fix_neg_strict description: Strict evaluation only allows for checking predictions with CONTAINS-1 relations and STRICT predictions must match the gold data exactly.
    """
    year_month_day = False
    if date_flag == 'year-month-day':
        year_month_day = True
    timeline_dict = {}
    for chemo in true_pos_list:
        if chemo[0] not in timeline_dict:
            timeline_dict[chemo[0]] = [chemo]
        else:
            timeline_dict[chemo[0]].append(chemo)
    begin_end_dict = {}
    false_to_remove = []
    false_contains_only = [pred for pred in false_list if pred[1] == 'CONTAINS-1']
    for pred in false_contains_only:
        if spell_flag:    
            timeline_chemos = list(timeline_dict.keys())
            matches = difflib.get_close_matches(pred[0], timeline_chemos)
            if len(matches) > 0: 
                chemo_match = matches[0]
            else:
                chemo_match = matches
        else:
            chemo_match = pred[0]
        if chemo_match in timeline_dict:
            for date in timeline_dict[chemo_match]:
                if date[1] == 'BEGINS-ON':
                    if chemo_match not in begin_end_dict:
                        begin_end_dict[chemo_match] = [date]
                    else:
                        begin_end_dict[chemo_match].append(date)
                elif date[1] == 'ENDS-ON':
                    if chemo_match not in begin_end_dict:
                        begin_end_dict[chemo_match] = [date]
                    else:
                        begin_end_dict[chemo_match].append(date)
            begin_count = set()
            end_count = set()
            begin_date = None
            end_date = None
            for relation in begin_end_dict[chemo_match]:
                end_relation = defaultdict(int)
                begin_relation = defaultdict(int)
                if relation[1] == 'BEGINS-ON':
                    begin_date = relation
                    begin_count.add(relation)
                if relation[1] == 'ENDS-ON':
                    end_date = relation
                    end_count.add(relation)
            if begin_date != None and len(begin_count) > 1:
                for relation in begin_end_dict[chemo_match]:
                    if relation[1] == 'BEGINS-ON':
                        begin_relation[relation] = 0
                begin_keys = list(begin_relation.keys())
                for i, key in enumerate(begin_keys):
                    if len(key[2]) == 4:
                        add_date = key[2] + '-01-01'
                    elif len(key[2]) == 7:
                        add_date = key[2] + '-01'
                    else:
                        add_date = key[2]
                    begin_keys[i] = dateutil.parser.parse(add_date)
                for key in begin_relation:
                    if end_date != None and key[2] != end_date[2]:
                        begin_relation[key] += 1
                    if len(key[2]) == 4:
                        add_date = key[2] + '-01-01'
                    elif len(key[2]) == 7:
                        add_date = key[2] + '-01'
                    else:
                        add_date = key[2]
                    if dateutil.parser.parse(add_date) == max(begin_keys):
                        begin_relation[key] += 1
                begin_date = max(begin_relation)
            if end_date != None and len(end_count) > 1:
                for relation in begin_end_dict[chemo_match]:
                    if relation[1] == 'ENDS-ON':
                        end_relation[relation] = 0
                end_keys = list(end_relation.keys())
                for i, key in enumerate(end_keys):
                    if len(key[2]) == 4:
                        add_date = key[2] + '-01-01'
                    elif len(key[2]) == 7:
                        add_date = key[2] + '-01'
                    else:
                        add_date = key[2]
                    end_keys[i] = dateutil.parser.parse(key[2])
                for key in end_relation:
                    if begin_date != None and key[2] != begin_date[2]:
                        end_relation[key] +=1
                    if len(key[2]) == 4:
                        add_date = key[2] + '-01-01'
                    elif len(key[2]) == 7:
                        add_date = key[2] + '-01'
                    else:
                        add_date = key[2]
                    if dateutil.parser.parse(add_date) == max(end_keys):
                        end_relation[key] += 1
                end_date = max(end_relation)
            # convert strings to dates
            if len(pred[2]) == 4: 
                add_date = pred[2] + '-01-01'
                pred_contains_date = dateutil.parser.parse(add_date)
                if begin_date != None:
                    if len(begin_date[2]) == 4:
                        add_date = begin_date[2] + '-01-01'
                        gold_begins_date = dateutil.parser.parse(add_date)
                    elif len(begin_date[2]) == 7:
                        add_date = begin_date[2] + '-01'
                        gold_begins_date = dateutil.parser.parse(add_date)
                    else:
                        gold_begins_date = dateutil.parser.parse(begin_date[2])
                if end_date != None:
                    if len(end_date[2]) == 4:
                        add_date = end_date[2] + '-01-01'
                        gold_end_date = dateutil.parser.parse(add_date)
                    elif len(end_date[2]) == 7:
                        add_date = end_date[2] + '-01'
                        gold_end_date = dateutil.parser.parse(add_date)
                    else:
                        gold_end_date = dateutil.parser.parse(end_date[2])
            elif len(pred[2]) == 7:
                add_date = pred[2] + '-01'
                pred_contains_date = dateutil.parser.parse(add_date)
                if begin_date != None:
                    if len(begin_date[2]) == 4:
                        add_date = begin_date[2] + '-01-01'
                        gold_begins_date = dateutil.parser.parse(add_date)
                    elif len(begin_date[2]) == 7:
                        add_date = begin_date[2] + '-01'
                        gold_begins_date = dateutil.parser.parse(add_date)
                    else:
                        gold_begins_date = dateutil.parser.parse(begin_date[2])
                if end_date != None:
                    if len(end_date[2]) == 4:
                        add_date = end_date[2] + '-01-01'
                        gold_end_date = dateutil.parser.parse(add_date)
                    elif len(end_date[2]) == 7:
                        add_date = end_date[2] + '-01'
                        gold_end_date = dateutil.parser.parse(add_date)
                    else:
                        gold_end_date = dateutil.parser.parse(end_date[2])
            else: 
                pred_contains_date = dateutil.parser.parse(pred[2])
                if begin_date != None:
                    if len(begin_date[2]) == 4:
                        add_date = begin_date[2] + '-01-01'
                        gold_begins_date = dateutil.parser.parse(add_date)
                    elif len(begin_date[2]) == 7:
                        add_date = begin_date[2] + '-01'
                        gold_begins_date = dateutil.parser.parse(add_date)
                    else:
                        gold_begins_date = dateutil.parser.parse(begin_date[2])
                if end_date != None:
                    if len(end_date[2]) == 4:
                        add_date = end_date[2] + '-01-01'
                        gold_end_date = dateutil.parser.parse(add_date)
                    elif len(end_date[2]) == 7:
                        add_date = end_date[2] + '-01'
                        gold_end_date = dateutil.parser.parse(add_date)
                    else:
                        gold_end_date = dateutil.parser.parse(end_date[2])
            # start date checking
            if begin_date != None and end_date == None:
                if gold_begins_date == pred_contains_date:
                    true_pos_list.append(pred)
                    if pred in false_list:
                        false_to_remove.append(pred)
            elif begin_date != None and end_date != None:
                if gold_begins_date == pred_contains_date or gold_end_date == pred_contains_date: 
                    true_pos_list.append(pred)
                    if pred in false_list:
                        false_to_remove.append(pred)
            elif begin_date == None and end_date != None: 
                if gold_end_date == pred_contains_date: 
                    true_pos_list.append(pred)
                    if pred in false_list:
                        false_to_remove.append(pred)
    false_list = [item for item in false_list if item not in false_to_remove]
    return true_pos_list, false_list, false_to_remove

def fix_neg_relaxed(true_pos_list, false_list, date_flag, spell_flag, gold):

    """
    :param date_flag: flag for deciding whether to evaluate predictions only by year, only by year and month, or the entire date (y-m-d)
    :param spell_flag: flag to fix misspellings (if they exist)
    @fix_neg_relaxed description: Relaxed evaluation only allows for checking predictions with CONTAINS-1, BEGINS-ON, and ENDS-ON relations,
    as well as allowing predicted CONTAINS-1 to match with gold BEGINS-ON and ENDS-ON if the date is within the timeline range.
    """
    only_year = False
    only_year_month = False
    year_month_day = False
    if date_flag == 'year':
        only_year = True
    elif date_flag == 'year-month':
        only_year_month = True
    elif date_flag == 'year-month-day':
        year_month_day = True
    timeline_dict = {}
    for chemo in true_pos_list:
        if chemo[0] not in timeline_dict:
            timeline_dict[chemo[0]] = [chemo]
        else:
            timeline_dict[chemo[0]].append(chemo)
    begin_end_dict = {}
    false_to_remove = []
    for pred in false_list:
        if spell_flag:    
            timeline_chemos = list(timeline_dict.keys())
            matches = difflib.get_close_matches(pred[0], timeline_chemos)
            if len(matches) > 0: 
                chemo_match = matches[0]
            else:
                chemo_match = matches
        else:
            chemo_match = pred[0]
        if chemo_match in timeline_dict: 
            for date in timeline_dict[chemo_match]:
                if date[1] == 'BEGINS-ON':
                    if chemo_match not in begin_end_dict:
                        begin_end_dict[chemo_match] = [date]
                    else:
                        begin_end_dict[chemo_match].append(date)
                elif date[1] == 'ENDS-ON':
                    if chemo_match not in begin_end_dict:
                        begin_end_dict[chemo_match] = [date]
                    else:
                        begin_end_dict[chemo_match].append(date)
                pred_contains_date = dateutil.parser.parse(pred[2])
            begin_count = set()
            end_count = set()
            begin_date = None
            end_date = None
            for relation in begin_end_dict[chemo_match]:
                end_relation = defaultdict(int)
                begin_relation = defaultdict(int)
                if relation[1] == 'BEGINS-ON':
                    begin_date = relation
                    begin_count.add(relation)
                if relation[1] == 'ENDS-ON':
                    end_date = relation
                    end_count.add(relation)
            if begin_date != None and len(begin_count) > 1:
                for relation in begin_end_dict[chemo_match]:
                    if relation[1] == 'BEGINS-ON':
                        begin_relation[relation] = 0
                begin_keys = list(begin_relation.keys())
                for i, key in enumerate(begin_keys):
                    if len(key[2]) == 4:
                        add_date = key[2] + '-01-01'
                    elif len(key[2]) == 7:
                        add_date = key[2] + '-01'
                    else:
                        add_date = key[2]
                    begin_keys[i] = dateutil.parser.parse(add_date)
                for key in begin_relation:
                    if len(key[2]) == 4:
                        add_date = key[2] + '-01-01'
                    elif len(key[2]) == 7:
                        add_date = key[2] + '-01'
                    else:
                        add_date = key[2]
                    if key in gold:
                        begin_relation[key] += 1
                    if dateutil.parser.parse(add_date) == max(begin_keys):
                        begin_relation[key] += 1
                begin_date = max(begin_relation)
            if end_date != None and len(end_count) > 1:
                for relation in begin_end_dict[chemo_match]:
                    if relation[1] == 'ENDS-ON':
                        end_relation[relation] = 0
                end_keys = list(end_relation.keys())
                for i, key in enumerate(end_keys):
                    if len(key[2]) == 4:
                        add_date = key[2] + '-01-01'
                    elif len(key[2]) == 7:
                        add_date = key[2] + '-01'
                    else:
                        add_date = key[2]
                    end_keys[i] = dateutil.parser.parse(key[2])
                for key in end_relation:
                    if len(key[2]) == 4:
                        add_date = key[2] + '-01-01'
                    elif len(key[2]) == 7:
                        add_date = key[2] + '-01'
                    else:
                        add_date = key[2]
                    if key in gold:
                        end_relation[key] += 1
                    if dateutil.parser.parse(add_date) == max(end_keys):
                        end_relation[key] += 1
                end_date = max(end_relation)
            #flags to parse dates
            if only_year is True: # only match years
                pred_contains_date = dateutil.parser.parse(pred[2]).year
                if begin_date != None:
                    gold_begins_date = dateutil.parser.parse(begin_date[2]).year
                if end_date != None:
                    gold_end_date = dateutil.parser.parse(end_date[2]).year
            elif only_year_month is True:
                pred_year = dateutil.parser.parse(pred[2]).year
                if len(pred[2]) == 4: # pred[2] is only year
                    pred_contains_date = 1
                else:
                    pred_contains_date = dateutil.parser.parse(pred[2]).month
                if begin_date != None:
                    gold_begins_year = dateutil.parser.parse(begin_date[2]).year
                    if len(begin_date[2]) == 4:
                        gold_begins_date = 1
                    else:
                        gold_begins_date = dateutil.parser.parse(begin_date[2]).month
                if end_date != None:
                    gold_end_year = dateutil.parser.parse(end_date[2]).year
                    if len(end_date[2]) == 4:
                        gold_end_date = 1
                    else:
                        gold_end_date = dateutil.parser.parse(end_date[2]).month
            elif year_month_day is True:
                if len(pred[2]) == 4:
                    add_date = pred[2] + '-01-01'
                    pred_contains_date = dateutil.parser.parse(add_date)
                elif len(pred[2]) == 7:
                    add_date = pred[2] + '-01'
                    pred_contains_date = dateutil.parser.parse(add_date)
                else:
                    pred_contains_date = dateutil.parser.parse(pred[2])
                if begin_date != None:
                    if len(begin_date[2]) == 4:
                        add_date = begin_date[2] + '-01-01'
                        gold_begins_date = dateutil.parser.parse(add_date)
                    elif len(begin_date[2]) == 7:
                        add_date = begin_date[2] + '-01'
                        gold_begins_date = dateutil.parser.parse(add_date)
                    else:
                        gold_begins_date = dateutil.parser.parse(begin_date[2])
                if end_date != None:
                    if len(end_date[2]) == 4:
                        add_date = end_date[2] + '-01-01'
                        gold_end_date = dateutil.parser.parse(add_date)
                    elif len(end_date[2]) == 7:
                        add_date = end_date[2] + '-01'
                        gold_end_date = dateutil.parser.parse(add_date)
                    else:
                        gold_end_date = dateutil.parser.parse(end_date[2])
            # evaluate dates
            if begin_date != None and end_date == None: 
                if only_year_month:
                    if gold_begins_year == pred_year:
                        if gold_begins_date == pred_contains_date:
                            true_pos_list.append(pred)
                            if pred in false_list:
                                false_to_remove.append(pred)
                else:
                    if year_month_day:
                        if pred[1] == 'ENDS-ON' and pred_contains_date == gold_begins_date:
                            continue
                    if gold_begins_date == pred_contains_date:
                        true_pos_list.append(pred)
                        if pred in false_list:
                            false_to_remove.append(pred)
            elif begin_date != None and end_date != None: 
                if only_year_month:
                    if gold_begins_year <= pred_year and gold_end_year >= pred_year:  
                        if gold_begins_date <= pred_contains_date and gold_end_date >= pred_contains_date:
                            true_pos_list.append(pred)
                            if pred in false_list:
                                false_to_remove.append(pred)
                else:
                    if year_month_day:
                        if pred[1] == 'ENDS-ON' and pred_contains_date == gold_begins_date:
                            continue
                        elif pred[1] == 'BEGINS-ON' and pred_contains_date == gold_end_date:
                            continue
                    if gold_begins_date <= pred_contains_date and gold_end_date >= pred_contains_date: 
                        true_pos_list.append(pred)
                        if pred in false_list:
                            false_to_remove.append(pred)
            elif begin_date == None and end_date != None: 
                if only_year_month:
                    if gold_end_year == pred_year:
                        if gold_end_date == pred_contains_date:
                            true_pos_list.append(pred)
                            if pred in false_list:
                                false_to_remove.append(pred)
                else:
                    if year_month_day:
                        if pred[1] == 'BEGINS-ON' and pred_contains_date == gold_end_date:
                            continue
                    if gold_end_date == pred_contains_date:
                        true_pos_list.append(pred)
                        if pred in false_list:
                            false_to_remove.append(pred)
    false_list = [item for item in false_list if item not in false_to_remove]
    return true_pos_list, false_list, false_to_remove

def fix_false_neg(true_pos_list, false_neg_list, false_pos_list, date_flag):

    """
    :param date_flag: flag for deciding whether to evaluate predictions only by year, only by year and month, or the entire date (y-m-d)
    :param spell_flag: flag to fix misspellings (if they exist)
    @fix_false_neg: Fixes false negatives that fix_neg_relaxed doesn't catch by comparing chemo and time expression in gold that is matches false negatives,
    and also fixes false positives and false negatives if they match
    """
    true_chemo_date = []
    false_neg_remove = []
    false_pos_remove = []
    
    if date_flag == 'year':
        for gold in true_pos_list:
            gold_y = dateutil.parser.parse(gold[2]).year
            true_chemo_date.append((gold[0], gold_y))
        for false in false_neg_list:
            year = dateutil.parser.parse(false[2]).year
            if (false[0], year) in true_chemo_date:
                false_neg_remove.append(false)
                true_pos_list.append(false)
            else:
                for false_pos in false_pos_list:
                    fp_year = dateutil.parser.parse(false_pos[2]).year
                    if (false_pos[0], fp_year) == (false[0], year):
                        false_pos_remove.append(false_pos)
                        false_neg_remove.append(false)
                        true_pos_list.append(false)
                        true_pos_list.append(false_pos)
            
    elif date_flag == 'year-month':
        for gold in true_pos_list:
            if len(gold[2]) > 4:
                gold_date = gold[2][0:7]
            else:
                gold_date = gold[2]
            true_chemo_date.append((gold[0], gold_date))
        for false in false_neg_list:
            if len(false[2]) > 4:
                fn_date = false[2][0:7]
            else:
                fn_date = false[2]
            if (false[0], fn_date) in true_chemo_date:
                false_neg_remove.append(false)
                true_pos_list.append(false)
            else:
                for false_pos in false_pos_list:
                    if len(false_pos[2]) > 4:
                        fp_date = false_pos[2][0:7]
                    else:
                        fp_date = false_pos[2]
                    if (false_pos[0], fp_date) == (false[0], fn_date):
                        false_pos_remove.append(false_pos)
                        false_neg_remove.append(false)
                        true_pos_list.append(false_pos)
                        true_pos_list.append(false)
            
    elif date_flag == 'year-month-day':
        for gold in true_pos_list:
            true_chemo_date.append((gold[0], gold[2]))
        for false in false_neg_list:
            if (false[0], false[2]) in true_chemo_date:
                false_neg_remove.append(false)
                true_pos_list.append(false)
            else:
                for false_pos in false_pos_list:
                    if len(false_pos[2]) == 10:
                        if (false_pos[0], false_pos[2]) == (false[0], false[2]):
                            false_pos_remove.append(false_pos)
                            false_neg_remove.append(false)
                            true_pos_list.append(false_pos)
                            true_pos_list.append(false)

    false_neg_list = [item for item in false_neg_list if item not in false_neg_remove]
    false_pos_list = [item for item in false_pos_list if item not in false_pos_remove]
    return true_pos_list, false_neg_list, false_pos_list

def confusion_matrix(args, output_path, goldfile, predfile, gold, preds, flag_strict, flag_date, flag_spell):
    save_path = output_path + "/eval_"
    i = 0
    if flag_strict:
        print_date = "day"
    else:
        print_date = str(args[9])
    print_patient = re.search(r'[a-z]+_pt\d+_[a-z]+', args[1]).group(0)
    print_strict = "_" + str(args[7])
    if flag_spell:
        print_spell = "spelling"
    else:
        print_spell = ""
    out_file = save_path + print_patient + print_strict + print_date + print_spell + ".txt"
      
    output = open(out_file, "w+")
    output.writelines(" ".join(args) + "\n")
    output.writelines("\n")
    output.writelines("gold file name: " + goldfile[86:] + "\n")
    output.writelines("pred file name: " + predfile[97:] + "\n")
    output.writelines("\n")

    true_pos = [prediction for prediction in preds if prediction in gold]
    false_pos = [prediction for prediction in preds if prediction not in gold]
    false_neg = [correct for correct in gold if correct not in preds]
    
    if flag_strict:
        true_pos, false_pos, false_remove = fix_neg_strict(true_pos, false_pos, flag_date, flag_spell)
    else:
        true_pos, false_pos, false_remove = fix_neg_relaxed(true_pos, false_pos, flag_date, flag_spell, gold)
        true_pos, false_neg, false_remove = fix_neg_relaxed(true_pos, false_neg, flag_date, flag_spell, gold)
        true_pos, false_neg, false_pos = fix_false_neg(true_pos, false_neg, false_pos, flag_date)

    true_pos_num = "# of TRUE_POSITIVES: " + str(len(true_pos)) + ", " + str(true_pos)
    false_pos_num = "# of FALSE_POSITIVES: " + str(len(false_pos)) + ", " + str(false_pos)
    false_neg_num = "# of FALSE_NEGATIVES: " + str(len(false_neg)) + ", " +  str(false_neg)

    output.writelines(true_pos_num + "\n")
    output.writelines(false_pos_num + "\n")
    output.writelines(false_neg_num + "\n")
    output.writelines("\n")

    precision = len(true_pos)/(len(true_pos)+len(false_pos))
    recall = len(true_pos)/(len(true_pos)+len(false_neg))
    f1 = 2 * (precision*recall) / (precision+recall)

    output.writelines(("Precision: ", str(precision)+ "\n"))
    output.writelines(("Recall: ", str(recall)+ "\n"))
    output.writelines(("F1: ", str(f1)+ "\n"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate predicted output against gold annotations")
    
    parser.add_argument('--goldpath', dest="gold_output", required=True, type=gold_file_parse,
                        help="A gold annotations CSV file to use for evaluations to compare against predicted", metavar="FILE")

    parser.add_argument('--predpath', dest="pred_output", required=True, type=preds_file_parse,
                        help="A predicted output CSV file to use for evaluations to compare against gold", metavar="FILE")
    
    parser.add_argument('--output', dest="output_path", required=True,
                        help="Write the folder to where the output directory should be", metavar="DIRECTORY")
    parser.add_argument('--strict', dest="strict_flag", required=True, nargs='?', choices=['strict', 'relaxed'],
                        help="Type 'strict' to do strict eval or 'relaxed' to do relaxed eval (contains and end dates can match in relaxed)", metavar="STRICT")
    parser.add_argument('--date', dest="eval_date", nargs='?', choices=['year', 'month', 'day'],
                        help="Type 'year' to only evaluate year, 'month' to evaluate year and month, or 'day' to evaluate entire year-month-day", metavar="DATE")
    parser.add_argument('--spell', dest="spelling", required=True, nargs='?', choices=['yes', 'no'],
                        help="Type 'yes' if you want to fix misspellings (if available) in dataset, 'no' otherwise", metavar="SPELL")

    args = parser.parse_args()
    args_used = sys.argv[1:]
    goldfile, gold = args.gold_output
    predfile, preds = args.pred_output
    output_path = args.output_path 
    strict_flag = args.strict_flag
    eval_date = args.eval_date
    spelling = args.spelling

    if strict_flag == 'strict':
        is_strict = True
        date_flag = 'year-month-day'
    else:
        is_strict = False
        if eval_date == 'year':
            date_flag = 'year'
        elif eval_date == 'month':
            date_flag = 'year-month'
        elif eval_date == 'day':
            date_flag = 'year-month-day'
        
    if spelling == 'yes':
        spell_flag = True
    else:
        spell_flag = False
    
    confusion_matrix(args_used, output_path, goldfile, predfile, gold, preds, is_strict, date_flag, spell_flag)
    
