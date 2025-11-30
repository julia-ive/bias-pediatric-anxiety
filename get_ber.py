import random
import pandas as pd
import numpy as np
import argparse
from statistics import mean
from sklearn.metrics import confusion_matrix


def get_test_sets(args):
    """
    Convert a CSV file into Pandas dataframes, split them based on demographic attributes,
    and create sampled test sets with equal sizes across attributes.

    Parameters with args:
        input_file (str): Path to the CSV file.
        split_dem_field (str): The field in the dataframe on which to split ('race_source_value' or 'gender_source_value').
        demo_atts (list): List of demographic values for the split.
            For 'race_source_value', ['White', 'Black or African American'}
            or 
            For 'gender_source_value', ['Male', 'Female']
        sample_count (int): The number of sampled test sets to generate.
        rnd_seed (int): Random seed for sampling.

    Returns:
        tuple: A tuple containing two dataframes representing the sampled test sets.
    """

    # Read the CSV file and convert it into a DataFrame
    try:
        input_file_path = args.input_file
        test_df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
        exit()

    # Access the parsed arguments
    split_dem_field = args.split_dem_field
    demo_atts = args.demo_atts.split(',')
    sample_count = args.sample_count
    rnd_seed = args.rnd_seed


    len_ll = list()
    case_set_ll = list()
    match_set_ll = list()

    # Create equally sized test sets across demographic groups
    for demo_att in demo_atts:
        
        final_test_case = test_df.loc[(test_df[split_dem_field] == demo_att) & (test_df['labels'] == 1)].reset_index(drop=True)
        final_test_match = test_df.loc[(test_df[split_dem_field] == demo_att) & (test_df['labels'] == 0)].reset_index(drop=True)

        case_set_ll.append(final_test_case)
        match_set_ll.append(final_test_match)

        if len(final_test_case) == 0:
            print(f"The case group size of '{demo_att}' is equal to zero. The sample size will default to 1")
        
        if len(final_test_match) == 0:
            print(f"The control group size of '{demo_att}' is equal to zero. The sample size will default to 1")
        
        
        len_ll.append(len(final_test_case))
        len_ll.append(len(final_test_match))


    # Get the min length set to ensure equal length across groups
    min_len = min(len_ll)

    # We do not sample if any demographic group is not represented
    if min_len == 0:
        sample_count = 1
    
    randomlist = random.sample(range(10, rnd_seed), sample_count)

    sampled_test_set_ll = list()

    for r in randomlist:
        
        if sample_count > 1:
            
            final_test_case1 = case_set_ll[0].sample(n = min_len, random_state=r)
            final_test_match1 = match_set_ll[0].sample(n = min_len, random_state=r)
            
            final_test_case2 = case_set_ll[1].sample(n = min_len, random_state=r)
            final_test_match2 = match_set_ll[1].sample(n = min_len, random_state=r)
        
        else:
            
            final_test_case1 = case_set_ll[0]
            final_test_match1 = match_set_ll[0]
            
            final_test_case2 = case_set_ll[1]
            final_test_match2 = match_set_ll[1]
            
            
        final_test1 = pd.concat([final_test_case1, final_test_match1]).reset_index(drop=True)
        final_test2 = pd.concat([final_test_case2, final_test_match2]).reset_index(drop=True)

        sampled_test_set_ll.append((final_test1, final_test2))

    return sampled_test_set_ll


def get_ber(labels, preds):
    """
    Calculate the BER score based on a list of true labels and predicted labels.

    Parameters:
        labels (list): A list of true (golden) labels.
        preds (list): A list of predicted labels.

    Returns:
        float: The Balanced error rate (BER) score, a value between 0.0 and 1.0.
    """

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    if (fp + tn)!=0 and (fn + tp)!=0:
        
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
    
    else:
        
        print('Invalid BER due to small sample size')
        fpr = 0.0
        fnr = 0.0
    
    ber = (fpr + fnr) / 2
    return ber

def run_ber_test(sampled_test_set_ll):
    """
    Calculate the BER ratio for a pair of dataframes (demo groups).

    Parameters:
        dataframes (tuple): A tuple containing two dataframes.

    Returns:
        float: The BER ratio, a value representing the BER comparison between the two demo groups.
    """

    list_ber_test = [list(), list()]

    for test_tuple in sampled_test_set_ll:

            for i, test_set in enumerate(test_tuple):

                test_labels = test_set['labels']
                test_preds = test_set['prediction']
                ber = get_ber(test_labels, test_preds)
                list_ber_test[i].append(ber)


    # Get averaged ber over the sampled test sets
    # Priviledged class goes first
    ber_pr = mean(list_ber_test[0])
    ber = mean(list_ber_test[1])

    if ber_pr == 0.0:
        print('BER for the privileged class is 0.0')
        return 0.0
    if ber == 0.0:
        print('BER for the non-privileged class is 0.0')
        return 0.0
    
    ber_ratio =  ber/ber_pr
    return ber_ratio


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--input-file', dest='input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('--split-field', dest='split_dem_field', type=str, help='Field for splitting race_source_value or gender_source_value.')
    parser.add_argument('--demo-attributes', dest='demo_atts', type=str, help='Comma-separated demographic values (e.g., "White, Black or African American" or "Male,Female"). Note that privileged class goes first')
    parser.add_argument('--sample-count', dest='sample_count', type=int, help='Number of equally sized test sets to create.')
    parser.add_argument('--random-seed', dest='rnd_seed', type=int, help='Random seed for reproducible sampling.')

    args = parser.parse_args()
    
    sample_test = get_test_sets(args)
    result = run_ber_test(sample_test)
    print(f'BER Ratio: {result}')
    if result >= 1.25:
        print('There is bias towards the privileged class ' + args.demo_atts.split(',')[0])
    elif result <= 0.8:
        print('There is bias towards the unprivileged class ' + args.demo_atts.split(',')[1])
    else:
        print('No selection bias observed.')
        

# example to run the code from command line
# python get_ber.py --input-file in_race.csv --split-field race_source_value  --demo-attributes "White,Black or African American" --sample-count 3  --random-seed 10678
# python get_ber.py --input-file in_gender.csv --split-field gender_source_value  --demo-attributes "Male,Female" --sample-count 3  --random-seed 10678